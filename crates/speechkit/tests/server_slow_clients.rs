//! Slow and stalled readers, over real sockets with small receive buffers.
//! In-memory tests (`oneshot`) never exercise hyper's write path, which is
//! where a reader that stops reading pushes back. The header and body
//! timeouts, for slow writers, are tested in `src/server/tests.rs`, since
//! they have no public setters.
#![cfg(feature = "server")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    net::SocketAddr,
    time::{Duration, Instant},
};

use serde_json::json;
use speechkit::{
    AudioBuffer, SampleRate,
    asr::AsrEngine,
    audio::encode_wav,
    server::{RunningServer, Server},
    tts::TtsEngine,
};
use speechkit_testkit::{
    asr::{FakeAsr, Script, Step, Trigger},
    contract::asr::tone,
    tts::FakeTts,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

const BOUNDARY: &str = "slow";

/// A client with a 4 KiB receive buffer, so the server's writes back up
/// after a few kilobytes it does not read.
async fn connect(address: SocketAddr) -> TcpStream {
    let socket = tokio::net::TcpSocket::new_v4().unwrap();
    socket.set_recv_buffer_size(4096).unwrap();
    socket.connect(address).await.unwrap()
}

fn request(path: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nhost: x\r\ncontent-type: {content_type}\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(body);
    request
}

/// About 2 MB of synthesized PCM.
fn long_speech() -> Vec<u8> {
    let body = json!({"model": "tts-1", "input": "word. ".repeat(680), "voice": "alpha",
        "response_format": "pcm"});
    request(
        "/v1/audio/speech",
        "application/json",
        body.to_string().as_bytes(),
    )
}

fn short_speech() -> Vec<u8> {
    let body = json!({"model": "tts-1", "input": "hi", "voice": "alpha"});
    request(
        "/v1/audio/speech",
        "application/json",
        body.to_string().as_bytes(),
    )
}

fn transcription(file: &[u8], stream: bool) -> Vec<u8> {
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"stream\"\r\n\r\n{stream}\r\n\
         --{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    request(
        "/v1/audio/transcriptions",
        &format!("multipart/form-data; boundary={BOUNDARY}"),
        &body,
    )
}

/// Sends `request` and reads the whole response at full speed, returning
/// its status code.
async fn exchange(address: SocketAddr, request: &[u8]) -> u16 {
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream.write_all(request).await.unwrap();
    let mut response = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), stream.read_to_end(&mut response))
        .await
        .expect("a prompt client waited over 10 s")
        .unwrap();
    String::from_utf8_lossy(&response)
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or(0)
}

/// Sends `request` until it is answered with `wanted`, for up to 10 s,
/// checking that it is refused as busy (503) at first.
async fn until_answered(address: SocketAddr, request: &[u8], wanted: u16) -> Duration {
    let started = Instant::now();
    assert_eq!(
        exchange(address, request).await,
        503,
        "the slow clients were not holding the server's capacity"
    );
    loop {
        let code = exchange(address, request).await;
        if code == wanted {
            return started.elapsed();
        }
        assert_eq!(code, 503, "answered neither {wanted} nor busy");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "still answered {code}, not {wanted}, after 10 s"
        );
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Reads `stream` `chunk` bytes every `gap`, until the server ends the
/// response or `limit` passes. Returns the bytes read and whether the
/// server ended it.
async fn read_slowly(
    mut stream: TcpStream,
    chunk: usize,
    gap: Duration,
    limit: Duration,
) -> (usize, bool) {
    let started = Instant::now();
    let mut buffer = vec![0; chunk];
    let mut total = 0;
    while started.elapsed() < limit {
        match tokio::time::timeout(gap, stream.read(&mut buffer)).await {
            Ok(Ok(0) | Err(_)) => return (total, true),
            Ok(Ok(read)) => total += read,
            Err(_) => {}
        }
        tokio::time::sleep(gap).await;
    }
    (total, false)
}

/// Waits up to 5 s for `active` to reach `count`.
async fn holding(count: usize, active: impl Fn() -> usize) {
    let started = Instant::now();
    while active() < count {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "only {} of {count} sessions started",
            active()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

async fn start(server: Server, timeout: Duration) -> RunningServer {
    server
        .with_bind("127.0.0.1:0")
        .with_timeout(timeout)
        .start(std::future::pending())
        .await
        .unwrap()
}

/// A client reading synthesized audio at 10 kB/s, far slower than it is
/// made, cannot keep the only synthesis slot past the request's deadline,
/// and its response ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_reader_of_speech_frees_its_slot_at_the_deadline() {
    let tts = TtsEngine::new(FakeTts::plain()).with_max_sessions(1);
    let running = start(
        Server::new().with_tts(tts.clone(), "fake-tts"),
        Duration::from_secs(1),
    )
    .await;
    let address = running.local_addr();
    let mut slow = connect(address).await;
    slow.write_all(&long_speech()).await.unwrap();
    let reader = tokio::spawn(read_slowly(
        slow,
        4096,
        Duration::from_millis(20),
        Duration::from_secs(20),
    ));
    holding(1, || tts.active_sessions()).await;

    let waited = until_answered(address, &short_speech(), 200).await;
    assert!(
        waited < Duration::from_secs(5),
        "the slot was free after {waited:?}"
    );
    // Whether the reader then got all of it depends on how much the
    // kernel buffered before the deadline: on Linux, often everything.
    let (_, ended) = reader.await.unwrap();
    assert!(ended, "the slow response was still going after 20 s");
    assert_eq!(tts.active_sessions(), 0);
    running.shutdown().await.unwrap();
}

/// A streamed transcription read slowly frees its admission at the
/// deadline, so the next upload is decoded (and refused as bad audio)
/// rather than turned away with 503.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_reader_of_a_streamed_transcription_frees_its_admission() {
    // Far more events than the socket and the event channel hold.
    let mut script = Script::new();
    for utterance in 0..3_000 {
        script = script.then(
            Trigger::AfterSamples(utterance * 100 + 1),
            Step::Segment(utterance, LONG_WORD),
        );
    }
    let asr = AsrEngine::new(FakeAsr::new(script)).with_max_sessions(1);
    let running = start(
        Server::new().with_asr(asr.clone(), "fake-asr"),
        Duration::from_secs(1),
    )
    .await;
    let address = running.local_addr();
    let audio = AudioBuffer::new(SampleRate::HZ_16000, tone(16_000 * 19));
    let mut slow = connect(address).await;
    slow.write_all(&transcription(&encode_wav(&audio).unwrap(), true))
        .await
        .unwrap();
    let reader = tokio::spawn(read_slowly(
        slow,
        256,
        Duration::from_millis(50),
        Duration::from_secs(20),
    ));
    holding(1, || asr.active_sessions()).await;

    let waited = until_answered(address, &transcription(b"not audio", false), 400).await;
    assert!(waited < Duration::from_secs(5), "admitted after {waited:?}");
    reader.abort();
    running.shutdown().await.unwrap();
}

/// Readers that stop reading, more of them than there are slots, hold up
/// neither the health check nor, once their deadlines pass, new work.
#[tokio::test(flavor = "multi_thread")]
async fn stalled_readers_do_not_hold_up_other_clients() {
    let tts = TtsEngine::new(FakeTts::plain()).with_max_sessions(2);
    let running = start(
        Server::new().with_tts(tts.clone(), "fake-tts"),
        Duration::from_secs(1),
    )
    .await;
    let address = running.local_addr();
    let mut stalled = Vec::new();
    for _ in 0..6 {
        let mut stream = connect(address).await;
        stream.write_all(&long_speech()).await.unwrap();
        stalled.push(stream);
    }
    holding(2, || tts.active_sessions()).await;

    let health = b"GET /health HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n";
    let started = Instant::now();
    assert_eq!(exchange(address, health).await, 200);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    let waited = until_answered(address, &short_speech(), 200).await;
    assert!(
        waited < Duration::from_secs(5),
        "a slot was free after {waited:?}"
    );
    drop(stalled);
    running.shutdown().await.unwrap();
}

/// 300 characters, so each event is a few hundred bytes.
const LONG_WORD: &str = "wordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordword\
wordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordword\
wordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordword\
wordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordwordword";

/// A reader that stops reading, on a server with no request timeout (the
/// default), loses its connection once the server has waited
/// `with_write_timeout` for it to take more, which frees its slot.
#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_reader_is_dropped_after_the_write_timeout() {
    let tts = TtsEngine::new(FakeTts::plain()).with_max_sessions(1);
    let running = Server::new()
        .with_tts(tts.clone(), "fake-tts")
        .with_bind("127.0.0.1:0")
        .with_write_timeout(Duration::from_millis(500))
        .start(std::future::pending())
        .await
        .unwrap();
    let mut stalled = connect(running.local_addr()).await;
    stalled.write_all(&long_speech()).await.unwrap();
    holding(1, || tts.active_sessions()).await;

    let started = Instant::now();
    while tts.active_sessions() > 0 {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "a stalled reader held the only slot for 5 s"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    drop(stalled);
    running.shutdown().await.unwrap();
}

/// The write timeout only catches a client that takes nothing: one that
/// reads slowly, but more often than the timeout, gets all of a response
/// that takes many timeouts to send.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_reader_that_keeps_reading_gets_everything() {
    let running = Server::new()
        .with_tts(TtsEngine::new(FakeTts::plain()), "fake-tts")
        .with_bind("127.0.0.1:0")
        .with_write_timeout(Duration::from_millis(300))
        .start(std::future::pending())
        .await
        .unwrap();
    let address = running.local_addr();
    let body = json!({"model": "tts-1", "input": "word. ".repeat(100), "voice": "alpha",
        "response_format": "pcm"});
    let speech = request(
        "/v1/audio/speech",
        "application/json",
        body.to_string().as_bytes(),
    );

    let mut fast = TcpStream::connect(address).await.unwrap();
    fast.write_all(&speech).await.unwrap();
    let mut whole = Vec::new();
    fast.read_to_end(&mut whole).await.unwrap();
    let mut slow = connect(address).await;
    slow.write_all(&speech).await.unwrap();
    let started = Instant::now();
    let (read, ended) = read_slowly(
        slow,
        4096,
        Duration::from_millis(50),
        Duration::from_secs(30),
    )
    .await;
    assert!(ended);
    assert!(
        started.elapsed() > Duration::from_secs(1),
        "the response took {:?}, under several write timeouts",
        started.elapsed()
    );
    // Chunk framing may differ a little between the two.
    assert!(
        read.abs_diff(whole.len()) < whole.len() / 100,
        "the slow reader got {read} of {} bytes",
        whole.len()
    );
    running.shutdown().await.unwrap();
}
