//! OpenAI Realtime against a local replay server, plus
//! named ignored tests for questions only a real key can answer.
#![cfg(feature = "openai")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use speechkit::cloud::{CloudRuntime, OpenAiRealtime, OpenAiRealtimeConfig};
use speechkit::{
    Secret, SpeechError,
    asr::{AsrBackend, AsrEngine, AsrEvent, AsrUpdate},
    vad::{EnergyVad, EnergyVadConfig},
};
use speechkit_testkit::{
    asr::Collected,
    contract::asr::{RATE, check_activity, options, run_asr_contract, tone},
    eventually, secs,
};
use tokio_tungstenite::tungstenite::Message;

/// What the replay server does besides the happy path.
#[derive(Clone, Copy)]
enum Mode {
    Normal,
    /// Answers every append with an error event.
    Fail,
    /// Answers the first append with an item 300 ms later.
    Late,
    /// Drops the connection after the first append.
    Drop,
    /// Reports the audio of a connection as one long utterance: the speech
    /// starts with the first append and never stops, so only a commit ends
    /// it.
    Speech,
}

/// Emulates the Realtime protocol: confirms settings, reports speech from
/// 500 ms into every second of audio to its end, and ends and commits a
/// server-VAD item for it, and answers the final commit.
async fn session(stream: tokio::net::TcpStream, mode: Mode, closed: Arc<AtomicUsize>) {
    serve_session(stream, mode).await;
    closed.fetch_add(1, Ordering::SeqCst);
}

#[expect(
    clippy::result_large_err,
    reason = "tungstenite's handshake callback fixes the error type to an HTTP response"
)]
async fn serve_session(stream: tokio::net::TcpStream, mode: Mode) {
    let Ok(mut ws) = tokio_tungstenite::accept_hdr_async(
        stream,
        |request: &tokio_tungstenite::tungstenite::handshake::server::Request, response| {
            assert!(
                !request.headers().contains_key("openai-beta"),
                "GA transcription must not select the retired beta interface"
            );
            Ok(response)
        },
    )
    .await
    else {
        return;
    };
    let mut buffered = 0_usize;
    let mut total = 0_usize;
    let mut items = 0_u32;
    let mut vad = true;
    while let Some(Ok(message)) = ws.next().await {
        let Message::Text(text) = message else {
            continue;
        };
        let request: Value = serde_json::from_str(&text).unwrap();
        let mut replies = Vec::new();
        match request["type"].as_str().unwrap() {
            "session.update" => {
                vad = !request["session"]["audio"]["input"]["turn_detection"].is_null();
                replies.push(json!({"type": "session.updated"}));
            }
            "input_audio_buffer.append" => {
                match mode {
                    Mode::Fail => replies.push(json!({"type": "error", "error": {"code": "invalid_value", "message": "nope"}})),
                    Mode::Drop => return,
                    Mode::Late if items == 0 => {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        replies.extend(item(&mut items, "late"));
                    }
                    Mode::Speech if total == 0 => {
                        let item_id = format!("item_{}", items + 1);
                        replies.push(json!({"type": "input_audio_buffer.speech_started", "item_id": item_id, "audio_start_ms": 0}));
                    }
                    Mode::Late | Mode::Normal | Mode::Speech => {}
                }
                let audio = base64::engine::general_purpose::STANDARD
                    .decode(request["audio"].as_str().unwrap())
                    .unwrap();
                buffered += audio.len() / 2;
                total += audio.len() / 2;
                while vad && !matches!(mode, Mode::Speech) && buffered >= 24_000 {
                    // The item's speech runs from 500 ms into its second
                    // of audio to its end, reported within the 1 s lag.
                    let end_ms = (total - buffered + 24_000) / 24;
                    buffered -= 24_000;
                    let item_id = format!("item_{}", items + 1);
                    replies.push(json!({"type": "input_audio_buffer.speech_started", "item_id": item_id, "audio_start_ms": end_ms - 500}));
                    replies.push(json!({"type": "input_audio_buffer.speech_stopped", "item_id": item_id, "audio_end_ms": end_ms}));
                    replies.extend(item(&mut items, "second"));
                }
            }
            "input_audio_buffer.commit" => {
                if buffered == 0 {
                    replies.push(json!({"type": "error", "error": {"code": "input_audio_buffer_commit_empty", "message": "empty"}}));
                } else {
                    buffered = 0;
                    replies.extend(item(&mut items, "tail"));
                }
            }
            _ => {}
        }
        for reply in replies {
            if ws
                .send(Message::Text(reply.to_string().into()))
                .await
                .is_err()
            {
                return;
            }
        }
    }
}

fn item(items: &mut u32, text: &str) -> Vec<Value> {
    *items += 1;
    let id = format!("item_{items}");
    vec![
        json!({"type": "input_audio_buffer.committed", "item_id": id, "previous_item_id": null}),
        json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": id, "delta": text}),
        json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": id, "transcript": format!("{text}.")}),
    ]
}

struct Server {
    _runtime: tokio::runtime::Runtime,
    url: String,
    /// Connections the server has seen end.
    closed: Arc<AtomicUsize>,
}

fn server(mode: Mode) -> Server {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let url = format!("ws://{}/v1/realtime", listener.local_addr().unwrap());
    let closed = Arc::new(AtomicUsize::new(0));
    runtime.spawn({
        let closed = closed.clone();
        async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(session(stream, mode, closed.clone()));
            }
        }
    });
    Server {
        _runtime: runtime,
        url,
        closed,
    }
}

fn config(url: &str) -> OpenAiRealtimeConfig {
    OpenAiRealtimeConfig::new("gpt-4o-transcribe", Arc::new(Secret::new("sk-test")))
        .with_endpoint(url)
        .with_timeout(Duration::from_secs(10))
}

fn engine(url: &str, runtime: &CloudRuntime) -> AsrEngine {
    let backend = OpenAiRealtime::new(config(url), runtime.clone()).unwrap();
    AsrEngine::new(backend)
}

/// What a stream of `backend` sends for `audio` at 24 kHz, fed in 100 ms
/// blocks.
/// Streams `audio` at twice real time: activity while silent trails the
/// audio sent by the lag, which assumes the service keeps up with the
/// audio, so a stream much faster than real time could outrun the replay
/// server's reports on a loaded machine.
fn events_of(backend: &dyn AsrBackend, audio: &[f32]) -> Vec<AsrEvent> {
    let sent = Collected::default();
    let mut stream = backend.open(&options(), sent.events()).unwrap();
    for chunk in audio.chunks(2_400) {
        stream.accept(chunk).unwrap();
        std::thread::sleep(Duration::from_millis(50));
    }
    stream.finish().unwrap();
    drop(stream);
    sent.take()
}

fn activity(events: &[AsrEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::SpeechStarted { at } => Some(format!("started {}", at.as_millis())),
            AsrEvent::SpeechEnded { at, utterance } => {
                Some(format!("ended {} {}", at.as_millis(), utterance.0))
            }
            _ => None,
        })
        .collect()
}

#[test]
fn server_vad_reports_speech_activity() {
    let server = server(Mode::Normal);
    let backend =
        OpenAiRealtime::new(config(&server.url), CloudRuntime::owned(1).unwrap()).unwrap();
    assert!(backend.capabilities().reports_activity);
    let events = events_of(&backend, &vec![0.1; 60_000]);
    check_activity(&events);
    assert_eq!(
        activity(&events),
        [
            "started 500",
            "ended 1000 0",
            "started 1500",
            "ended 2000 1"
        ]
    );
    // Between speech, activity trails the audio sent by the lag, 1 s.
    let mut known = events.iter().filter_map(|event| match event {
        AsrEvent::ActivityKnown { through } => Some(*through),
        _ => None,
    });
    // Once the stream has finished, activity is known to the end of the
    // audio, not of the silence appended before the final commit.
    assert_eq!(known.next_back(), Some(Duration::from_millis(2_500)));
    let without = OpenAiRealtime::new(
        config(&server.url).with_server_vad(false),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    assert!(!without.capabilities().reports_activity);
}

/// Checks that `events`, from a stream that reconnected, read as one
/// stream: segments in order, back to back, ending at the audio's end.
fn continuous(events: &[AsrEvent], audio_end: Duration) {
    check_activity(events);
    let segments: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::Segment(segment) => Some(segment),
            _ => None,
        })
        .collect();
    let mut end = Duration::ZERO;
    for (index, segment) in segments.iter().enumerate() {
        assert_eq!(segment.utterance.0, index as u64, "{segments:#?}");
        assert_eq!(segment.start, end, "{segments:#?}");
        assert!(segment.end >= segment.start, "{segments:#?}");
        end = segment.end;
    }
    assert_eq!(end, audio_end, "{segments:#?}");
}

#[test]
fn server_vad_reconnects_at_a_pause_before_the_limit() {
    let server = server(Mode::Normal);
    // The stream sends 5 s of audio in 2.5 s: at least two reconnects.
    let limit = Duration::from_millis(900);
    let backend = OpenAiRealtime::new(
        config(&server.url).with_connection_limit(limit),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    let events = events_of(&backend, &vec![0.1; 120_000]);
    continuous(&events, Duration::from_secs(5));
    assert!(
        eventually(Duration::from_secs(5), || server
            .closed
            .load(Ordering::SeqCst)
            >= 3),
        "{} connections",
        server.closed.load(Ordering::SeqCst)
    );
    // Every "second." item is a full second of speech, which the server
    // reports relative to its own connection: shifted back, the times
    // still end on the stream's audio.
    for event in &events {
        if let AsrEvent::SpeechStarted { at } = event {
            assert!(*at < Duration::from_secs(5), "{at:?}");
        }
    }
}

#[test]
fn a_stream_without_activity_reconnects_at_the_limit() {
    let server = server(Mode::Normal);
    let backend = OpenAiRealtime::new(
        config(&server.url)
            .with_server_vad(false)
            .with_connection_limit(Duration::from_millis(900)),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    let events = events_of(&backend, &vec![0.1; 72_000]);
    continuous(&events, Duration::from_secs(3));
    let segments = events
        .iter()
        .filter(|event| matches!(event, AsrEvent::Segment(_)))
        .count();
    assert!(
        segments >= 2,
        "each connection commits its audio: {events:#?}"
    );
}

#[test]
fn a_forced_reconnect_splits_the_speech_in_progress() {
    let server = server(Mode::Normal);
    let vad = EnergyVad::new(EnergyVadConfig::default());
    let backend = OpenAiRealtime::new(
        config(&server.url)
            .with_vad(vad)
            .with_connection_limit(Duration::from_millis(900)),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    // One utterance of 3 s, longer than a connection lasts here.
    let events = events_of(&backend, &vec![0.1; 72_000]);
    continuous(&events, Duration::from_secs(3));
    let activity = activity(&events);
    assert_eq!(activity.len(), 2, "one start and one end: {activity:?}");
    let segments = events
        .iter()
        .filter(|event| matches!(event, AsrEvent::Segment(_)))
        .count();
    assert!(segments >= 2, "the utterance was split: {events:#?}");
}

#[test]
fn a_forced_reconnect_keeps_server_vad_speech_going() {
    let server = server(Mode::Speech);
    let backend = OpenAiRealtime::new(
        config(&server.url).with_connection_limit(Duration::from_millis(900)),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    // One utterance of 3 s, longer than a connection lasts here. Every
    // forced reconnect commits it again, and how many there are depends on
    // how fast the machine is, since a connection's age is wall-clock time:
    // the speech goes on across all of them.
    let events = events_of(&backend, &vec![0.1; 72_000]);
    continuous(&events, Duration::from_secs(3));
    let segments = events
        .iter()
        .filter(|event| matches!(event, AsrEvent::Segment(_)))
        .count();
    assert!(segments >= 2, "the utterance was split: {events:#?}");
    // `continuous` checked that the IDs run 0, 1, ...: the last is one less
    // than the count, and the one end of the speech names it.
    assert_eq!(
        activity(&events),
        [
            "started 0".to_owned(),
            format!("ended 3000 {}", segments - 1)
        ],
        "one start and one end: {events:#?}"
    );
    assert!(
        eventually(Duration::from_secs(5), || server
            .closed
            .load(Ordering::SeqCst)
            >= 2),
        "the stream reconnected: {} connections",
        server.closed.load(Ordering::SeqCst)
    );
}

#[test]
fn client_vad_commits_at_each_end_of_speech() {
    let server = server(Mode::Normal);
    let vad = EnergyVad::new(EnergyVadConfig::default());
    let backend = OpenAiRealtime::new(
        config(&server.url).with_vad(vad),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    assert!(backend.capabilities().reports_activity);
    let mut audio = vec![0.1_f32; 24_000];
    audio.extend(vec![0.0; 24_000]);
    audio.extend(vec![0.1; 24_000]);
    let events = events_of(&backend, &audio);
    check_activity(&events);
    let activity = activity(&events);
    assert_eq!(activity.len(), 4, "{activity:?}");
    assert!(activity[1].ends_with(" 0") && activity[3].ends_with(" 1"));
    // One commit at the pause, and one for the speech still going at the
    // end.
    let texts: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::Segment(segment) => Some(segment.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["tail.", "tail."]);
    let both = OpenAiRealtimeConfig::new("m", Arc::new(Secret::new("k")))
        .with_vad(EnergyVad::new(EnergyVadConfig::default()))
        .with_server_vad(true);
    assert!(both.vad.is_none(), "server VAD replaces a VAD");
}

#[test]
fn events_arrive_while_the_session_is_idle() {
    let server = server(Mode::Late);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine.start(RATE, options(), secs(10)).unwrap();
    let mut updates = session.updates();
    // Enough for a few feed blocks after resampling.
    session.push(tone(8_000), secs(10)).unwrap();
    // No more audio: the item still arrives.
    loop {
        match updates.recv(secs(10)).unwrap() {
            AsrUpdate::Segment(segment) => {
                assert_eq!(segment.text, "late.");
                break;
            }
            AsrUpdate::Closed(result) => panic!("closed early: {result:?}"),
            _ => {}
        }
    }
}

#[test]
fn a_dropped_connection_fails_the_idle_session() {
    let server = server(Mode::Drop);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine.start(RATE, options(), secs(10)).unwrap();
    session.push(tone(8_000), secs(10)).unwrap();
    let result = session.wait(secs(10)).expect("the connection dropped");
    let failure = result.unwrap_err();
    assert!(failure.error.retryable(), "{}", failure.error);
}

#[test]
fn the_slot_is_held_until_the_connection_closes() {
    let server = server(Mode::Normal);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine.start(RATE, options(), secs(10)).unwrap();
    session.push(tone(1_600), secs(10)).unwrap();
    session.cancel();
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
    assert!(eventually(Duration::from_secs(5), || server
        .closed
        .load(Ordering::SeqCst)
        == 1));
}

#[test]
fn contract_offline() {
    let server = server(Mode::Normal);
    let runtime = CloudRuntime::owned(2).unwrap();
    run_asr_contract(|| engine(&server.url, &runtime));
}

#[test]
fn transcript_follows_commit_order() {
    let server = server(Mode::Normal);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    for _ in 0..25 {
        session
            .push(tone(1_600), speechkit_testkit::secs(10))
            .unwrap();
    }
    let result = session.finish(speechkit_testkit::secs(20));
    let outcome = result.as_ref().unwrap();
    // 2.5 s at 16 kHz is 60 000 frames at 24 kHz: two server-VAD items,
    // then the tail committed on finish.
    assert_eq!(outcome.text(), "second. second. tail.");
}

#[test]
fn server_errors_fail_the_session() {
    let server = server(Mode::Fail);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    let _ = session.push(tone(1_600), speechkit_testkit::secs(10));
    let result = session.finish(speechkit_testkit::secs(10));
    let failure = result.as_ref().unwrap_err();
    let message = format!("{}", std::error::Error::source(&failure.error).unwrap());
    assert!(message.contains("invalid_value: nope"), "{message}");
    assert!(!failure.error.retryable());
}

#[test]
fn unreachable_server_is_retryable() {
    let config = OpenAiRealtimeConfig::new("m", Arc::new(Secret::new("k")))
        .with_endpoint("ws://127.0.0.1:1/")
        .with_timeout(Duration::from_secs(2));
    let backend = OpenAiRealtime::new(config, CloudRuntime::owned(1).unwrap()).unwrap();
    let engine = AsrEngine::new(backend);
    let error = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap_err();
    assert!(error.retryable(), "{error}");
    let bad = OpenAiRealtimeConfig::new("m", Arc::new(Secret::new("k"))).with_endpoint("https://x");
    assert!(matches!(
        OpenAiRealtime::new(bad, CloudRuntime::owned(1).unwrap()),
        Err(SpeechError::InvalidInput(_))
    ));
}

fn real_engine(server_vad: bool) -> Option<AsrEngine> {
    let key = std::env::var("OPENAI_API_KEY").ok()?;
    let config = OpenAiRealtimeConfig::new("gpt-4o-transcribe", Arc::new(Secret::new(key)))
        .with_server_vad(server_vad);
    let backend = OpenAiRealtime::new(config, CloudRuntime::owned(2).unwrap()).unwrap();
    Some(AsrEngine::new(backend))
}

/// With server VAD on, turning it off before the final commit must be
/// confirmed by the live service.
#[test]
#[ignore = "needs OPENAI_API_KEY"]
fn realtime_server_vad_close_barrier() {
    let Some(engine) = real_engine(true) else {
        return;
    };
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    for _ in 0..30 {
        session
            .push(tone(1_600), speechkit_testkit::secs(30))
            .unwrap();
    }
    let result = session.finish(speechkit_testkit::secs(60));
    assert!(result.is_ok(), "{result:?}");
}

/// The error code for committing an empty buffer must be
/// `input_audio_buffer_commit_empty`, which finish treats as success.
#[test]
#[ignore = "needs OPENAI_API_KEY"]
fn realtime_commit_empty_error_code() {
    let Some(engine) = real_engine(false) else {
        return;
    };
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    let result = session.finish(speechkit_testkit::secs(60));
    assert!(result.is_ok(), "{result:?}");
}
