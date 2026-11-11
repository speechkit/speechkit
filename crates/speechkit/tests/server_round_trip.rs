//! Round trips across speechkit's own boundaries: what one part writes,
//! another reads. The server's synthesized audio decodes with
//! `audio::decode` and transcribes on the same server, and the OpenAI
//! clients chain through the server. A WAV the server wrote that the
//! library could not read went unnoticed until a user tried it.
#![cfg(all(feature = "server", feature = "openai"))]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::atomic::Ordering, time::Duration};

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use speechkit::{
    AudioBuffer, SampleRate,
    asr::AsrEngine,
    audio::{self, DecodeLimits},
    server::Server,
    tts::TtsEngine,
};
use speechkit_testkit::{
    asr::FakeAsr,
    tts::{FakeTts, samples_for},
};
use tower::ServiceExt;

const BOUNDARY: &str = "speechkit-round-trip";
/// Two sentences, long enough for `FakeAsr::hello_world` to say both
/// words: it says "hello" after 8,000 samples at 16 kHz.
const SENTENCES: [&str; 2] = [
    "Hello there, this is the first of two sentences.",
    "And this is the second one, a little longer than the first.",
];

fn text() -> String {
    SENTENCES.join(" ")
}

/// The samples the 16 kHz fake makes for the sentences, at `rate`.
fn expected_at(rate: SampleRate) -> f64 {
    let native: usize = SENTENCES.iter().map(|sentence| samples_for(sentence)).sum();
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let native = native as f64;
    native * f64::from(rate.hz()) / 16_000.0
}

/// Whether `samples` is within 2% (plus a filter's edge) of `expected`.
fn close(samples: u64, expected: f64) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let got = samples as f64;
    (got - expected).abs() <= expected * 0.02 + 32.0
}

fn speech(body: &Value) -> Request<Body> {
    Request::post("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn transcription(file: &[u8]) -> Request<Body> {
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    Request::post("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

/// `/v1/audio/speech` → `audio::decode` → `/v1/audio/transcriptions`, for
/// every format the server writes: what it synthesizes, it can read back.
#[tokio::test(flavor = "multi_thread")]
async fn synthesized_audio_decodes_and_transcribes() {
    let fake = FakeAsr::hello_world();
    let heard = fake.stats();
    let app = Server::new()
        .with_asr(AsrEngine::new(fake), "fake-asr")
        .with_tts(TtsEngine::new(FakeTts::plain()), "fake-tts")
        .router();
    for format in ["wav", "pcm"] {
        let (status, bytes) = send(
            &app,
            speech(&json!({"model": "tts-1", "input": text(), "voice": "alpha",
                "response_format": format})),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{format}");
        let audio = if format == "wav" {
            audio::decode(&bytes, DecodeLimits::default()).unwrap()
        } else {
            // Raw 16-bit little-endian PCM at 24 kHz, as OpenAI's is.
            let (samples, rest) = bytes.as_chunks::<2>();
            assert!(rest.is_empty(), "an odd number of PCM bytes");
            let samples = samples
                .iter()
                .map(|&pair| f32::from(i16::from_le_bytes(pair)) / 32_768.0)
                .collect();
            AudioBuffer::new(SampleRate::HZ_24000, samples)
        };
        assert_eq!(audio.sample_rate, SampleRate::HZ_24000, "{format}");
        let length = audio.samples.len() as u64;
        assert!(
            close(length, expected_at(SampleRate::HZ_24000)),
            "{format}: {length} samples"
        );
        let upload = speechkit::audio::encode_wav(&audio).unwrap();
        let file = if format == "wav" { &bytes } else { &upload };

        let before = heard.samples.load(Ordering::Relaxed);
        let (status, body) = send(&app, transcription(file)).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "{format}: {}",
            String::from_utf8_lossy(&body)
        );
        let text: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(text["text"], "hello world", "{format}");
        // The recognizer heard all of it, at its own rate.
        let fed = heard.samples.load(Ordering::Relaxed) - before;
        assert!(
            close(fed, expected_at(SampleRate::HZ_16000)),
            "{format}: the recognizer heard {fed} samples"
        );
    }
}

/// The OpenAI speech client and the OpenAI transcription client, chained
/// through one speechkit server: text → audio → text.
#[test]
fn the_openai_clients_chain_through_the_server() {
    use speechkit::cloud::{
        CloudRuntime, OpenAiSpeech, OpenAiSpeechConfig, OpenAiTranscription,
        OpenAiTranscriptionConfig,
    };
    use speechkit::{
        asr::AsrOptions,
        tts::{TtsOptions, Voice},
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let fake = FakeAsr::hello_world();
    let heard = fake.stats();
    let running = runtime
        .block_on(
            Server::new()
                .with_asr(AsrEngine::new(fake), "fake-asr")
                .with_tts(TtsEngine::new(FakeTts::plain()), "fake-tts")
                .with_bind("127.0.0.1:0")
                .start(std::future::pending()),
        )
        .unwrap();
    let base = format!("http://{}/v1", running.local_addr());
    let client_runtime = CloudRuntime::owned(1).unwrap();

    let speaker = OpenAiSpeechConfig::new("any")
        .with_endpoint(base.clone())
        .with_voices(vec![Voice::new("alpha")]);
    let speaker = TtsEngine::new(OpenAiSpeech::new(speaker, client_runtime.clone()).unwrap());
    let audio = speaker
        .synthesize(&text(), TtsOptions::default(), speechkit_testkit::secs(30))
        .unwrap();
    assert!(close(
        audio.samples.len() as u64,
        expected_at(audio.sample_rate)
    ));

    for streaming in [false, true] {
        let before = heard.samples.load(Ordering::Relaxed);
        let listener = OpenAiTranscriptionConfig::new("any")
            .with_endpoint(base.clone())
            .with_streaming(streaming);
        let listener =
            AsrEngine::new(OpenAiTranscription::new(listener, client_runtime.clone()).unwrap());
        let session = listener
            .start(
                audio.sample_rate,
                AsrOptions::default(),
                speechkit_testkit::secs(30),
            )
            .unwrap();
        session
            .push(audio.samples.clone(), speechkit_testkit::secs(30))
            .unwrap();
        let text = session.finish(speechkit_testkit::secs(30)).unwrap().text();
        assert_eq!(text, "hello world", "streaming {streaming}");
        let fed = heard.samples.load(Ordering::Relaxed) - before;
        assert!(
            close(fed, expected_at(SampleRate::HZ_16000)),
            "streaming {streaming}: the recognizer heard {fed} samples"
        );
    }
    runtime
        .block_on(async { tokio::time::timeout(Duration::from_secs(10), running.shutdown()).await })
        .unwrap()
        .unwrap();
}
