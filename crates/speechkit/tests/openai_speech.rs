//! OpenAI speech synthesis against a local mock server, with no network
//! access.
#![cfg(feature = "openai")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::Arc, time::Duration};

use speechkit::cloud::{CloudRuntime, OpenAiSpeech, OpenAiSpeechConfig};
use speechkit::{
    SampleRate, Secret, SpeechError,
    tts::{TtsEngine, TtsOptions},
};
use speechkit_testkit::{contract::tts::run_tts_contract, secs};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header, method, path},
};

/// A mock server on its own runtime, since the backend blocks on its own.
struct Server {
    runtime: tokio::runtime::Runtime,
    mock: MockServer,
}

impl Server {
    fn start() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let mock = runtime.block_on(MockServer::start());
        Self { runtime, mock }
    }

    fn mount(&self, mock: Mock) {
        self.runtime.block_on(mock.mount(&self.mock));
    }

    fn url(&self) -> String {
        format!("{}/v1", self.mock.uri())
    }
}

/// `seconds` of a 440 Hz tone at 24 kHz, as PCM16 bytes.
fn pcm(seconds: f32) -> Vec<u8> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "short, positive durations"
    )]
    let frames = (24_000.0 * seconds) as usize;
    (0..frames)
        .flat_map(|i| {
            #[expect(clippy::cast_precision_loss, reason = "small indices")]
            let t = i as f32 / 24_000.0;
            #[expect(clippy::cast_possible_truncation, reason = "amplitude is bounded")]
            let value = ((t * 440.0 * std::f32::consts::TAU).sin() * 8_000.0) as i16;
            value.to_le_bytes()
        })
        .collect()
}

fn engine(url: &str, key: Option<Arc<Secret>>) -> TtsEngine {
    let mut config = OpenAiSpeechConfig::new("gpt-4o-mini-tts").with_endpoint(url);
    if let Some(key) = key {
        config = config.with_api_key(key);
    }
    let backend = OpenAiSpeech::new(config, CloudRuntime::owned(1).unwrap()).unwrap();
    TtsEngine::new(backend)
}

#[test]
fn contract_suite_offline() {
    let server = Server::start();
    server.mount(
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(pcm(0.2))),
    );
    let url = server.url();
    run_tts_contract(|| engine(&url, None));
}

#[test]
fn sends_the_request_and_streams_a_long_body() {
    let server = Server::start();
    server.mount(
        Mock::given(method("POST"))
            .and(path("/v1/audio/speech"))
            .and(header("authorization", "Bearer sk-test"))
            .and(body_partial_json(serde_json::json!({
                "model": "gpt-4o-mini-tts",
                "voice": "nova",
                "input": "Hello there.",
                "speed": 1.25,
                "response_format": "pcm",
            })))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(pcm(10.0))),
    );
    let engine = engine(&server.url(), Some(Arc::new(Secret::new("sk-test"))));
    let audio = engine
        .synthesize(
            "Hello there.",
            TtsOptions::default().with_voice("nova").with_speed(1.25),
            secs(60),
        )
        .unwrap();
    assert_eq!(audio.sample_rate, SampleRate::HZ_24000);
    assert_eq!(audio.samples.len(), 240_000);
}

/// The error and its source, as one line.
fn detail(error: &SpeechError) -> String {
    match std::error::Error::source(error) {
        Some(source) => format!("{error}: {source}"),
        None => error.to_string(),
    }
}

fn failure(status: u16, body: &str) -> SpeechError {
    let server = Server::start();
    server.mount(
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(status).set_body_string(body.to_owned())),
    );
    let engine = engine(&server.url(), None);
    engine
        .synthesize("Hello.", TtsOptions::default(), secs(30))
        .unwrap_err()
        .error
}

#[test]
fn errors_carry_status_and_retryability() {
    let limited = failure(
        429,
        r#"{"error":{"type":"rate_limit","code":"rate_limit_exceeded"}}"#,
    );
    assert!(limited.retryable(), "{limited}");
    let text = detail(&limited);
    assert!(
        text.contains("openai-speech") && text.contains("429"),
        "{text}"
    );
    let bad = failure(400, r#"{"error":{"type":"invalid_request_error"}}"#);
    assert!(!bad.retryable(), "{bad}");
    let server_error = failure(503, "overloaded");
    assert!(server_error.retryable());
}

#[test]
fn half_a_sample_fails_the_session() {
    let server = Server::start();
    server.mount(
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(vec![0_u8; 101])),
    );
    let failure = engine(&server.url(), None)
        .synthesize("Hello.", TtsOptions::default(), secs(30))
        .unwrap_err();
    let text = detail(&failure.error);
    assert!(text.contains("middle of a sample"), "{text}");
    assert_eq!(
        failure.duration,
        speechkit::SampleRate::HZ_24000.duration_of(50)
    );
}

#[test]
fn unknown_voices_and_speeds_are_rejected_before_a_request() {
    let engine = engine("http://127.0.0.1:9/v1", None);
    assert!(matches!(
        engine.start(
            TtsOptions::default().with_voice("robot"),
            Duration::from_secs(10)
        ),
        Err(SpeechError::InvalidInput(_))
    ));
    assert!(
        engine
            .start(
                TtsOptions::default().with_speed(5.0),
                Duration::from_secs(10)
            )
            .is_err()
    );
}

#[test]
fn bad_settings_are_rejected() {
    let runtime = CloudRuntime::owned(1).unwrap();
    let config = |url: &str| OpenAiSpeechConfig::new("tts-1").with_endpoint(url);
    assert!(OpenAiSpeech::new(config("not a url"), runtime.clone()).is_err());
    assert!(
        OpenAiSpeech::new(
            config("http://h/v1").with_voices(Vec::new()),
            runtime.clone()
        )
        .is_err()
    );
    assert!(
        OpenAiSpeech::new(
            OpenAiSpeechConfig::new(" ").with_endpoint("http://h/v1"),
            runtime
        )
        .is_err()
    );
}

#[test]
#[ignore = "needs OPENAI_API_KEY and network"]
fn live_openai_speech() {
    let Some(key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    else {
        return;
    };
    let engine = engine(
        "https://api.openai.com/v1",
        Some(Arc::new(Secret::new(key))),
    );
    let audio = engine
        .synthesize("Hello from speechkit.", TtsOptions::default(), secs(60))
        .unwrap();
    assert!(
        audio.samples.len() > 12_000,
        "{} samples",
        audio.samples.len()
    );
}
