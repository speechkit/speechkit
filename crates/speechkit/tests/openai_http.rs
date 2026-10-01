//! OpenAI HTTP transcription against a local mock server, with no network
//! access.
#![cfg(feature = "openai")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::Arc, time::Duration};

use speechkit::cloud::{CloudRuntime, OpenAiTranscription, OpenAiTranscriptionConfig};
use speechkit::{
    AudioBuffer, SampleRate, Secret, SpeechError,
    asr::{AsrEngine, AsrOptions},
    vad::OfflineRecognizer,
};
use speechkit_testkit::{
    contract::asr::{run_asr_contract, tone},
    secs,
};
use wiremock::{
    Match, Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path},
};

/// Matches a body containing `needle`. Multipart bodies hold binary WAV
/// data, so string matchers do not apply.
struct BodyHas(&'static str);

impl Match for BodyHas {
    fn matches(&self, request: &Request) -> bool {
        request
            .body
            .windows(self.0.len())
            .any(|window| window == self.0.as_bytes())
    }
}

fn body_string_contains(needle: &'static str) -> BodyHas {
    BodyHas(needle)
}

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

    fn requests(&self) -> usize {
        self.runtime
            .block_on(self.mock.received_requests())
            .map_or(0, |r| r.len())
    }

    fn url(&self, prefix: &str) -> String {
        format!("{}{prefix}", self.mock.uri())
    }
}

fn backend(config: OpenAiTranscriptionConfig) -> OpenAiTranscription {
    OpenAiTranscription::new(config, CloudRuntime::owned(1).unwrap()).unwrap()
}

fn key() -> Arc<Secret> {
    Arc::new(Secret::new("sk-test-key"))
}

fn opts() -> AsrOptions {
    AsrOptions::default()
}

const SSE_BODY: &str = "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello \"}\n\n\
                        data: {\"type\":\"transcript.text.delta\",\"delta\":\"world\"}\n\n\
                        data: {\"type\":\"transcript.text.done\",\"text\":\"hello world\"}\n\n";

fn transcription() -> wiremock::MockBuilder {
    Mock::given(method("POST")).and(path("/v1/audio/transcriptions"))
}

#[test]
fn json_response() {
    let server = Server::start();
    server.mount(
        transcription()
            .and(header("authorization", "Bearer sk-test-key"))
            .and(body_string_contains("name=\"model\""))
            .and(body_string_contains("whisper-1"))
            .and(body_string_contains("name=\"language\""))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"text":"你好"}"#)),
    );
    let config = OpenAiTranscriptionConfig::new("whisper-1")
        .with_endpoint(server.url("/v1"))
        .with_api_key(key());
    let text = backend(config)
        .recognize(&tone(16_000), &opts().with_language("zh"))
        .unwrap();
    assert_eq!(text, "你好");
}

#[test]
fn sse_response_reports_partials_then_a_segment() {
    let server = Server::start();
    server.mount(
        transcription()
            .and(body_string_contains("name=\"stream\""))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(SSE_BODY),
            ),
    );
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(server.url("/v1"))
        .with_api_key(key())
        .with_streaming(true);
    let engine = AsrEngine::new(backend(config));
    assert!(engine.capabilities().reports_partials);
    let session = engine
        .start(SampleRate::HZ_16000, opts(), Duration::from_secs(10))
        .unwrap();
    let mut observer = session.updates();
    session.push(tone(16_000), secs(10)).unwrap();
    let result = session.finish(secs(30));
    assert_eq!(result.as_ref().unwrap().text(), "hello world");
    let (_, updates) = speechkit_testkit::contract::asr::drain(&mut observer);
    let partials: Vec<_> = updates
        .iter()
        .filter_map(|u| match u {
            speechkit::asr::AsrUpdate::Partial(p) => Some(p.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        partials.is_empty() || partials.last() == Some(&"hello world"),
        "{partials:?}"
    );
}

#[test]
fn sse_without_done_is_retryable() {
    let server = Server::start();
    server.mount(
        transcription().respond_with(
            ResponseTemplate::new(200)
                .set_body_string("data: {\"type\":\"transcript.text.delta\",\"delta\":\"x\"}\n\n"),
        ),
    );
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(server.url("/v1"))
        .with_api_key(key())
        .with_streaming(true);
    let error = backend(config)
        .recognize(&tone(1_600), &opts())
        .unwrap_err();
    assert!(error.retryable(), "{error}");
}

#[test]
fn error_statuses_and_retryability() {
    for (status, retryable) in [(401, false), (400, false), (429, true), (503, true)] {
        let server = Server::start();
        server.mount(
            transcription().respond_with(
                ResponseTemplate::new(status).set_body_string(
                    r#"{"error":{"message":"sk-test-key is wrong","type":"auth"}}"#,
                ),
            ),
        );
        let config = OpenAiTranscriptionConfig::new("m")
            .with_endpoint(server.url("/v1"))
            .with_api_key(key());
        let error = backend(config)
            .recognize(&tone(1_600), &opts())
            .unwrap_err();
        assert_eq!(error.retryable(), retryable, "{status}");
        let chain = format!("{error}: {}", std::error::Error::source(&error).unwrap());
        assert!(chain.contains(&status.to_string()), "{chain}");
        assert!(!chain.contains("sk-test-key"), "{chain}");
    }
}

#[test]
fn oversized_upload_is_rejected_before_sending() {
    let server = Server::start();
    server.mount(transcription().respond_with(ResponseTemplate::new(200).set_body_string("{}")));
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(server.url("/v1"))
        .with_api_key(key());
    let backend = backend(config);
    // The upload limit is 24 MiB: 16-bit samples after a 44-byte header.
    let too_long = vec![0.0; 12 * 1024 * 1024];
    let error = backend.recognize(&too_long, &opts()).unwrap_err();
    assert!(matches!(error, SpeechError::InvalidInput(_)), "{error}");
    let engine = AsrEngine::new(backend);
    let audio = AudioBuffer::new(SampleRate::HZ_16000, too_long);
    let failure = engine.transcribe(&audio, opts(), secs(10)).unwrap_err();
    assert!(
        matches!(failure.error, SpeechError::InvalidInput(_)),
        "{}",
        failure.error
    );
    assert_eq!(server.requests(), 0);
}

#[test]
fn timeout_is_retryable() {
    let server = Server::start();
    server.mount(
        transcription().respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_string(r#"{"text":"late"}"#),
        ),
    );
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(server.url("/v1"))
        .with_api_key(key())
        .with_timeout(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let error = backend(config)
        .recognize(&tone(1_600), &opts())
        .unwrap_err();
    assert!(error.retryable(), "{error}");
    assert!(started.elapsed() < Duration::from_secs(4));
}

#[test]
fn proxy_prefix_is_kept() {
    let server = Server::start();
    server.mount(
        Mock::given(method("POST"))
            .and(path("/proxy/openai/v1/audio/transcriptions"))
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"text":"via proxy"}"#)),
    );
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(server.url("/proxy/openai/v1/"))
        .with_api_key(key());
    assert_eq!(
        backend(config).recognize(&tone(1_600), &opts()).unwrap(),
        "via proxy"
    );
}

#[test]
fn keyless_servers_get_no_authorization_header() {
    let server = Server::start();
    server.mount(
        transcription()
            .respond_with(ResponseTemplate::new(200).set_body_string(r#"{"text":"ok"}"#)),
    );
    let config = OpenAiTranscriptionConfig::new("m").with_endpoint(server.url("/v1"));
    assert_eq!(
        backend(config).recognize(&tone(160), &opts()).unwrap(),
        "ok"
    );
    let requests = server
        .runtime
        .block_on(server.mock.received_requests())
        .unwrap();
    assert!(!requests[0].headers.contains_key("authorization"));
}

#[test]
fn invalid_configs() {
    let runtime = CloudRuntime::owned(1).unwrap();
    let bad = [
        OpenAiTranscriptionConfig::new("m").with_endpoint("ftp://x/v1"),
        OpenAiTranscriptionConfig::new(" ").with_endpoint("http://x/v1"),
        OpenAiTranscriptionConfig::new("m")
            .with_endpoint("http://x/v1")
            .with_timeout(Duration::ZERO),
    ];
    for config in bad {
        assert!(OpenAiTranscription::new(config, runtime.clone()).is_err());
    }
}

#[test]
fn contract_over_scripted_sse() {
    let server = Server::start();
    server
        .mount(transcription().respond_with(ResponseTemplate::new(200).set_body_string(SSE_BODY)));
    let runtime = CloudRuntime::owned(2).unwrap();
    let url = server.url("/v1");
    run_asr_contract(|| {
        let config = OpenAiTranscriptionConfig::new("m")
            .with_endpoint(url.clone())
            .with_api_key(key())
            .with_streaming(true);
        let backend = OpenAiTranscription::new(config, runtime.clone()).unwrap();
        AsrEngine::new(backend)
    });
}

#[test]
fn many_sse_deltas_do_not_accumulate_a_partial_history() {
    use speechkit::asr::{AsrBackend, AsrEvent};
    let server = Server::start();
    let delta = "data: {\"type\":\"transcript.text.delta\",\"delta\":\"abcdefghijklmnopqrst\"}\n\n";
    let final_text = "abcdefghijklmnopqrst".repeat(1_000);
    let body = format!(
        "{}data: {{\"type\":\"transcript.text.done\",\"text\":\"{final_text}\"}}\n\n",
        delta.repeat(1_000)
    );
    server.mount(transcription().respond_with(ResponseTemplate::new(200).set_body_string(body)));
    let backend = backend(
        OpenAiTranscriptionConfig::new("m")
            .with_endpoint(server.url("/v1"))
            .with_api_key(key())
            .with_streaming(true),
    );
    // Every partial goes out as it arrives; the stream keeps none of them.
    let sent = speechkit_testkit::asr::Collected::default();
    let mut stream = backend.open(&opts(), sent.events()).unwrap();
    stream.accept(&tone(160)).unwrap();
    stream.finish().unwrap();
    let events = sent.take();
    assert!(
        matches!(events.last(), Some(AsrEvent::Segment(segment)) if segment.text == final_text)
    );

    // A session that has ended stops the upload reader at the next partial.
    let partials = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counted = partials.clone();
    let stopping = speechkit::asr::AsrEvents::forward(
        move |event| {
            if matches!(event, AsrEvent::Partial(_)) {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            speechkit::Flow::Stop
        },
        |_| {},
    );
    let mut stream = backend.open(&opts(), stopping).unwrap();
    stream.accept(&tone(160)).unwrap();
    let error = stream.finish().unwrap_err();
    assert!(matches!(error, SpeechError::Cancelled));
    assert_eq!(
        partials.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "a stopped session must stop the upload reader"
    );
}

/// The live API's JSON, text, and SSE framing must parse. A tone may
/// transcribe to anything, even nothing, so only success is checked.
#[test]
#[ignore = "needs OPENAI_API_KEY and network"]
fn live_openai_http_with_and_without_streaming() {
    let Some(key) = std::env::var("OPENAI_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    else {
        return;
    };
    let key = Arc::new(Secret::new(key));
    for streaming in [false, true] {
        let config = OpenAiTranscriptionConfig::new("gpt-4o-mini-transcribe")
            .with_endpoint("https://api.openai.com/v1")
            .with_api_key(key.clone())
            .with_streaming(streaming);
        let result = backend(config).recognize(&tone(32_000), &opts());
        assert!(result.is_ok(), "streaming {streaming}: {result:?}");
    }
}
