//! In-memory tests of the HTTP API, plus interop with the
//! `speechkit::cloud` OpenAI client over a real localhost socket.
#![cfg(all(feature = "server", feature = "openai"))]
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers fail the calling test"
)]

use std::{sync::Arc, time::Duration};

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::Value;
use speechkit::server::Server;
use speechkit::{
    AudioBuffer, SampleRate,
    asr::{AsrCapabilities, AsrEngine, AsrLimits},
    audio::encode_wav,
};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    contract::asr::tone,
    eventually,
};
use tower::ServiceExt;

const BOUNDARY: &str = "speechkit-test-boundary";

fn wav(seconds: usize) -> Vec<u8> {
    let audio = AudioBuffer::new(SampleRate::HZ_16000, tone(16_000 * seconds));
    encode_wav(&audio).unwrap()
}

fn multipart(file: &[u8], fields: &[(&str, &str)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(file);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    body
}

fn post(body: Vec<u8>) -> Request<Body> {
    Request::post("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .unwrap()
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, String) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

fn engine(fake: FakeAsr, sessions: usize) -> AsrEngine {
    AsrEngine::new(fake).with_max_sessions(sessions)
}

fn server(engine: AsrEngine) -> Server {
    Server::new().with_asr(engine, "fake-model")
}

fn app(engine: AsrEngine) -> Router {
    server(engine).router()
}

fn json(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

#[tokio::test(flavor = "multi_thread")]
async fn json_text_and_sse_success() {
    let app = app(engine(FakeAsr::hello_world(), 8));
    let (status, body) = send(&app, post(multipart(&wav(1), &[("model", "whatever")]))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(json(&body)["text"], "hello world");

    let (status, body) = send(
        &app,
        post(multipart(&wav(1), &[("response_format", "text")])),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, "hello world");

    let (status, body) = send(&app, post(multipart(&wav(1), &[("stream", "true")]))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("transcript.text.done"), "{body}");
    assert!(body.contains("\"text\":\"hello world\""), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn pushes_respect_a_small_input_queue() {
    let limits = AsrLimits::default().with_input_queue(Duration::from_millis(50));
    let engine = AsrEngine::new(FakeAsr::hello_world())
        .with_max_sessions(1)
        .with_limits(limits);
    let app = app(engine);
    let (status, body) = send(&app, post(multipart(&wav(1), &[]))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(json(&body)["text"], "hello world");

    let (status, body) = send(&app, post(multipart(&wav(1), &[("stream", "true")]))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("\"text\":\"hello world\""), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn oversized_body_is_413() {
    let app = server(engine(FakeAsr::hello_world(), 8))
        .with_max_body_bytes(4_096)
        .router();
    let (status, body) = send(&app, post(multipart(&wav(1), &[]))).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(json(&body)["error"]["type"], "invalid_request_error");
}

#[tokio::test(flavor = "multi_thread")]
async fn webm_is_415() {
    let webm = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/audio/tone.webm"
    ))
    .unwrap();
    let app = app(engine(FakeAsr::hello_world(), 8));
    let (status, body) = send(&app, post(multipart(&webm, &[]))).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE, "{body}");
    assert!(body.contains("transcode"), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_are_400() {
    let offline =
        FakeAsr::hello_world().with_capabilities(AsrCapabilities::new(SampleRate::HZ_16000));
    let app = app(engine(offline, 8));
    let cases = [
        multipart(&wav(1), &[("stream", "true")]),
        multipart(&wav(1), &[("response_format", "srt")]),
        multipart(b"not audio", &[]),
        multipart(&wav(1), &[("language", "zh")]),
        format!("--{BOUNDARY}--\r\n").into_bytes(),
    ];
    for body in cases {
        let (status, text) = send(&app, post(body)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    }
    let not_multipart = Request::post("/v1/audio/transcriptions")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, not_multipart).await.0, StatusCode::BAD_REQUEST);
}

fn blocked(gate: &Gate) -> FakeAsr {
    FakeAsr::new(
        Script::new()
            .then(
                Trigger::AfterSamples(1),
                Step::BlockUntilReleased(gate.clone()),
            )
            .then(Trigger::OnFinish, Step::Segment(0, "done")),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn saturated_engine_is_503() {
    let gate = Gate::new();
    let engine = engine(blocked(&gate), 1);
    let app = app(engine.clone());
    let first = tokio::spawn({
        let app = app.clone();
        async move { send(&app, post(multipart(&wav(1), &[]))).await }
    });
    assert!(gate.wait_entered(1, Duration::from_secs(10)));
    let (status, body) = send(&app, post(multipart(&wav(1), &[]))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    let (status, health) = send(&app, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&health)["active_sessions"], 1);
    assert_eq!(json(&health)["max_sessions"], 1);
    gate.release();
    let (status, body) = first.await.unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(json(&body)["text"], "done");
}

#[tokio::test(flavor = "multi_thread")]
async fn slow_engine_times_out_with_504() {
    let slow = FakeAsr::new(Script::new().then(
        Trigger::AfterSamples(1),
        Step::Sleep(Duration::from_secs(2)),
    ));
    let app = server(engine(slow, 8))
        .with_timeout(Duration::from_millis(200))
        .router();
    let started = std::time::Instant::now();
    let (status, body) = send(&app, post(multipart(&wav(1), &[]))).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT, "{body}");
    assert!(started.elapsed() < Duration::from_millis(1_900));
}

#[tokio::test(flavor = "multi_thread")]
async fn disconnect_cancels_the_session() {
    let gate = Gate::new();
    let fake = blocked(&gate);
    let stats = fake.stats();
    let engine = engine(fake, 8);
    let app = app(engine.clone());
    let request = tokio::spawn({
        let app = app.clone();
        async move { send(&app, post(multipart(&wav(1), &[]))).await }
    });
    assert!(gate.wait_entered(1, Duration::from_secs(10)));
    request.abort();
    let _ = request.await;
    gate.release();
    assert!(eventually(Duration::from_secs(10), || stats.cancelled() == 1));
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn sse_disconnect_cancels_the_session() {
    let gate = Gate::new();
    let fake = blocked(&gate);
    let stats = fake.stats();
    let engine = engine(fake, 8);
    let app = app(engine.clone());
    let response = app
        .clone()
        .oneshot(post(multipart(&wav(1), &[("stream", "true")])))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(gate.wait_entered(1, Duration::from_secs(10)));
    drop(response);
    gate.release();
    assert!(eventually(Duration::from_secs(10), || stats.cancelled() == 1));
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn health_and_models() {
    let app = app(engine(FakeAsr::hello_world(), 3));
    let (status, body) = send(&app, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let health = json(&body);
    assert_eq!(health["status"], "ok");
    assert_eq!(health["active_sessions"], 0);
    assert_eq!(health["max_sessions"], 3);
    let (status, body) = send(
        &app,
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(json(&body)["data"][0]["id"], "fake-model");
}

/// A router with no engine fails its health check.
#[tokio::test(flavor = "multi_thread")]
async fn a_server_without_engines_is_unhealthy() {
    let app = Server::new().router();
    let (status, body) = send(&app, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{body}");
    assert_eq!(json(&body)["status"], "no engine");
}

#[tokio::test(flavor = "multi_thread")]
async fn auth_requires_the_bearer_token() {
    let token = Arc::new(speechkit::Secret::new("let-me-in"));
    let app = server(engine(FakeAsr::hello_world(), 8))
        .with_auth(token)
        .router();
    let (status, body) = send(&app, post(multipart(&wav(1), &[]))).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{body}");
    let mut wrong = post(multipart(&wav(1), &[]));
    wrong
        .headers_mut()
        .insert("authorization", "Bearer nope".parse().unwrap());
    assert_eq!(send(&app, wrong).await.0, StatusCode::UNAUTHORIZED);
    let mut right = post(multipart(&wav(1), &[]));
    right
        .headers_mut()
        .insert("authorization", "Bearer let-me-in".parse().unwrap());
    assert_eq!(send(&app, right).await.0, StatusCode::OK);
    let health = Request::get("/health").body(Body::empty()).unwrap();
    assert_eq!(
        send(&app, health).await.0,
        StatusCode::OK,
        "health stays open"
    );
}

/// The `speechkit::cloud` client and this server speak the same protocol.
#[test]
fn interop_with_the_openai_http_client() {
    use speechkit::cloud::{CloudRuntime, OpenAiTranscription, OpenAiTranscriptionConfig};
    use speechkit::{asr::AsrOptions, vad::OfflineRecognizer};

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let running = runtime
        .block_on(
            server(engine(FakeAsr::hello_world(), 8))
                .with_bind("127.0.0.1:0")
                .start(std::future::pending()),
        )
        .unwrap();
    let base = format!("http://{}/v1", running.local_addr());
    let client_runtime = CloudRuntime::owned(1).unwrap();
    let opts = AsrOptions::default();
    for streaming in [false, true] {
        let config = OpenAiTranscriptionConfig::new("any")
            .with_endpoint(base.clone())
            .with_streaming(streaming);
        let client = OpenAiTranscription::new(config, client_runtime.clone()).unwrap();
        let text = client.recognize(&tone(16_000), &opts).unwrap();
        assert_eq!(text, "hello world", "streaming {streaming}");
    }
    runtime.block_on(running.shutdown()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn start_needs_an_engine() {
    let Err(error) = Server::new()
        .with_bind("127.0.0.1:0")
        .start(std::future::pending())
        .await
    else {
        panic!("a server with no engine started");
    };
    assert!(
        matches!(error, speechkit::SpeechError::InvalidInput(_)),
        "{error}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn saturated_engine_rejects_before_decoding() {
    let engine = engine(FakeAsr::hello_world(), 1);
    let session = engine
        .start(
            SampleRate::HZ_16000,
            speechkit::asr::AsrOptions::default(),
            Duration::from_secs(10),
        )
        .unwrap();
    let app = app(engine);
    // Invalid audio would return 400 if decoding ran before admission.
    let (status, _) = send(&app, post(multipart(b"not audio", &[]))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    drop(session);
}

#[tokio::test(flavor = "multi_thread")]
async fn upload_admission_is_bounded_and_released_on_disconnect() {
    let app = app(engine(FakeAsr::hello_world(), 1));
    let (polled, entered) = tokio::sync::oneshot::channel();
    let body = Body::from_stream(futures::stream::once(async move {
        polled.send(()).unwrap();
        std::future::pending::<Result<axum::body::Bytes, std::io::Error>>().await
    }));
    let mut request = post(Vec::new());
    *request.body_mut() = body;
    let pending = tokio::spawn(app.clone().oneshot(request));
    entered.await.unwrap();
    let (status, _) = send(&app, post(multipart(b"bad", &[]))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    pending.abort();
    assert!(pending.await.unwrap_err().is_cancelled());
    let (status, _) = send(&app, post(multipart(b"bad", &[]))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn websocket_open_runs_outside_tokio_tasks() {
    use speechkit::Secret;
    use speechkit::cloud::{CloudRuntime, OpenAiRealtime, OpenAiRealtimeConfig};
    // A real handshake gets 503. A nested-runtime panic would instead
    // produce 500 without reaching this local peer.
    let peer = Router::new().route(
        "/",
        axum::routing::get(|| async { StatusCode::SERVICE_UNAVAILABLE }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, peer).await.unwrap() });
    let backend = OpenAiRealtime::new(
        OpenAiRealtimeConfig::new("test", Arc::new(Secret::new("dummy"))).with_endpoint(&endpoint),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    let mut backends: Vec<Arc<dyn speechkit::asr::AsrBackend>> = vec![Arc::new(backend)];
    #[cfg(feature = "dashscope")]
    backends.push(Arc::new(
        speechkit::cloud::DashScopeAsr::new(
            speechkit::cloud::DashScopeAsrConfig::new("test", Arc::new(Secret::new("dummy")))
                .with_endpoint(&endpoint),
            CloudRuntime::owned(1).unwrap(),
        )
        .unwrap(),
    ));
    for backend in backends.drain(..) {
        let engine = AsrEngine::new(backend);
        let (status, body) = send(&app(engine), post(multipart(&wav(1), &[]))).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY, "{body}");
        assert!(body.contains("handshake returned HTTP 503"), "{body}");
    }
    server.abort();
}

#[tokio::test(flavor = "multi_thread")]
async fn http_sse_partials_arrive_before_done() {
    use futures::StreamExt;
    use speechkit::asr::AsrOptions;
    use speechkit::cloud::{CloudRuntime, OpenAiTranscription, OpenAiTranscriptionConfig};
    let (release, gate) = tokio::sync::oneshot::channel::<()>();
    let gate = Arc::new(tokio::sync::Mutex::new(Some(gate)));
    let (sent, received) = tokio::sync::oneshot::channel::<()>();
    let sent = Arc::new(tokio::sync::Mutex::new(Some(sent)));
    let mock = axum::Router::new().route(
        "/v1/audio/transcriptions",
        axum::routing::post(move || {
            let gate = gate.clone();
            let sent = sent.clone();
            async move {
                let first = futures::stream::once(async {
                    Ok::<_, std::io::Error>(
                        "data: {\"type\":\"transcript.text.delta\",\"delta\":\"hello\"}\n\n",
                    )
                });
                let last = futures::stream::once(async move {
                    sent.lock().await.take().unwrap().send(()).unwrap();
                    gate.lock().await.take().unwrap().await.unwrap();
                    Ok::<_, std::io::Error>(
                        "data: {\"type\":\"transcript.text.done\",\"text\":\"hello\"}\n\n",
                    )
                });
                (
                    [("content-type", "text/event-stream")],
                    Body::from_stream(first.chain(last)),
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
    let backend = OpenAiTranscription::new(
        OpenAiTranscriptionConfig::new("test")
            .with_endpoint(format!("http://{addr}/v1"))
            .with_streaming(true),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    let engine = AsrEngine::new(backend);
    let session = engine
        .start(
            SampleRate::HZ_16000,
            AsrOptions::default(),
            Duration::from_secs(10),
        )
        .unwrap();
    let mut observer = session.updates();
    session.try_push(vec![0.0; 1600]).unwrap();
    session.close_input();
    received.await.unwrap();
    let update = observer.recv(Duration::from_secs(5));
    assert!(
        matches!(update, Ok(speechkit::asr::AsrUpdate::Partial(ref p)) if p.text == "hello"),
        "{update:?}"
    );
    release.send(()).unwrap();
    assert!(session.finish(Duration::from_secs(3)).is_ok());
    server.abort();
}
