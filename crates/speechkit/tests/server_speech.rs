//! In-memory tests of `POST /v1/audio/speech`.
#![cfg(all(feature = "server", feature = "openai"))]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::time::Duration;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode},
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use speechkit::server::Server;
use speechkit::{asr::AsrEngine, tts::TtsEngine};
use speechkit_testkit::{
    Gate,
    asr::FakeAsr,
    eventually,
    tts::{FakeTts, SAMPLES_PER_CHAR, TtsStep, TtsTrigger},
};
use tower::ServiceExt;

fn tts(fake: FakeTts, sessions: usize) -> TtsEngine {
    TtsEngine::new(fake).with_max_sessions(sessions)
}

fn server(tts: TtsEngine) -> Server {
    let asr = AsrEngine::new(FakeAsr::hello_world());
    Server::new()
        .with_asr(asr, "fake-asr")
        .with_tts(tts, "fake-tts")
}

fn app(tts: TtsEngine) -> Router {
    server(tts).router()
}

fn post(body: &Value) -> Request<Body> {
    Request::post("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn send(app: &Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, body.to_vec())
}

const TEXT: &str = "Hello there. Second one.";

/// Samples at 24 kHz for `TEXT` from the 16 kHz fake.
fn expected_samples() -> f64 {
    let chars = "Hello there.".len() + "Second one.".len();
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let native = (chars * SAMPLES_PER_CHAR) as f64;
    native * 1.5
}

fn close_to_expected(samples: usize) -> bool {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let got = samples as f64;
    (got - expected_samples()).abs() <= expected_samples() * 0.02 + 16.0
}

#[tokio::test(flavor = "multi_thread")]
async fn pcm_and_wav_success() {
    let app = app(tts(FakeTts::plain(), 8));
    let response = app
        .clone()
        .oneshot(post(
            &json!({"model": "tts-1", "input": TEXT, "voice": "alpha", "response_format": "pcm"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "audio/pcm");
    assert_eq!(response.headers()["x-sample-rate"], "24000");
    let pcm = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(pcm.len() % 2, 0);
    assert!(
        close_to_expected(pcm.len() / 2),
        "{} samples",
        pcm.len() / 2
    );

    let (status, wav) = send(&app, post(&json!({"input": TEXT}))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[4..8], &[0xFF; 4], "placeholder size");
    assert!(close_to_expected((wav.len() - 44) / 2));

    // Missing and null optional fields keep the existing defaults.
    let (status, wav) = send(
        &app,
        post(&json!({"input": TEXT, "response_format": null, "speed": null, "voice": null})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(&wav[..4], b"RIFF");
}

#[tokio::test(flavor = "multi_thread")]
async fn bad_requests_are_400() {
    let app = server(tts(FakeTts::plain(), 8)).router();
    let cases = [
        json!({"input": "x".repeat(4097)}),
        json!({"input": "hi", "response_format": "mp3"}),
        json!({"input": "hi", "response_format": "opus"}),
        json!({"input": "hi", "response_format": 123}),
        json!({"input": "hi", "response_format": true}),
        json!({"input": "hi", "response_format": []}),
        json!({"input": "hi", "response_format": {}}),
        json!({"input": "hi", "voice": "robot"}),
        json!({"input": "hi", "speed": 9.0}),
        json!({"voice": "alpha"}),
    ];
    for case in cases {
        let (status, body) = send(&app, post(&case)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{case}");
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error", "{case}");
    }
    let (status, body) = send(&app, post(&json!({"input": "x".repeat(4097)}))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(String::from_utf8(body).unwrap().contains("limit is 4096"));
    let request = Request::post("/v1/audio/speech")
        .header("content-type", "application/json")
        .body(Body::from("{not json"))
        .unwrap();
    assert_eq!(send(&app, request).await.0, StatusCode::BAD_REQUEST);
}

#[tokio::test(flavor = "multi_thread")]
async fn saturated_engine_is_503() {
    let gate = Gate::new();
    let fake = FakeTts::new(vec![(
        TtsTrigger::OnOpen,
        TtsStep::BlockUntilReleased(gate.clone()),
    )]);
    let engine = tts(fake, 1);
    let app = app(engine.clone());
    let first = tokio::spawn({
        let app = app.clone();
        async move { send(&app, post(&json!({"input": TEXT}))).await }
    });
    assert!(gate.wait_entered(1, Duration::from_secs(10)));
    let (status, _) = send(&app, post(&json!({"input": TEXT}))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let (_, health) = send(&app, Request::get("/health").body(Body::empty()).unwrap()).await;
    let health: Value = serde_json::from_slice(&health).unwrap();
    assert_eq!(health["speech"]["active_sessions"], 1);
    gate.release();
    assert_eq!(first.await.unwrap().0, StatusCode::OK);
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn disconnect_cancels_synthesis() {
    let gate = Gate::new();
    let fake = FakeTts::new(vec![(
        TtsTrigger::BeforeChunk(1),
        TtsStep::BlockUntilReleased(gate.clone()),
    )]);
    let stats = fake.stats();
    let engine = tts(fake, 8);
    let app = app(engine.clone());
    let response = app
        .clone()
        .oneshot(post(&json!({"input": TEXT, "response_format": "pcm"})))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let first = body.frame().await.unwrap().unwrap();
    assert!(first.is_data());
    assert!(gate.wait_entered(1, Duration::from_secs(10)));
    drop(body);
    gate.release();
    assert!(eventually(Duration::from_secs(10), || stats.cancelled() == 1));
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
}

#[tokio::test(flavor = "multi_thread")]
async fn models_lists_both_engines() {
    let app = app(tts(FakeTts::plain(), 8));
    let (status, body) = send(
        &app,
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let models: Value = serde_json::from_slice(&body).unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["fake-asr", "fake-tts"]);
}

#[tokio::test(flavor = "multi_thread")]
async fn without_a_speech_engine_it_is_404() {
    let asr = AsrEngine::new(FakeAsr::hello_world());
    let app = Server::new().with_asr(asr, "fake-asr").router();
    assert_eq!(
        send(&app, post(&json!({"input": "hi"}))).await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_speech_only_server_lists_only_speech() {
    let app = Server::new()
        .with_tts(tts(FakeTts::plain(), 2), "fake-tts")
        .router();
    let transcription = Request::post("/v1/audio/transcriptions")
        .header("content-type", "multipart/form-data; boundary=X")
        .body(Body::from("--X--\r\n"))
        .unwrap();
    let (status, body) = send(&app, transcription).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let error: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
    let (status, body) = send(&app, Request::get("/health").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    let health: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["status"], "ok");
    assert!(health.get("model").is_none(), "{health}");
    assert_eq!(health["speech"]["model"], "fake-tts");
    assert_eq!(health["speech"]["max_sessions"], 2);
    let (_, body) = send(
        &app,
        Request::get("/v1/models").body(Body::empty()).unwrap(),
    )
    .await;
    let models: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(models["data"].as_array().unwrap().len(), 1, "{models}");
    assert_eq!(models["data"][0]["id"], "fake-tts");
    assert_eq!(
        send(&app, post(&json!({"input": "hi"}))).await.0,
        StatusCode::OK
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn auth_is_required() {
    use speechkit::Secret;

    let app = server(tts(FakeTts::plain(), 8))
        .with_auth(Secret::new("token"))
        .router();
    assert_eq!(
        send(&app, post(&json!({"input": "hi"}))).await.0,
        StatusCode::UNAUTHORIZED
    );
    let mut request = post(&json!({"input": "hi"}));
    request
        .headers_mut()
        .insert("authorization", "Bearer token".parse().unwrap());
    assert_eq!(send(&app, request).await.0, StatusCode::OK);
}

/// The `speechkit::cloud` OpenAI speech client and this server speak the
/// same protocol: PCM at 24 kHz, streamed.
#[test]
fn interop_with_the_openai_speech_client() {
    use speechkit::cloud::{CloudRuntime, OpenAiSpeech, OpenAiSpeechConfig};
    use speechkit::{
        SampleRate,
        tts::{TtsOptions, Voice},
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let server_tts = tts(FakeTts::plain(), 8);
    let running = runtime
        .block_on(
            server(server_tts.clone())
                .with_bind("127.0.0.1:0")
                .start(std::future::pending()),
        )
        .unwrap();
    let config = OpenAiSpeechConfig::new("any")
        .with_endpoint(format!("http://{}/v1", running.local_addr()))
        .with_voices(vec![Voice::new("alpha"), Voice::new("beta")]);
    let client =
        TtsEngine::new(OpenAiSpeech::new(config, CloudRuntime::owned(1).unwrap()).unwrap());
    let audio = client
        .synthesize(
            TEXT,
            TtsOptions::default().with_voice("beta"),
            speechkit_testkit::secs(30),
        )
        .unwrap();
    assert_eq!(audio.sample_rate, SampleRate::HZ_24000);
    assert!(
        close_to_expected(audio.samples.len()),
        "{}",
        audio.samples.len()
    );
    assert!(audio.samples.iter().any(|s| s.abs() > 0.0005));
    assert!(eventually(Duration::from_secs(10), || server_tts
        .active_sessions()
        == 0));
    runtime.block_on(running.shutdown()).unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn unread_response_expires_and_releases_synthesis_slot() {
    let engine = tts(FakeTts::plain(), 1);
    let app = server(engine.clone())
        .with_timeout(Duration::from_millis(300))
        .router();
    let response = app
        .oneshot(post(
            &json!({"input":"a".repeat(4000),"response_format":"pcm"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Keep the response alive without consuming it, filling both queues.
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
    assert!(
        response.into_body().collect().await.is_err(),
        "a timeout must truncate the body"
    );
}

#[cfg(feature = "dashscope")]
#[tokio::test(flavor = "multi_thread")]
async fn dashscope_open_runs_outside_tokio_tasks() {
    use speechkit::cloud::{CloudRuntime, DashScopeTts, DashScopeTtsConfig};
    use speechkit::{Secret, tts::Voice};
    use std::sync::Arc;
    let peer = Router::new().route(
        "/",
        axum::routing::get(|| async { StatusCode::SERVICE_UNAVAILABLE }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("ws://{}/", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, peer).await.unwrap() });
    let backend = DashScopeTts::new(
        DashScopeTtsConfig::new(
            "test",
            Arc::new(Secret::new("dummy")),
            vec![Voice::new("v")],
        )
        .with_endpoint(endpoint),
        CloudRuntime::owned(1).unwrap(),
    )
    .unwrap();
    let engine = TtsEngine::new(backend);
    let (status, body) = send(&app(engine), post(&json!({"input":"hi"}))).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert!(
        String::from_utf8(body)
            .unwrap()
            .contains("handshake returned HTTP 503")
    );
    server.abort();
}
