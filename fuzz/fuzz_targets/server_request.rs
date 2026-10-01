//! Any request body must get an HTTP response, never a panic or a hang.
//! The first byte picks the endpoint and content type;
//! the rest is the body, sent under a fixed multipart boundary so the
//! multipart parser sees well-framed but arbitrary parts.
#![no_main]

use std::sync::OnceLock;

use axum::{body::Body, http::Request};
use libfuzzer_sys::fuzz_target;
use speechkit::{asr::AsrEngine, server::Server, tts::TtsEngine};
use speechkit_testkit::{asr::FakeAsr, tts::FakeTts};
use tower::ServiceExt;

fn runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime")
    })
}

fn app() -> axum::Router {
    Server::new()
        .with_asr(AsrEngine::new(FakeAsr::hello_world()), "fake")
        .with_tts(TtsEngine::new(FakeTts::plain()), "fake-tts")
        .with_max_body_bytes(1 << 20)
        .router()
}

fuzz_target!(|data: &[u8]| {
    let Some((&selector, body)) = data.split_first() else {
        return;
    };
    let (path, content_type) = match selector % 3 {
        0 => ("/v1/audio/transcriptions", "multipart/form-data; boundary=X"),
        1 => ("/v1/audio/speech", "application/json"),
        _ => ("/v1/audio/transcriptions", "application/json"),
    };
    let request = Request::post(path)
        .header("content-type", content_type)
        .body(Body::from(body.to_vec()))
        .expect("request");
    let response = runtime().block_on(async {
        tokio::time::timeout(std::time::Duration::from_secs(5), app().oneshot(request)).await
    });
    let response = response.expect("the server hung").expect("infallible");
    assert!(response.status().as_u16() < 600);
});
