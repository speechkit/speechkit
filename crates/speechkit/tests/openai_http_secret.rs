//! The API key never appears in log output, at any level.
#![cfg(feature = "openai")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use speechkit::cloud::{CloudRuntime, OpenAiTranscription, OpenAiTranscriptionConfig};
use speechkit::{Secret, asr::AsrOptions, vad::OfflineRecognizer};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<u8>>>);

impl Write for Captured {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn secret_is_never_logged() {
    let captured = Captured::default();
    let writer = captured.clone();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::TRACE)
            .with_writer(move || writer.clone())
            .finish(),
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let server = runtime.block_on(MockServer::start());
    runtime.block_on(
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_string("sk-very-secret-key rejected"))
            .mount(&server),
    );
    let config = OpenAiTranscriptionConfig::new("m")
        .with_endpoint(format!("{}/v1", server.uri()))
        .with_api_key(Arc::new(Secret::new("sk-very-secret-key")));
    tracing::info!(?config, "configured");
    let backend = OpenAiTranscription::new(config, CloudRuntime::owned(1).unwrap()).unwrap();
    tracing::info!(?backend, "built");
    let error = backend
        .recognize(&[0.0; 1_600], &AsrOptions::default())
        .unwrap_err();
    tracing::error!(%error, source = %std::error::Error::source(&error).unwrap(), "failed");
    let logs = String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    assert!(logs.contains("uploading audio"), "{logs}");
    assert!(logs.contains("Secret(***)"), "{logs}");
    assert!(!logs.contains("sk-very-secret-key"), "{logs}");
}
