//! DashScope synthesis against a local replay server.
#![cfg(feature = "dashscope")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::Arc, time::Duration};

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use speechkit::cloud::{CloudRuntime, DashScopeTts, DashScopeTtsConfig};
use speechkit::{
    SampleRate, Secret, SpeechError,
    tts::{TtsEngine, TtsOptions, Voice},
};
use speechkit_testkit::{contract::tts::run_tts_contract, secs};
use tokio_tungstenite::tungstenite::Message;

/// Samples of audio the server makes per character of text.
const PER_CHAR: usize = 100;

/// Emulates synthesis tasks on one connection: for each task, audio
/// proportional to the text, in frames that split samples in half, then
/// `task-finished`. Text containing `FAIL` fails the task instead.
async fn connection(stream: tokio::net::TcpStream) {
    let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
        return;
    };
    let mut text = String::new();
    while let Some(Ok(message)) = ws.next().await {
        let Message::Text(message) = message else {
            continue;
        };
        let request: Value = serde_json::from_str(&message).unwrap();
        let task_id = request["header"]["task_id"].as_str().unwrap().to_owned();
        let mut replies = Vec::new();
        match request["header"]["action"].as_str().unwrap() {
            "run-task" => {
                assert_eq!(request["payload"]["parameters"]["format"], "pcm");
                text.clear();
                replies.push(Message::Text(
                    json!({"header": {"event": "task-started", "task_id": "someone-else"}})
                        .to_string()
                        .into(),
                ));
                replies.push(Message::Text(
                    json!({"header": {"event": "task-started", "task_id": task_id}})
                        .to_string()
                        .into(),
                ));
            }
            "continue-task" => {
                text.push_str(request["payload"]["input"]["text"].as_str().unwrap());
            }
            "finish-task" => {
                if text.contains("FAIL") {
                    replies.push(Message::Text(
                        json!({"header": {"event": "task-failed", "task_id": task_id,
                            "error_code": "InvalidParameter", "error_message": "bad text"}})
                        .to_string()
                        .into(),
                    ));
                } else {
                    let bytes: Vec<u8> = (0..text.chars().count() * PER_CHAR)
                        .flat_map(|i| i16::try_from(i % 2_000).unwrap().to_le_bytes())
                        .collect();
                    for frame in bytes.chunks(333) {
                        replies.push(Message::Binary(frame.to_vec().into()));
                    }
                    replies.push(Message::Text(
                        json!({"header": {"event": "result-generated", "task_id": task_id}})
                            .to_string()
                            .into(),
                    ));
                    replies.push(Message::Text(
                        json!({"header": {"event": "task-finished", "task_id": task_id}})
                            .to_string()
                            .into(),
                    ));
                }
            }
            _ => {}
        }
        for reply in replies {
            if ws.send(reply).await.is_err() {
                return;
            }
        }
    }
}

struct Server {
    _runtime: tokio::runtime::Runtime,
    url: String,
}

fn server() -> Server {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let url = format!(
        "ws://{}/api-ws/v1/inference/",
        listener.local_addr().unwrap()
    );
    runtime.spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(connection(stream));
        }
    });
    Server {
        _runtime: runtime,
        url,
    }
}

fn engine(url: &str, runtime: &CloudRuntime) -> TtsEngine {
    let voices = vec![Voice::new("longxiaochun_v2"), Voice::new("longwan_v2")];
    let config = DashScopeTtsConfig::new("cosyvoice-v2", Arc::new(Secret::new("sk-test")), voices)
        .with_endpoint(url)
        .with_sample_rate(SampleRate::HZ_16000)
        .with_timeout(Duration::from_secs(10));
    let backend = DashScopeTts::new(config, runtime.clone()).unwrap();
    TtsEngine::new(backend)
}

#[test]
fn contract_offline() {
    let server = server();
    let runtime = CloudRuntime::owned(2).unwrap();
    run_tts_contract(|| engine(&server.url, &runtime));
}

#[test]
fn each_chunk_gets_exactly_its_audio() {
    let server = server();
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let text = "第一句。Second sentence! 第三句。";
    let audio = engine
        .synthesize(
            text,
            TtsOptions::default().with_voice("longwan_v2"),
            secs(30),
        )
        .unwrap();
    // Chunks concatenate back to the input, whitespace included.
    let chars = text.chars().count();
    assert_eq!(audio.samples.len(), chars * PER_CHAR);
    assert_eq!(audio.sample_rate, SampleRate::HZ_16000);
}

#[test]
fn a_failed_task_fails_the_session_and_keeps_progress() {
    let server = server();
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let failure = engine
        .synthesize(
            "Fine first. Then FAIL here.",
            TtsOptions::default(),
            secs(30),
        )
        .unwrap_err();
    assert!(matches!(failure.error, SpeechError::Backend { .. }));
    assert!(!failure.error.retryable());
    // The first chunk, "Fine first. ", was synthesized before the failure.
    assert_eq!(
        failure.duration,
        SampleRate::HZ_16000.duration_of(12 * PER_CHAR as u64)
    );
}

#[test]
fn bad_settings_are_rejected() {
    let runtime = CloudRuntime::owned(1).unwrap();
    let key = Arc::new(Secret::new("sk"));
    let voices = vec![Voice::new("v")];
    let config =
        |url: &str| DashScopeTtsConfig::new("m", key.clone(), voices.clone()).with_endpoint(url);
    assert!(DashScopeTts::new(config("http://not-ws"), runtime.clone()).is_err());
    assert!(
        DashScopeTts::new(
            DashScopeTtsConfig::new("m", key.clone(), Vec::new()),
            runtime.clone()
        )
        .is_err()
    );
    assert!(DashScopeTts::new(config("wss://h/x").with_timeout(Duration::ZERO), runtime).is_err());
}

#[test]
#[ignore = "needs DASHSCOPE_API_KEY and network"]
fn live_dashscope_tts() {
    let Some(key) = std::env::var("DASHSCOPE_API_KEY")
        .ok()
        .filter(|k| !k.is_empty())
    else {
        return;
    };
    let config = DashScopeTtsConfig::new(
        "cosyvoice-v2",
        Arc::new(Secret::new(key)),
        vec![Voice::new("longxiaochun_v2")],
    );
    let engine =
        TtsEngine::new(DashScopeTts::new(config, CloudRuntime::owned(2).unwrap()).unwrap());
    let audio = engine
        .synthesize(
            "你好，欢迎使用 speechkit。",
            TtsOptions::default(),
            secs(60),
        )
        .unwrap();
    assert!(
        audio.samples.len() > 11_025,
        "{} samples",
        audio.samples.len()
    );
}
