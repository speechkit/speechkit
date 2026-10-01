//! DashScope recognition against a local replay server.
#![cfg(feature = "dashscope")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::Arc, time::Duration};

use futures::{SinkExt, StreamExt};
use serde_json::{Value, json};
use speechkit::cloud::{CloudRuntime, DashScopeAsr, DashScopeAsrConfig};
use speechkit::{
    Secret,
    asr::{AsrBackend, AsrEngine, AsrEvent, AsrUpdate},
};
use speechkit_testkit::{
    asr::Collected,
    contract::asr::{RATE, check_activity, options, run_asr_contract, tone},
    secs,
};
use tokio_tungstenite::tungstenite::Message;

/// What the replay server does besides the happy path.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    /// Fails the task at the first audio.
    Fail,
    /// Answers the first audio with a sentence 300 ms later.
    Late,
    /// Drops the connection at the first audio.
    Drop,
}

/// Emulates a recognition task: a partial and a final sentence for every
/// second of audio, a final sentence for the tail, and `task-finished`.
async fn task(stream: tokio::net::TcpStream, mode: Mode) {
    let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
        return;
    };
    let mut task_id = String::new();
    let mut samples = 0_usize;
    let mut sentences = 0_u32;
    let mut heard = 0_usize;
    while let Some(Ok(message)) = ws.next().await {
        let mut replies = Vec::new();
        match message {
            Message::Text(text) => {
                let request: Value = serde_json::from_str(&text).unwrap();
                request["header"]["task_id"]
                    .as_str()
                    .unwrap()
                    .clone_into(&mut task_id);
                match request["header"]["action"].as_str().unwrap() {
                    "run-task" => {
                        // An event for another task must be ignored.
                        replies.push(
                            json!({"header": {"event": "task-started", "task_id": "someone-else"}}),
                        );
                        replies
                            .push(json!({"header": {"event": "task-started", "task_id": task_id}}));
                    }
                    "finish-task" => {
                        if samples > heard {
                            replies.push(sentence(
                                &task_id,
                                &mut sentences,
                                "tail",
                                heard,
                                samples,
                            ));
                        }
                        replies.push(
                            json!({"header": {"event": "task-finished", "task_id": task_id}}),
                        );
                    }
                    _ => {}
                }
            }
            Message::Binary(audio) => {
                match mode {
                    Mode::Fail => replies.push(json!({"header": {"event": "task-failed", "task_id": task_id, "error_code": "InvalidParameter", "error_message": "bad audio"}})),
                    Mode::Drop => return,
                    Mode::Late if samples == 0 => {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        replies.push(sentence(&task_id, &mut sentences, "late", 0, 1_600));
                    }
                    Mode::Late | Mode::Normal => {}
                }
                samples += audio.len() / 2;
                while samples - heard >= 16_000 {
                    // Speech from 200 ms into each second to its end.
                    let begin = heard + 3_200;
                    heard += 16_000;
                    replies.push(json!({"header": {"event": "result-generated", "task_id": task_id},
                        "payload": {"output": {"sentence": {"text": "sec", "sentence_end": false, "begin_time": begin / 16}}}}));
                    let final_sentence = sentence(&task_id, &mut sentences, "second", begin, heard);
                    // A re-delivered final must be dropped.
                    replies.push(final_sentence.clone());
                    replies.push(final_sentence);
                    replies.push(
                        json!({"header": {"event": "result-generated", "task_id": task_id},
                        "payload": {"output": {"sentence": {"heartbeat": true}}}}),
                    );
                }
            }
            _ => continue,
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

fn sentence(task_id: &str, count: &mut u32, text: &str, begin: usize, end: usize) -> Value {
    *count += 1;
    json!({"header": {"event": "result-generated", "task_id": task_id},
    "payload": {"output": {"sentence": {
        "text": text, "sentence_end": true, "sentence_id": count,
        "begin_time": begin / 16, "end_time": end / 16,
    }}}})
}

struct Server {
    _runtime: tokio::runtime::Runtime,
    url: String,
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
    let url = format!(
        "ws://{}/api-ws/v1/inference/",
        listener.local_addr().unwrap()
    );
    runtime.spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(task(stream, mode));
        }
    });
    Server {
        _runtime: runtime,
        url,
    }
}

fn engine(url: &str, runtime: &CloudRuntime) -> AsrEngine {
    let config =
        DashScopeAsrConfig::new("paraformer-realtime-v2", Arc::new(Secret::new("sk-test")))
            .with_endpoint(url)
            .with_timeout(Duration::from_secs(10));
    let backend = DashScopeAsr::new(config, runtime.clone()).unwrap();
    AsrEngine::new(backend)
}

#[test]
fn contract_offline() {
    let server = server(Mode::Normal);
    let runtime = CloudRuntime::owned(2).unwrap();
    run_asr_contract(|| engine(&server.url, &runtime));
}

#[test]
fn sentences_commit_once_in_order() {
    let server = server(Mode::Normal);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    for _ in 0..25 {
        session.push(tone(1_600), secs(10)).unwrap();
    }
    let result = session.finish(secs(20));
    let outcome = result.as_ref().unwrap();
    assert_eq!(outcome.text(), "second second tail");
    let segments = &outcome.segments;
    assert_eq!(segments[1].end, Duration::from_secs(2));
    assert!(segments.windows(2).all(|w| w[0].utterance < w[1].utterance));
}

#[test]
fn task_failure_fails_the_session() {
    let server = server(Mode::Fail);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    let _ = session.push(tone(1_600), secs(10));
    let result = session.finish(secs(10));
    let failure = result.as_ref().unwrap_err();
    let message = std::error::Error::source(&failure.error)
        .unwrap()
        .to_string();
    assert!(message.contains("InvalidParameter: bad audio"), "{message}");
}

/// The activity lag is longer than the 2.5 s clip. Between sentences,
/// activity is known up to the audio sent minus the lag, which assumes the
/// service keeps up with the audio, and this test pushes the clip at once.
/// With the long lag, activity is known only up to the last sentence's end
/// until the task finishes, however slow the replay server is. The lag
/// arithmetic is tested in `state.rs`.
#[test]
fn sentences_report_speech_activity() {
    let server = server(Mode::Normal);
    let config = DashScopeAsrConfig::new("paraformer-realtime-v2", Arc::new(Secret::new("k")))
        .with_endpoint(&server.url)
        .with_activity_lag(Duration::from_secs(5))
        .with_timeout(Duration::from_secs(10));
    let backend = DashScopeAsr::new(config, CloudRuntime::owned(1).unwrap()).unwrap();
    assert!(backend.capabilities().reports_activity);
    let sent = Collected::default();
    let mut stream = backend.open(&options(), sent.events()).unwrap();
    for _ in 0..25 {
        stream.accept(&tone(1_600)).unwrap();
    }
    stream.finish().unwrap();
    drop(stream);
    let events = sent.take();
    check_activity(&events);
    let activity: Vec<String> = events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::SpeechStarted { at } => Some(format!("started {}", at.as_millis())),
            AsrEvent::SpeechEnded { at, utterance } => {
                Some(format!("ended {} {}", at.as_millis(), utterance.0))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        activity,
        [
            "started 200",
            "ended 1000 0",
            "started 1200",
            "ended 2000 1",
            "started 2000",
            "ended 2500 2"
        ]
    );
    assert_eq!(
        events.last(),
        Some(&AsrEvent::ActivityKnown {
            through: Duration::from_millis(2_500)
        })
    );
}

#[test]
fn events_arrive_while_the_session_is_idle() {
    let server = server(Mode::Late);
    let engine = engine(&server.url, &CloudRuntime::owned(1).unwrap());
    let session = engine.start(RATE, options(), secs(10)).unwrap();
    let mut updates = session.updates();
    session.push(tone(1_600), secs(10)).unwrap();
    loop {
        match updates.recv(secs(10)).unwrap() {
            AsrUpdate::Segment(segment) => {
                assert_eq!(segment.text, "late");
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
    session.push(tone(1_600), secs(10)).unwrap();
    let result = session.wait(secs(10)).expect("the connection dropped");
    let failure = result.unwrap_err();
    assert!(failure.error.retryable(), "{}", failure.error);
}

/// Long-silence probe: 90 s of real-time silence must not
/// close the task, thanks to heartbeats.
#[test]
#[ignore = "needs DASHSCOPE_API_KEY and 90 s"]
fn dashscope_long_silence_survives() {
    let Ok(key) = std::env::var("DASHSCOPE_API_KEY") else {
        return;
    };
    let config = DashScopeAsrConfig::new("paraformer-realtime-v2", Arc::new(Secret::new(key)));
    let backend = DashScopeAsr::new(config, CloudRuntime::owned(2).unwrap()).unwrap();
    let engine = AsrEngine::new(backend);
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .unwrap();
    for _ in 0..900 {
        session.push(vec![0.0; 1_600], secs(10)).unwrap();
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(session.finish(secs(60)).is_ok());
}
