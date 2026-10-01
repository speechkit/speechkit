//! DashScope recognition messages: pure encoding and decoding.

use crate::SpeechError;
use serde_json::{Value, json};

pub(crate) use crate::cloud::dashscope::task::finish_task;
use crate::cloud::dashscope::task::{Header, header};

/// The backend name used in errors.
pub(crate) const BACKEND: &str = "dashscope-asr";

/// `run-task`, starting recognition of 16 kHz PCM16. Heartbeats keep long
/// silences from closing the task.
pub(crate) fn run_task(task_id: &str, model: &str, language: Option<&str>) -> Value {
    let mut parameters = json!({ "format": "pcm", "sample_rate": 16_000, "heartbeat": true });
    if let Some(language) = language {
        parameters["language_hints"] = json!([language]);
    }
    json!({
        "header": { "action": "run-task", "task_id": task_id, "streaming": "duplex" },
        "payload": {
            "task_group": "audio",
            "task": "asr",
            "function": "recognition",
            "model": model,
            "parameters": parameters,
            "input": {}
        }
    })
}

/// A recognized sentence.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Sentence {
    /// The text so far, or the final text.
    pub(crate) text: String,
    /// Whether the sentence is final.
    pub(crate) end: bool,
    /// The sentence number, if given.
    pub(crate) id: Option<f64>,
    /// Start and end in milliseconds, if given.
    pub(crate) begin_ms: Option<f64>,
    /// End in milliseconds, if given.
    pub(crate) end_ms: Option<f64>,
}

/// A server event for this task.
#[derive(Debug)]
pub(crate) enum Event {
    /// The task is running.
    Started,
    /// A sentence, partial or final.
    Result(Sentence),
    /// The task ended normally.
    Finished,
    /// The task failed.
    Failed(SpeechError),
    /// An event for another task, or nothing to act on.
    Ignore,
}

/// Decodes one event. Events whose `task_id` names another task are
/// ignored, as are heartbeats and sentences without text.
pub(crate) fn parse(value: &Value, task_id: &str) -> Event {
    match header(value, task_id, BACKEND) {
        Header::Started => Event::Started,
        Header::Finished => Event::Finished,
        Header::Failed(error) => Event::Failed(error),
        Header::Other("result-generated") => {
            let sentence = &value["payload"]["output"]["sentence"];
            if sentence["heartbeat"].as_bool() == Some(true) {
                return Event::Ignore;
            }
            let Some(text) = sentence["text"].as_str() else {
                return Event::Ignore;
            };
            Event::Result(Sentence {
                text: text.to_owned(),
                end: sentence["sentence_end"].as_bool() == Some(true),
                id: sentence["sentence_id"].as_f64(),
                begin_ms: sentence["begin_time"].as_f64(),
                end_ms: sentence["end_time"].as_f64(),
            })
        }
        Header::Other(_) | Header::Ignore => Event::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests() {
        let run = run_task("t1", "paraformer-realtime-v2", Some("zh"));
        assert_eq!(run["header"]["action"], "run-task");
        assert_eq!(run["payload"]["parameters"]["sample_rate"], 16_000);
        assert_eq!(run["payload"]["parameters"]["language_hints"][0], "zh");
        assert!(run_task("t", "m", None)["payload"]["parameters"]["language_hints"].is_null());
        assert_eq!(finish_task("t1")["header"]["action"], "finish-task");
    }

    #[test]
    fn events() {
        let event = |s: &str| parse(&serde_json::from_str(s).unwrap(), "mine");
        assert!(matches!(
            event(r#"{"header":{"event":"task-started","task_id":"mine"}}"#),
            Event::Started
        ));
        assert!(matches!(
            event(r#"{"header":{"event":"task-started","task_id":"other"}}"#),
            Event::Ignore
        ));
        assert!(matches!(
            event(r#"{"header":{"event":"task-finished"}}"#),
            Event::Finished
        ));
        let Event::Failed(error) = event(
            r#"{"header":{"event":"task-failed","error_code":"Throttling.RateQuota","error_message":"slow"}}"#,
        ) else {
            panic!("failed event");
        };
        assert!(error.retryable());
        let result = r#"{"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"text":"你好","sentence_end":true,"sentence_id":1,"begin_time":0,"end_time":900}}}}"#;
        let Event::Result(sentence) = event(result) else {
            panic!("result")
        };
        assert!(sentence.end);
        assert_eq!(sentence.end_ms, Some(900.0));
        assert!(matches!(
            event(
                r#"{"header":{"event":"result-generated"},"payload":{"output":{"sentence":{"heartbeat":true,"text":""}}}}"#
            ),
            Event::Ignore
        ));
        assert!(matches!(
            event(r#"{"header":{"event":"result-generated"},"payload":{}}"#),
            Event::Ignore
        ));
    }
}
