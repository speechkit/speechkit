//! DashScope synthesis messages: pure encoding and decoding.
//!
//! One chunk of text is one task: `run-task`, then `continue-task` with
//! the text, then `finish-task`. Audio arrives as binary PCM16 frames
//! until `task-finished`, so a chunk's audio is exactly what arrives
//! between those events.

use crate::SpeechError;
use serde_json::{Value, json};

pub(crate) use crate::cloud::dashscope::task::finish_task;
use crate::cloud::dashscope::task::{Header, header};

/// The backend name used in errors.
pub(crate) const BACKEND: &str = "dashscope-tts";

/// `run-task`, starting synthesis of plain text into PCM16 at
/// `sample_rate`. `rate` is the speaking speed, 1.0 being normal.
pub(crate) fn run_task(
    task_id: &str,
    model: &str,
    voice: &str,
    sample_rate: u32,
    rate: f32,
) -> Value {
    json!({
        "header": { "action": "run-task", "task_id": task_id, "streaming": "duplex" },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": model,
            "parameters": {
                "text_type": "PlainText",
                "voice": voice,
                "format": "pcm",
                "sample_rate": sample_rate,
                "volume": 50,
                "rate": rate,
                "pitch": 1.0
            },
            "input": {}
        }
    })
}

/// `continue-task` carrying `text`.
pub(crate) fn continue_task(task_id: &str, text: &str) -> Value {
    json!({
        "header": { "action": "continue-task", "task_id": task_id, "streaming": "duplex" },
        "payload": { "input": { "text": text } }
    })
}

/// A server event for this task.
#[derive(Debug)]
pub(crate) enum Event {
    /// The task is running.
    Started,
    /// The task ended normally; all its audio has been sent.
    Finished,
    /// The task failed.
    Failed(SpeechError),
    /// An event for another task, or nothing to act on.
    Ignore,
}

/// Decodes one event. Events whose `task_id` names another task are
/// ignored, as is `result-generated`, since audio comes in binary frames.
pub(crate) fn parse(value: &Value, task_id: &str) -> Event {
    match header(value, task_id, BACKEND) {
        Header::Started => Event::Started,
        Header::Finished => Event::Finished,
        Header::Failed(error) => Event::Failed(error),
        Header::Other(_) | Header::Ignore => Event::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests() {
        let run = run_task("t1", "cosyvoice-v2", "longxiaochun_v2", 22_050, 1.5);
        assert_eq!(run["header"]["action"], "run-task");
        assert_eq!(run["payload"]["task"], "tts");
        assert_eq!(run["payload"]["function"], "SpeechSynthesizer");
        let parameters = &run["payload"]["parameters"];
        assert_eq!(parameters["voice"], "longxiaochun_v2");
        assert_eq!(parameters["format"], "pcm");
        assert_eq!(parameters["sample_rate"], 22_050);
        assert_eq!(parameters["rate"], 1.5);
        let text = continue_task("t1", "你好\"。");
        assert_eq!(text["header"]["action"], "continue-task");
        assert_eq!(text["payload"]["input"]["text"], "你好\"。");
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
            event(r#"{"header":{"event":"task-finished","task_id":"other"}}"#),
            Event::Ignore
        ));
        assert!(matches!(
            event(r#"{"header":{"event":"task-finished"}}"#),
            Event::Finished
        ));
        assert!(matches!(
            event(r#"{"header":{"event":"result-generated","task_id":"mine"}}"#),
            Event::Ignore
        ));
        let Event::Failed(error) = event(
            r#"{"header":{"event":"task-failed","error_code":"InvalidParameter","error_message":"bad voice"}}"#,
        ) else {
            panic!("failed event");
        };
        assert!(!error.retryable());
    }
}
