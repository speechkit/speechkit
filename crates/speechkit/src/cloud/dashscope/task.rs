//! The task envelope that recognition and synthesis share.

use crate::SpeechError;
use serde_json::{Value, json};

use crate::cloud::ws::server_error;

/// `finish-task`: no more input for this task.
pub(crate) fn finish_task(task_id: &str) -> Value {
    json!({
        "header": { "action": "finish-task", "task_id": task_id, "streaming": "duplex" },
        "payload": { "input": {} }
    })
}

/// A server event, decoded as far as the envelope allows.
pub(crate) enum Header<'a> {
    /// The task is running.
    Started,
    /// The task ended normally.
    Finished,
    /// The task failed.
    Failed(SpeechError),
    /// Another event of this task, by name.
    Other(&'a str),
    /// An event for another task, or one without a name.
    Ignore,
}

/// Decodes the header of one event for `task_id`, naming `backend` in
/// errors.
pub(crate) fn header<'a>(value: &'a Value, task_id: &str, backend: &str) -> Header<'a> {
    let header = &value["header"];
    if header["task_id"].as_str().is_some_and(|id| id != task_id) {
        return Header::Ignore;
    }
    match header["event"].as_str() {
        Some("task-started") => Header::Started,
        Some("task-finished") => Header::Finished,
        Some("task-failed") => {
            let code = header["error_code"].as_str().unwrap_or_default();
            let retryable = code.contains("Throttling") || code.contains("InternalError");
            Header::Failed(server_error(
                backend,
                "the task failed",
                &header["error_code"],
                &header["error_message"],
                retryable,
            ))
        }
        Some(other) => Header::Other(other),
        None => Header::Ignore,
    }
}
