//! Snapshots to append-only deltas, for the SSE wire.

/// Turns successive whole-transcript snapshots into append-only deltas.
///
/// `transcript.text.delta` events accumulate by appending, so a snapshot
/// can only be sent when it extends what was already sent. A snapshot
/// that revises sent text yields nothing; the builder waits until a later
/// snapshot extends the sent text again. The final
/// `transcript.text.done` event carries the authoritative text either way.
#[derive(Debug, Default, Clone)]
pub(crate) struct TextDeltaBuilder {
    sent: String,
}

impl TextDeltaBuilder {
    /// A builder that has sent nothing.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The delta to send for `snapshot`, if any.
    pub(crate) fn update(&mut self, snapshot: &str) -> Option<String> {
        let delta = snapshot.strip_prefix(self.sent.as_str())?;
        if delta.is_empty() {
            return None;
        }
        let delta = delta.to_owned();
        snapshot.clone_into(&mut self.sent);
        Some(delta)
    }

    /// Everything sent so far.
    #[cfg(test)]
    fn sent(&self) -> &str {
        &self.sent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_only_extend() {
        let mut builder = TextDeltaBuilder::new();
        assert_eq!(builder.update("hel").as_deref(), Some("hel"));
        assert_eq!(builder.update("hel"), None);
        assert_eq!(builder.update("hello").as_deref(), Some("lo"));
        assert_eq!(builder.update("help"), None);
        assert_eq!(builder.update("hello world").as_deref(), Some(" world"));
        assert_eq!(builder.sent(), "hello world");
        assert_eq!(builder.update("你好"), None);
    }
}
