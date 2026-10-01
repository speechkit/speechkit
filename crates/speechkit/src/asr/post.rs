//! Post-processing of committed segments, such as punctuation.

use crate::SpeechError;

/// Rewrites the text of committed segments, for example to add
/// punctuation.
///
/// Configure it with [`AsrEngine::with_post_processor`](super::AsrEngine::with_post_processor).
/// The session worker applies it only to committed segments, never to
/// partial results. If it fails, the original text is kept and a warning
/// is logged; the session does not fail.
pub trait PostProcessor: Send + Sync + 'static {
    /// Returns the rewritten text.
    ///
    /// # Errors
    ///
    /// Any error; the session keeps the original text.
    fn process(&self, text: &str) -> Result<String, SpeechError>;
}

/// Applies a post-processor, remembering the last input so a segment whose
/// text repeats the previous one is not processed again.
pub(crate) struct PostProcessing {
    processor: std::sync::Arc<dyn PostProcessor>,
    last: Option<(String, String)>,
}

impl PostProcessing {
    pub(crate) fn new(processor: std::sync::Arc<dyn PostProcessor>) -> Self {
        Self {
            processor,
            last: None,
        }
    }

    pub(crate) fn apply(&mut self, text: String) -> String {
        if text.trim().is_empty() {
            return text;
        }
        if let Some((input, output)) = &self.last
            && *input == text
        {
            return output.clone();
        }
        let output = match self.processor.process(&text) {
            Ok(output) => output,
            Err(error) => {
                tracing::warn!(%error, "segment post-processing failed; keeping the original text");
                text.clone()
            }
        };
        self.last = Some((text, output.clone()));
        output
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct Upper(AtomicUsize);

    impl PostProcessor for Upper {
        fn process(&self, text: &str) -> Result<String, SpeechError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(text.to_uppercase())
        }
    }

    struct Broken;

    impl PostProcessor for Broken {
        fn process(&self, _: &str) -> Result<String, SpeechError> {
            Err(SpeechError::InvalidModel("broken".into()))
        }
    }

    #[test]
    fn uppercases_and_skips_repeats() {
        let upper = Arc::new(Upper(AtomicUsize::new(0)));
        let mut post = PostProcessing::new(upper.clone());
        assert_eq!(post.apply("hello".into()), "HELLO");
        assert_eq!(post.apply("hello".into()), "HELLO");
        assert_eq!(post.apply("  ".into()), "  ");
        assert_eq!(post.apply("world".into()), "WORLD");
        assert_eq!(upper.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn failure_keeps_original_text() {
        let mut post = PostProcessing::new(Arc::new(Broken));
        assert_eq!(post.apply("hello".into()), "hello");
    }
}
