//! T-05: synthesized text never appears in logs above TRACE. Its own test
//! binary, because the log capture is process-wide.
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    io::Write,
    sync::{Arc, Mutex},
};

use speechkit::tts::TtsEngine;
use speechkit_testkit::{contract::tts::t05_no_text_logging, tts::FakeTts};

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
fn t05_no_text_logging_at_debug() {
    let captured = Captured::default();
    let writer = captured.clone();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_writer(move || writer.clone())
            .finish(),
    )
    .unwrap();
    let make = || TtsEngine::new(FakeTts::plain());
    let logs = || String::from_utf8(captured.0.lock().unwrap().clone()).unwrap();
    t05_no_text_logging(&make, &logs);
}
