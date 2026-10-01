//! Fake VAD and offline recognizer, for testing `VadBackend`.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use speechkit::{
    SampleRate, SpeechError,
    asr::{AsrCapabilities, AsrOptions},
    vad::{OfflineRecognizer, SpeechSegment, Vad, VadModel},
};

/// A VAD that cuts the stream into fixed windows, each counted as one
/// utterance followed by a pause. It reports each window's speech only
/// when the window ends, so its start delay is one window. It ignores
/// `max_speech`.
#[derive(Debug, Clone)]
pub struct WindowVad {
    rate: SampleRate,
    window: usize,
}

impl WindowVad {
    /// Windows of `window` samples at 16 kHz.
    pub fn new(window: usize) -> Self {
        Self {
            rate: SampleRate::HZ_16000,
            window: window.max(1),
        }
    }
}

struct WindowStream {
    rate: SampleRate,
    window: usize,
    buffer: Vec<f32>,
    position: u64,
}

impl WindowStream {
    fn cut(&mut self, len: usize) -> SpeechSegment {
        let samples: Vec<f32> = self.buffer.drain(..len).collect();
        let start = self.rate.duration_of(self.position);
        self.position += len as u64;
        SpeechSegment { start, samples }
    }
}

impl Vad for WindowStream {
    fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegment> {
        self.buffer.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.buffer.len() >= self.window {
            out.push(self.cut(self.window));
        }
        out
    }

    fn flush(&mut self) -> Vec<SpeechSegment> {
        if self.buffer.is_empty() {
            Vec::new()
        } else {
            vec![self.cut(self.buffer.len())]
        }
    }

    fn speaking_since(&self) -> Option<Duration> {
        None
    }
}

impl VadModel for WindowVad {
    fn sample_rate(&self) -> SampleRate {
        self.rate
    }

    fn start_delay(&self) -> Duration {
        self.rate.duration_of(self.window as u64)
    }

    fn create(&self, _max_speech: Duration) -> Result<Box<dyn Vad>, SpeechError> {
        Ok(Box::new(WindowStream {
            rate: self.rate,
            window: self.window,
            buffer: Vec::new(),
            position: 0,
        }))
    }
}

/// An offline recognizer that names each utterance by its length, and
/// returns nothing for utterances shorter than `min_samples`.
#[derive(Debug)]
pub struct CountingRecognizer {
    caps: AsrCapabilities,
    min_samples: usize,
    calls: Arc<AtomicUsize>,
}

impl CountingRecognizer {
    /// A recognizer at 16 kHz.
    pub fn new(min_samples: usize) -> Self {
        Self {
            caps: AsrCapabilities::new(SampleRate::HZ_16000),
            min_samples,
            calls: Arc::default(),
        }
    }

    /// How many times `recognize` was called, shared with clones of the
    /// returned counter.
    pub fn calls(&self) -> Arc<AtomicUsize> {
        self.calls.clone()
    }
}

impl OfflineRecognizer for CountingRecognizer {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn recognize(&self, samples: &[f32], _: &AsrOptions) -> Result<String, SpeechError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(if samples.len() < self.min_samples {
            String::new()
        } else {
            format!("n{}", samples.len())
        })
    }
}
