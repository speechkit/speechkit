//! A fake wake-word model, for testing watchings.

use std::time::Duration;

use speechkit::{
    SampleRate, SpeechError,
    wake::{WakeEvent, WakeWordDetector, WakeWordModel},
};

/// A wake-word model that hears a keyword wherever a sample equals
/// [`MARK`](Self::MARK). The keyword ends right after that sample and
/// starts [`LENGTH`](Self::LENGTH) before its end. Like a real detector, it
/// reports the keyword late: once it has heard `latency` more audio, or at
/// the end of the stream.
#[derive(Debug, Clone)]
pub struct FakeWakeWord {
    rate: SampleRate,
    latency: Duration,
    per_call: Duration,
}

impl FakeWakeWord {
    /// The sample that marks a keyword's last sample. Other audio in tests
    /// stays below it.
    pub const MARK: f32 = 1.0;
    /// How long each keyword is.
    pub const LENGTH: Duration = Duration::from_millis(300);
    /// The keyword it reports.
    pub const KEYWORD: &'static str = "hey kit";

    /// A model at 16 kHz that reports each keyword 300 ms after it ends.
    pub fn new() -> Self {
        Self {
            rate: SampleRate::HZ_16000,
            latency: Duration::from_millis(300),
            per_call: Duration::ZERO,
        }
    }

    /// Reports each keyword `latency` after it ends.
    #[must_use]
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Sleeps `per_call` in every `accept`, like a slow model.
    #[must_use]
    pub fn with_delay(mut self, per_call: Duration) -> Self {
        self.per_call = per_call;
        self
    }
}

impl Default for FakeWakeWord {
    fn default() -> Self {
        Self::new()
    }
}

impl WakeWordModel for FakeWakeWord {
    fn sample_rate(&self) -> SampleRate {
        self.rate
    }

    fn create(&self) -> Result<Box<dyn WakeWordDetector>, SpeechError> {
        Ok(Box::new(Detector {
            model: self.clone(),
            fed: 0,
            pending: Vec::new(),
        }))
    }
}

struct Detector {
    model: FakeWakeWord,
    /// Frames fed so far.
    fed: u64,
    /// Ends of keywords heard but not reported yet, in frames.
    pending: Vec<u64>,
}

impl Detector {
    fn event(&self, end: u64) -> WakeEvent {
        let end = self.model.rate.duration_of(end);
        WakeEvent {
            keyword: FakeWakeWord::KEYWORD.to_owned(),
            start: end.saturating_sub(FakeWakeWord::LENGTH),
            end,
        }
    }
}

impl WakeWordDetector for Detector {
    fn accept(&mut self, samples: &[f32]) -> Vec<WakeEvent> {
        std::thread::sleep(self.model.per_call);
        for (index, &sample) in samples.iter().enumerate() {
            if sample.to_bits() == FakeWakeWord::MARK.to_bits() {
                self.pending.push(self.fed + index as u64 + 1);
            }
        }
        self.fed += samples.len() as u64;
        let latency = self.model.rate.frames_in(self.model.latency);
        let due: Vec<u64> = self
            .pending
            .iter()
            .copied()
            .filter(|&end| end + latency <= self.fed)
            .collect();
        self.pending.retain(|&end| end + latency > self.fed);
        due.into_iter().map(|end| self.event(end)).collect()
    }

    fn flush(&mut self) -> Vec<WakeEvent> {
        std::mem::take(&mut self.pending)
            .into_iter()
            .map(|end| self.event(end))
            .collect()
    }
}
