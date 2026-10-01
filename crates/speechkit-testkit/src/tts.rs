//! A scripted fake speech synthesis backend.

use std::{
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use speechkit::{
    Flow, SampleRate, SpeechError,
    tts::{TtsBackend, TtsCapabilities, TtsOptions, TtsStream, Voice},
};

use crate::Gate;

/// Samples produced per character of a chunk: 10 ms at 16 kHz.
pub const SAMPLES_PER_CHAR: usize = 160;

/// The sample level of chunk `index` (0-based), so tests can check order.
pub fn level(index: usize) -> f32 {
    #[expect(clippy::cast_precision_loss, reason = "chunk indexes are small")]
    let value = 0.001 * (index % 900 + 1) as f32;
    value
}

/// How many samples the fake produces for `chunk`.
pub fn samples_for(chunk: &str) -> usize {
    chunk.trim().chars().count() * SAMPLES_PER_CHAR
}

/// When a scripted step runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsTrigger {
    /// When the stream opens.
    OnOpen,
    /// Before chunk `n` (0-based) is synthesized.
    BeforeChunk(usize),
}

/// What a scripted step does.
#[derive(Debug)]
pub enum TtsStep {
    /// Sleep.
    Sleep(Duration),
    /// Block until the gate is released.
    BlockUntilReleased(Gate),
    /// Fail with (a copy of) this error.
    Fail(SpeechError),
    /// Panic.
    Panic,
}

/// Counters shared by a fake and its streams.
#[derive(Debug, Default)]
pub struct FakeTtsStats {
    /// Streams opened.
    pub opened: AtomicUsize,
    /// `cancel` calls.
    pub cancelled: AtomicUsize,
    /// Chunks synthesized, in order.
    pub chunks: Mutex<Vec<String>>,
}

impl FakeTtsStats {
    /// Streams opened so far.
    pub fn opened(&self) -> usize {
        self.opened.load(Ordering::SeqCst)
    }

    /// `cancel` calls so far.
    pub fn cancelled(&self) -> usize {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// The chunks synthesized so far.
    pub fn chunks(&self) -> Vec<String> {
        self.chunks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// A fake that produces, for each chunk, `SAMPLES_PER_CHAR` samples per
/// character at [`level`] of the chunk's index, and runs a script.
#[derive(Clone)]
pub struct FakeTts {
    caps: TtsCapabilities,
    voices: Vec<Voice>,
    script: Arc<Vec<(TtsTrigger, TtsStep)>>,
    stats: Arc<FakeTtsStats>,
}

impl FakeTts {
    /// A streaming fake at 16 kHz with speed 0.5–2.0 and voices `alpha`
    /// (English) and `beta` (Chinese), running `script`.
    pub fn new(script: Vec<(TtsTrigger, TtsStep)>) -> Self {
        let mut caps = TtsCapabilities::new(SampleRate::HZ_16000, 1_000);
        caps.streams_audio = true;
        caps.speed = Some(0.5..=2.0);
        Self {
            caps,
            voices: vec![
                Voice::new("alpha").with_languages(["en"]),
                Voice::new("beta").with_languages(["zh"]),
            ],
            script: Arc::new(script),
            stats: Arc::default(),
        }
    }

    /// A fake without a script.
    pub fn plain() -> Self {
        Self::new(Vec::new())
    }

    /// Replaces the capabilities.
    #[must_use]
    pub fn with_capabilities(mut self, caps: TtsCapabilities) -> Self {
        self.caps = caps;
        self
    }

    /// The shared counters.
    pub fn stats(&self) -> Arc<FakeTtsStats> {
        self.stats.clone()
    }
}

impl std::fmt::Debug for FakeTts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeTts")
            .field("caps", &self.caps)
            .finish_non_exhaustive()
    }
}

#[expect(clippy::panic, reason = "the script asked for a panic")]
fn scripted_panic() -> ! {
    panic!("scripted panic")
}

fn run(script: &[(TtsTrigger, TtsStep)], trigger: TtsTrigger) -> Result<(), SpeechError> {
    for (when, step) in script {
        if *when != trigger {
            continue;
        }
        match step {
            TtsStep::Sleep(duration) => std::thread::sleep(*duration),
            TtsStep::BlockUntilReleased(gate) => gate.wait(),
            TtsStep::Fail(error) => return Err(error.clone()),
            TtsStep::Panic => scripted_panic(),
        }
    }
    Ok(())
}

impl TtsBackend for FakeTts {
    fn name(&self) -> &'static str {
        "fake-tts"
    }

    fn capabilities(&self) -> &TtsCapabilities {
        &self.caps
    }

    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn open(&self, _: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        self.stats.opened.fetch_add(1, Ordering::SeqCst);
        run(&self.script, TtsTrigger::OnOpen)?;
        Ok(Box::new(FakeStream {
            script: self.script.clone(),
            stats: self.stats.clone(),
            streaming: self.caps.streams_audio,
            index: 0,
        }))
    }
}

struct FakeStream {
    script: Arc<Vec<(TtsTrigger, TtsStep)>>,
    stats: Arc<FakeTtsStats>,
    streaming: bool,
    index: usize,
}

impl TtsStream for FakeStream {
    fn synthesize(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError> {
        run(&self.script, TtsTrigger::BeforeChunk(self.index))?;
        self.stats
            .chunks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(chunk.to_owned());
        let burst = vec![level(self.index); samples_for(chunk)];
        self.index += 1;
        if self.streaming {
            for piece in burst.chunks(SAMPLES_PER_CHAR) {
                if sink(piece) == Flow::Stop {
                    return Ok(());
                }
            }
        } else {
            let _ = sink(&burst);
        }
        Ok(())
    }

    fn cancel(&mut self) {
        self.stats.cancelled.fetch_add(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bursts_match_text_and_honor_stop() {
        let fake = FakeTts::plain();
        let mut stream = fake.open(&TtsOptions::default()).unwrap();
        let mut got = Vec::new();
        stream
            .synthesize(" abc ", &mut |audio| {
                got.extend_from_slice(audio);
                Flow::Continue
            })
            .unwrap();
        assert_eq!(got.len(), 3 * SAMPLES_PER_CHAR);
        assert!(got.iter().all(|&s| (s - level(0)).abs() < f32::EPSILON));
        let mut pieces = 0;
        stream
            .synthesize("defg", &mut |_| {
                pieces += 1;
                Flow::Stop
            })
            .unwrap();
        assert_eq!(pieces, 1);
        assert_eq!(fake.stats().chunks(), [" abc ", "defg"]);
        stream.cancel();
        assert_eq!(fake.stats().cancelled(), 1);
    }

    #[test]
    fn scripts_fail_block_and_panic() {
        let failing = FakeTts::new(vec![(
            TtsTrigger::OnOpen,
            TtsStep::Fail(SpeechError::Capacity),
        )]);
        assert!(failing.open(&TtsOptions::default()).is_err());
        let gate = Gate::new();
        let blocking = FakeTts::new(vec![
            (
                TtsTrigger::BeforeChunk(0),
                TtsStep::BlockUntilReleased(gate.clone()),
            ),
            (TtsTrigger::BeforeChunk(1), TtsStep::Panic),
        ]);
        let mut stream = blocking.open(&TtsOptions::default()).unwrap();
        let worker = std::thread::spawn(move || {
            let first = stream.synthesize("a", &mut |_| Flow::Continue);
            let second = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                stream.synthesize("b", &mut |_| Flow::Continue)
            }));
            (first.is_ok(), second.is_err())
        });
        assert!(gate.wait_entered(1, Duration::from_secs(5)));
        gate.release();
        assert_eq!(worker.join().unwrap(), (true, true));
        let whole =
            FakeTts::plain().with_capabilities(TtsCapabilities::new(SampleRate::HZ_16000, 10));
        let mut stream = whole.open(&TtsOptions::default()).unwrap();
        let mut calls = 0;
        stream
            .synthesize("abcd", &mut |_| {
                calls += 1;
                Flow::Continue
            })
            .unwrap();
        assert_eq!(calls, 1);
    }
}
