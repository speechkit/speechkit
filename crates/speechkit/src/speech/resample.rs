//! Incremental mono resampling.

use rubato::{
    Async, FixedAsync, Resampler as _, SincInterpolationParameters, WindowFunction,
    audioadapter_buffers::direct::InterleavedSlice,
};

use super::{SampleRate, SpeechError};

/// Input frames per filter call.
const BLOCK: usize = 256;

#[cfg(test)]
thread_local! {
    static FILTERS_BUILT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many resampling filters this thread has built, for tests of which
/// thread does that work.
#[cfg(test)]
pub(crate) fn filters_built_here() -> usize {
    FILTERS_BUILT.with(std::cell::Cell::get)
}

fn failure(error: impl std::error::Error + Send + Sync + 'static) -> SpeechError {
    SpeechError::backend("resampler", false, error)
}

struct Filter {
    rubato: Async<f32>,
    /// One staged input block.
    input: Vec<f32>,
    /// Scratch output for one block.
    output: Vec<f32>,
    /// Input frames waiting for a full block.
    pending: Vec<f32>,
    /// Filter delay still to discard from the start of the output.
    skip: usize,
}

/// Converts mono audio from one sample rate to another, incrementally.
///
/// Feed input with [`process`](Self::process), then call
/// [`flush`](Self::flush) once at the end. Output is appended to the
/// caller's buffer. The filter delay is removed, so after `flush` the
/// output holds exactly `round(input_frames × to / from)` frames, lined up
/// with the input. When the rates match, audio passes through unchanged.
pub struct Resampler {
    from: SampleRate,
    to: SampleRate,
    filter: Option<Filter>,
    input_frames: u64,
    output_frames: u64,
    flushed: bool,
}

impl Resampler {
    /// A resampler from `from` to `to`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Backend`] if the filter cannot be built.
    pub fn new(from: SampleRate, to: SampleRate) -> Result<Self, SpeechError> {
        let filter = if from == to {
            None
        } else {
            #[cfg(test)]
            FILTERS_BUILT.with(|built| built.set(built.get() + 1));
            let parameters = SincInterpolationParameters::new(256, WindowFunction::BlackmanHarris2)
                .f_cutoff(0.95)
                .oversampling_factor(256);
            let rubato = Async::<f32>::new_sinc(
                f64::from(to.hz()) / f64::from(from.hz()),
                1.0,
                &parameters,
                BLOCK,
                1,
                FixedAsync::Input,
            )
            .map_err(failure)?;
            Some(Filter {
                skip: rubato.output_delay(),
                output: vec![0.0; rubato.output_frames_max()],
                input: Vec::with_capacity(BLOCK),
                pending: Vec::with_capacity(BLOCK),
                rubato,
            })
        };
        Ok(Self {
            from,
            to,
            filter,
            input_frames: 0,
            output_frames: 0,
            flushed: false,
        })
    }

    /// The input rate.
    pub fn from_rate(&self) -> SampleRate {
        self.from
    }

    /// The output rate.
    pub fn to_rate(&self) -> SampleRate {
        self.to
    }

    /// Whether the rates match, so audio passes through unchanged.
    pub fn is_passthrough(&self) -> bool {
        self.filter.is_none()
    }

    /// Resamples `input`, appending finished output to `out`.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Closed`] after [`flush`](Self::flush), or
    /// [`SpeechError::Backend`] if the filter fails.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), SpeechError> {
        if self.flushed {
            return Err(SpeechError::Closed);
        }
        self.input_frames += input.len() as u64;
        if self.filter.is_none() {
            out.extend_from_slice(input);
            self.output_frames += input.len() as u64;
            return Ok(());
        }
        for piece in input.chunks(BLOCK) {
            self.filter_mut().pending.extend_from_slice(piece);
            while self.filter_mut().pending.len() >= BLOCK {
                let filter = self.filter_mut();
                filter.input.clear();
                filter.input.extend(filter.pending.drain(..BLOCK));
                self.run_block(out)?;
            }
        }
        Ok(())
    }

    /// Flushes the filter at end of input, appending the remaining output.
    /// Later calls do nothing.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Backend`] if the filter fails.
    pub fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), SpeechError> {
        if self.flushed {
            return Ok(());
        }
        self.flushed = true;
        if self.filter.is_none() || self.input_frames == 0 {
            return Ok(());
        }
        let filter = self.filter_mut();
        filter.input.clear();
        filter.input.append(&mut filter.pending);
        filter.input.resize(BLOCK, 0.0);
        self.run_block(out)?;
        let delay = self.filter_mut().rubato.output_delay() as u64;
        let delay_in_input = (delay * u64::from(self.from.hz())).div_ceil(u64::from(self.to.hz()));
        let blocks = delay_in_input.div_ceil(BLOCK as u64) + 2;
        for _ in 0..blocks {
            if self.output_frames >= self.wanted() {
                return Ok(());
            }
            self.filter_mut().input.fill(0.0);
            self.run_block(out)?;
        }
        Ok(())
    }

    /// Output frames owed for the input so far: `round(in × to / from)`.
    pub(crate) fn wanted(&self) -> u64 {
        let from = u128::from(self.from.hz());
        let wanted = (u128::from(self.input_frames) * u128::from(self.to.hz()) + from / 2) / from;
        u64::try_from(wanted).unwrap_or(u64::MAX)
    }

    fn filter_mut(&mut self) -> &mut Filter {
        // Only called on the resampling path, where the filter exists.
        self.filter
            .as_mut()
            .expect("filter exists whenever the rates differ")
    }

    /// Runs the staged block through the filter.
    fn run_block(&mut self, out: &mut Vec<f32>) -> Result<(), SpeechError> {
        let owed = self.wanted().saturating_sub(self.output_frames);
        let filter = self.filter_mut();
        let input = InterleavedSlice::new(&filter.input[..], 1, BLOCK).map_err(failure)?;
        let frames = filter.output.len();
        let mut output =
            InterleavedSlice::new_mut(&mut filter.output[..], 1, frames).map_err(failure)?;
        let (_, written) = filter
            .rubato
            .process_into_buffer(&input, &mut output, None)
            .map_err(failure)?;
        let skipped = filter.skip.min(written);
        filter.skip -= skipped;
        let count = (written - skipped).min(usize::try_from(owed).unwrap_or(usize::MAX));
        out.extend_from_slice(&filter.output[skipped..skipped + count]);
        self.output_frames += count as u64;
        Ok(())
    }
}

impl std::fmt::Debug for Resampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Resampler")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;

    use proptest::prelude::*;

    use super::*;

    /// `frames` samples of a sine at `hz`, amplitude 0.5.
    fn sine(rate: SampleRate, frames: usize, hz: f32) -> Vec<f32> {
        (0..frames)
            .map(|i| 0.5 * (TAU * hz * i as f32 / rate.hz() as f32).sin())
            .collect()
    }

    fn rms(samples: &[f32]) -> f32 {
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    fn resample_all(from: SampleRate, to: SampleRate, chunks: &[&[f32]]) -> Vec<f32> {
        let mut resampler = Resampler::new(from, to).unwrap();
        let mut out = Vec::new();
        for chunk in chunks {
            resampler.process(chunk, &mut out).unwrap();
        }
        resampler.flush(&mut out).unwrap();
        out
    }

    #[test]
    fn sine_44100_to_16000_keeps_length_and_level() {
        let input = sine(SampleRate::HZ_44100, 44_100, 440.0);
        let out = resample_all(SampleRate::HZ_44100, SampleRate::HZ_16000, &[&input]);
        assert!((15_998..=16_002).contains(&out.len()), "{}", out.len());
        let (a, b) = (rms(&input), rms(&out));
        assert!((a - b).abs() / a < 0.01, "rms {a} vs {b}");
    }

    #[test]
    fn upsampling_keeps_length() {
        let input = sine(SampleRate::HZ_16000, 8_000, 300.0);
        let out = resample_all(SampleRate::HZ_16000, SampleRate::HZ_24000, &[&input]);
        assert_eq!(out.len(), 12_000);
    }

    #[test]
    fn passthrough_is_bit_exact() {
        let input = sine(SampleRate::HZ_16000, 1_600, 440.0);
        let mut resampler = Resampler::new(SampleRate::HZ_16000, SampleRate::HZ_16000).unwrap();
        assert!(resampler.is_passthrough());
        let mut out = Vec::new();
        resampler.process(&input, &mut out).unwrap();
        resampler.flush(&mut out).unwrap();
        assert_eq!(out, input);
    }

    #[test]
    fn process_after_flush_is_closed() {
        let mut resampler = Resampler::new(SampleRate::HZ_48000, SampleRate::HZ_16000).unwrap();
        let mut out = Vec::new();
        resampler.flush(&mut out).unwrap();
        resampler.flush(&mut out).unwrap();
        assert!(out.is_empty());
        assert!(matches!(
            resampler.process(&[0.0], &mut out),
            Err(SpeechError::Closed)
        ));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        #[test]
        fn chunking_does_not_change_output(sizes in prop::collection::vec(1_usize..700, 1..40)) {
            let input = sine(SampleRate::HZ_44100, 13_230, 440.0);
            let whole = resample_all(SampleRate::HZ_44100, SampleRate::HZ_16000, &[&input]);
            let mut chunks = Vec::new();
            let mut rest = &input[..];
            for size in sizes.iter().cycle() {
                if rest.is_empty() {
                    break;
                }
                let (head, tail) = rest.split_at((*size).min(rest.len()));
                chunks.push(head);
                rest = tail;
            }
            let pieces = resample_all(SampleRate::HZ_44100, SampleRate::HZ_16000, &chunks);
            prop_assert_eq!(whole.len(), pieces.len());
            for (a, b) in whole.iter().zip(&pieces) {
                prop_assert!((a - b).abs() < 1e-5);
            }
        }
    }
}
