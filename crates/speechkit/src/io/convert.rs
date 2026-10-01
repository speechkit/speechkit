//! Converting between device samples and mono f32. Pure and
//! allocation-free, so the audio callbacks can use it.

/// A sample format a device may deliver.
pub(crate) trait DeviceSample: Copy {
    /// The sample as f32 in [-1.0, 1.0].
    fn to_f32(self) -> f32;
}

impl DeviceSample for f32 {
    fn to_f32(self) -> f32 {
        if self.is_nan() {
            0.0
        } else {
            self.clamp(-1.0, 1.0)
        }
    }
}

impl DeviceSample for i16 {
    fn to_f32(self) -> f32 {
        f32::from(self) / 32_768.0
    }
}

impl DeviceSample for u16 {
    fn to_f32(self) -> f32 {
        (f32::from(self) - 32_768.0) / 32_768.0
    }
}

/// Averages each frame of `interleaved` audio with `channels` channels and
/// calls `emit` with the mono sample. A trailing partial frame is ignored.
pub(crate) fn mixdown<T: DeviceSample>(
    interleaved: &[T],
    channels: usize,
    mut emit: impl FnMut(f32),
) {
    let channels = channels.max(1);
    #[expect(clippy::cast_precision_loss, reason = "channel counts are tiny")]
    let scale = 1.0 / channels as f32;
    for frame in interleaved.chunks_exact(channels) {
        let sum: f32 = frame.iter().map(|s| s.to_f32()).sum();
        emit((sum * scale).clamp(-1.0, 1.0));
    }
}

/// A sample format a device may accept.
pub(crate) trait OutputSample: Copy {
    /// The device sample for `value`, clamped to [-1.0, 1.0]; NaN is
    /// silence.
    fn from_f32(value: f32) -> Self;
}

fn clamp(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(-1.0, 1.0)
    }
}

impl OutputSample for f32 {
    fn from_f32(value: f32) -> Self {
        clamp(value)
    }
}

impl OutputSample for i16 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the clamped value scales into i16's range"
    )]
    fn from_f32(value: f32) -> Self {
        (clamp(value) * 32_767.0).round() as i16
    }
}

impl OutputSample for u16 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the clamped value scales into u16's range"
    )]
    fn from_f32(value: f32) -> Self {
        (clamp(value) * 32_767.0 + 32_768.0).round() as u16
    }
}

/// Fills interleaved `out` with `channels` copies of each mono sample from
/// `next`. When `next` runs dry, the remaining frames are silence. Returns
/// the number of silent frames.
pub(crate) fn fill_interleaved<T: OutputSample>(
    out: &mut [T],
    channels: usize,
    mut next: impl FnMut() -> Option<f32>,
) -> usize {
    let silence = T::from_f32(0.0);
    let mut silent = 0;
    for frame in out.chunks_mut(channels.max(1)) {
        let value = if let Some(sample) = next() {
            T::from_f32(sample)
        } else {
            silent += 1;
            silence
        };
        frame.fill(value);
    }
    silent
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn mono<T: DeviceSample>(samples: &[T], channels: usize) -> Vec<f32> {
        let mut out = Vec::new();
        mixdown(samples, channels, |s| out.push(s));
        out
    }

    #[test]
    fn formats_map_to_the_unit_range() {
        assert!((i16::MIN.to_f32() + 1.0).abs() < f32::EPSILON);
        assert!(0_i16.to_f32().abs() < f32::EPSILON);
        assert!((u16::MIN.to_f32() + 1.0).abs() < f32::EPSILON);
        assert!(32_768_u16.to_f32().abs() < f32::EPSILON);
        assert!((2.0_f32.to_f32() - 1.0).abs() < f32::EPSILON);
        assert!(f32::NAN.to_f32().abs() < f32::EPSILON);
    }

    #[test]
    fn stereo_frames_are_averaged() {
        assert_eq!(mono(&[1.0_f32, 0.0, -0.5, -0.5, 0.25], 2), [0.5, -0.5]);
        assert_eq!(mono(&[0.1_f32, 0.2], 0), [0.1, 0.2]);
    }

    #[test]
    fn output_formats_cover_their_range() {
        assert_eq!(i16::from_f32(1.0), i16::MAX);
        assert_eq!(i16::from_f32(-2.0), -i16::MAX);
        assert_eq!(i16::from_f32(f32::NAN), 0);
        assert_eq!(u16::from_f32(0.0), 32_768);
        assert_eq!(u16::from_f32(1.0), u16::MAX);
        assert_eq!(u16::from_f32(-1.0), 1);
        assert!((f32::from_f32(3.0) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn filling_duplicates_channels_and_counts_silence() {
        let mut source = [0.5_f32, -0.5].into_iter();
        let mut out = [9_i16; 8];
        let silent = fill_interleaved(&mut out, 2, || source.next());
        assert_eq!(out, [16_384, 16_384, -16_384, -16_384, 0, 0, 0, 0]);
        assert_eq!(silent, 2);
    }

    proptest! {
        /// Output then input gives back the value, within one step.
        #[test]
        fn i16_round_trip(value in -1.0_f32..=1.0) {
            prop_assert!((i16::from_f32(value).to_f32() - value).abs() <= 1.0 / 16_384.0);
        }

        #[test]
        fn u16_round_trip(value in -1.0_f32..=1.0) {
            prop_assert!((u16::from_f32(value).to_f32() - value).abs() <= 1.0 / 16_384.0);
        }

        #[test]
        fn i16_output_is_bounded_and_counted(samples in prop::collection::vec(any::<i16>(), 0..600), channels in 1_usize..8) {
            let out = mono(&samples, channels);
            prop_assert_eq!(out.len(), samples.len() / channels);
            prop_assert!(out.iter().all(|s| (-1.0..=1.0).contains(s)));
        }

        #[test]
        fn u16_output_is_bounded(samples in prop::collection::vec(any::<u16>(), 0..600), channels in 1_usize..8) {
            prop_assert!(mono(&samples, channels).iter().all(|s| (-1.0..=1.0).contains(s)));
        }

        #[test]
        fn f32_output_is_finite_and_bounded(samples in prop::collection::vec(any::<f32>(), 0..600), channels in 1_usize..8) {
            prop_assert!(mono(&samples, channels).iter().all(|s| s.is_finite() && (-1.0..=1.0).contains(s)));
        }

        #[test]
        fn identical_channels_keep_the_value(values in prop::collection::vec(-1.0_f32..=1.0, 0..200), channels in 1_usize..6) {
            let interleaved: Vec<f32> = values.iter().flat_map(|&v| std::iter::repeat_n(v, channels)).collect();
            let out = mono(&interleaved, channels);
            for (a, b) in out.iter().zip(&values) {
                prop_assert!((a - b).abs() < 1e-5);
            }
        }
    }
}
