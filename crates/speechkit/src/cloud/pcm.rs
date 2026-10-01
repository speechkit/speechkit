//! PCM16 decoding shared by the synthesis backends.

use crate::SpeechError;

/// Turns PCM16 bytes into samples, keeping an odd trailing byte for the
/// next piece.
#[derive(Debug)]
pub(crate) struct Pcm16Decoder {
    backend: &'static str,
    carry: Option<u8>,
}

impl Pcm16Decoder {
    /// A decoder with nothing buffered, naming `backend` in its error.
    pub(crate) fn new(backend: &'static str) -> Self {
        Self {
            backend,
            carry: None,
        }
    }

    /// The samples completed by `bytes`.
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut rest = bytes;
        if let Some(low) = self.carry.take() {
            let Some((&high, tail)) = rest.split_first() else {
                self.carry = Some(low);
                return out;
            };
            out.push(sample([low, high]));
            rest = tail;
        }
        let (pairs, remainder) = rest.as_chunks::<2>();
        out.extend(pairs.iter().map(|&pair| sample(pair)));
        self.carry = remainder.first().copied();
        out
    }

    /// Checks that the stream ended on a sample boundary.
    ///
    /// # Errors
    ///
    /// A non-retryable backend error if half a sample is left.
    pub(crate) fn finish(self) -> Result<(), SpeechError> {
        match self.carry {
            None => Ok(()),
            Some(_) => Err(SpeechError::backend(
                self.backend,
                false,
                "the PCM stream ended in the middle of a sample",
            )),
        }
    }
}

fn sample(bytes: [u8; 2]) -> f32 {
    f32::from(i16::from_le_bytes(bytes)) / 32_768.0
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn decodes_extremes() {
        let mut decoder = Pcm16Decoder::new("test");
        let samples = decoder.push(&[0x00, 0x80, 0xff, 0x7f, 0x00, 0x00]);
        assert_eq!(samples, [-1.0, 32_767.0 / 32_768.0, 0.0]);
        decoder.finish().unwrap();
    }

    #[test]
    fn half_a_sample_at_the_end_is_an_error() {
        let mut decoder = Pcm16Decoder::new("test");
        assert_eq!(decoder.push(&[1, 2, 3]).len(), 1);
        assert!(decoder.finish().is_err());
    }

    proptest! {
        /// Any split of the byte stream gives the same samples.
        #[test]
        fn chunking_does_not_matter(
            bytes in proptest::collection::vec(any::<u8>(), 0..200).prop_map(|mut b| {
                b.truncate(b.len() / 2 * 2);
                b
            }),
            cuts in proptest::collection::vec(0usize..200, 0..10),
        ) {
            let whole = Pcm16Decoder::new("test").push(&bytes);
            let mut cuts: Vec<usize> = cuts.into_iter().map(|c| c.min(bytes.len())).collect();
            cuts.sort_unstable();
            let mut decoder = Pcm16Decoder::new("test");
            let mut pieces = Vec::new();
            let mut start = 0;
            for cut in cuts.into_iter().chain([bytes.len()]) {
                pieces.extend(decoder.push(&bytes[start..cut]));
                start = cut;
            }
            prop_assert_eq!(pieces, whole);
            prop_assert!(decoder.finish().is_ok());
        }
    }
}
