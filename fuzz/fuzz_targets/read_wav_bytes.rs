//! Any RIFF body must parse as a WAV or fail cleanly. The input follows a
//! RIFF/WAVE header, so every run takes the WAV reader.
#![no_main]

use libfuzzer_sys::fuzz_target;
use speechkit::audio::{self, DecodeLimits};

fuzz_target!(|data: &[u8]| {
    let size = u32::try_from(data.len() + 4).unwrap_or(u32::MAX);
    let mut wav = Vec::with_capacity(data.len() + 12);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(data);
    let _ = audio::decode(&wav, DecodeLimits::default());
});
