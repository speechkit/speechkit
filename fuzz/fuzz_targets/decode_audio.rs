//! Any input must decode or fail cleanly: no panic, no hang, bounded memory.
#![no_main]

use libfuzzer_sys::fuzz_target;
use speechkit::audio::{self, DecodeLimits};

fuzz_target!(|data: &[u8]| {
    let _ = audio::decode(data, DecodeLimits::default());
});
