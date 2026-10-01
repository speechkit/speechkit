//! Speech synthesis against real models. Ignored by
//! default; run with `cargo test -p speechkit --features sherpa -- --ignored` after
//! setting `SPEECHKIT_MODEL_TTS_*` to unpacked model directories.
//!
//! Each model runs the TTS contract suite, then synthesizes a fixed
//! sentence whose duration and RMS must fall inside the stored range. The
//! ranges are wide on purpose: they catch silence, noise, clipping, and a
//! wrong sample rate, not a change of voice.
#![cfg(feature = "sherpa")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{ops::RangeInclusive, path::PathBuf, time::Duration};

use speechkit::sherpa::TtsConfig;
use speechkit::tts::{TtsEngine, TtsOptions};
use speechkit_testkit::{contract::tts::run_tts_contract, gates::model_dir, rms, secs};

/// A model, the sentence it reads, and the accepted output.
struct Expected {
    id: &'static str,
    text: &'static str,
    seconds: RangeInclusive<f32>,
    rms: RangeInclusive<f32>,
}

const EXPECTED: &[Expected] = &[
    Expected {
        id: "tts-piper-en",
        text: "The quick brown fox jumps over the lazy dog.",
        seconds: 1.5..=5.0,
        rms: 0.02..=0.4,
    },
    Expected {
        id: "tts-kokoro-en",
        text: "The quick brown fox jumps over the lazy dog.",
        seconds: 1.5..=5.0,
        rms: 0.02..=0.4,
    },
    Expected {
        id: "tts-matcha-zh",
        text: "今天天气很好，我们一起去公园散步吧。",
        seconds: 2.0..=7.0,
        rms: 0.02..=0.4,
    },
];

fn engine(dir: &PathBuf) -> TtsEngine {
    let backend = TtsConfig::new(dir).load().unwrap();
    TtsEngine::new(backend)
}

fn check(expected: &Expected) {
    let Some(dir) = model_dir(expected.id) else {
        return;
    };
    run_tts_contract(|| engine(&dir));
    let engine = engine(&dir);
    let audio = engine
        .synthesize(expected.text, TtsOptions::default(), secs(120))
        .unwrap();
    let seconds =
        Duration::from_secs_f64(audio.samples.len() as f64 / f64::from(audio.sample_rate.hz()))
            .as_secs_f32();
    let level = rms(&audio.samples);
    assert!(
        expected.seconds.contains(&seconds),
        "{}: {seconds} s is outside {:?}",
        expected.id,
        expected.seconds
    );
    assert!(
        expected.rms.contains(&level),
        "{}: RMS {level} is outside {:?}",
        expected.id,
        expected.rms
    );
}

#[test]
#[ignore = "needs a TTS model; set SPEECHKIT_MODEL_TTS_PIPER_EN"]
fn piper_en() {
    check(&EXPECTED[0]);
}

#[test]
#[ignore = "needs a TTS model; set SPEECHKIT_MODEL_TTS_KOKORO_EN"]
fn kokoro_en() {
    check(&EXPECTED[1]);
}

#[test]
#[ignore = "needs a TTS model and vocoder; set SPEECHKIT_MODEL_TTS_MATCHA_ZH"]
fn matcha_zh() {
    check(&EXPECTED[2]);
}
