//! Decoding rules, one test per rule.
//!
//! Fixtures come from `cargo xtask gen-audio-fixtures`. By default the
//! committed `fixtures/audio` directory is used; `SPEECHKIT_DECODE_FIXTURES`
//! points elsewhere (CI regenerates them with its own ffmpeg). A missing
//! file prints "skipped" and passes, unless `SPEECHKIT_REQUIRE_FIXTURES` is
//! set, as it is in CI, where a skip fails.
#![cfg(feature = "decode")]
#![expect(clippy::panic, reason = "test helpers fail the calling test")]

use std::{path::PathBuf, time::Duration};

use speechkit::{
    AudioBuffer, SpeechError,
    audio::{self, DecodeLimits, EXTENSIONS},
};
use speechkit_testkit::rms;

const SOURCE_FRAMES: usize = 48_000;
const LOSSY_FRAMES_TOLERANCE: usize = 4_800;

fn fixture(name: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("SPEECHKIT_DECODE_FIXTURES").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/audio"),
        PathBuf::from,
    );
    let path = dir.join(name);
    if path.is_file() {
        return Some(path);
    }
    assert!(
        std::env::var_os("SPEECHKIT_REQUIRE_FIXTURES").is_none(),
        "fixture {} is missing",
        path.display()
    );
    eprintln!("skipped: fixture {} is missing", path.display());
    None
}

fn load(name: &str) -> Option<AudioBuffer> {
    fixture(name).map(|path| {
        audio::read(&path, DecodeLimits::default()).unwrap_or_else(|e| panic!("{name}: {e}"))
    })
}

fn error(name: &str) -> Option<String> {
    fixture(name).map(|path| match audio::read(&path, DecodeLimits::default()) {
        Err(SpeechError::InvalidInput(message)) => message,
        other => panic!("{name}: expected InvalidInput, got {other:?}"),
    })
}

#[test]
fn concat_same_mp3_errors() {
    let Some(message) = error("concat_same.mp3") else {
        return;
    };
    assert!(message.contains("consecutive"), "{message}");
}

#[test]
fn concat_channels_mp3_errors() {
    let Some(message) = error("concat_channels.mp3") else {
        return;
    };
    assert!(message.contains("consecutive"), "{message}");
}

#[test]
fn chained_ogg_errors() {
    let Some(message) = error("chained.ogg") else {
        return;
    };
    assert!(message.contains("changes mid-file"), "{message}");
}

#[test]
fn truncated_mp3_keeps_prefix() {
    let Some(decoded) = load("truncated.mp3") else {
        return;
    };
    assert!(
        decoded.samples.len() > SOURCE_FRAMES / 2,
        "{}",
        decoded.samples.len()
    );
    assert!(decoded.samples.len() < SOURCE_FRAMES);
}

#[test]
fn corrupt_mp3_skips_packet() {
    let Some(decoded) = load("corrupt.mp3") else {
        return;
    };
    assert_eq!(decoded.sample_rate.hz(), 16_000);
    let delta = decoded.samples.len().abs_diff(SOURCE_FRAMES);
    assert!(delta <= LOSSY_FRAMES_TOLERANCE, "{}", decoded.samples.len());
}

#[test]
fn zero_sample_stream_errors() {
    let Some(message) = error("zero_samples.mp3") else {
        return;
    };
    assert!(message.contains("no audio could be decoded"), "{message}");
}

#[test]
fn opus_webm_rejected() {
    let Some(message) = error("tone.webm") else {
        return;
    };
    assert!(message.contains("Opus"), "{message}");
    assert!(message.contains("transcode"), "{message}");
}

#[test]
fn duration_limit_enforced() {
    let Some(path) = fixture("speech.flac") else {
        return;
    };
    let limits = DecodeLimits::new(Duration::from_secs(2));
    match audio::read(&path, limits) {
        Err(SpeechError::InvalidInput(message)) => assert!(message.contains("limit"), "{message}"),
        other => panic!("expected the limit to apply, got {other:?}"),
    }
    let exact = DecodeLimits::new(Duration::from_secs(3));
    assert_eq!(
        audio::read(&path, exact).unwrap().samples.len(),
        SOURCE_FRAMES
    );
}

#[test]
fn error_lists_formats() {
    let error = audio::decode(b"definitely not audio", DecodeLimits::default()).unwrap_err();
    assert!(
        error.to_string().contains("supported formats: WAV"),
        "{error}"
    );
    let missing = audio::read("/nonexistent/speechkit.wav", DecodeLimits::default()).unwrap_err();
    assert!(missing.to_string().contains("speechkit.wav"), "{missing}");
}

#[test]
fn samples_clamped_to_session_range() {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut bytes = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut bytes, spec).unwrap();
    for sample in [0.5_f32, 1.5, -2.0, f32::INFINITY, f32::NAN] {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    let audio = audio::decode(bytes.get_ref(), DecodeLimits::default()).unwrap();
    assert_eq!(audio.samples, [0.5, 1.0, -1.0, 0.0, 0.0]);
}

#[test]
fn lossless_exact() {
    let Some(source_path) = fixture("source.wav") else {
        return;
    };
    let source = audio::read(source_path, DecodeLimits::default()).unwrap();
    assert_eq!(source.samples.len(), SOURCE_FRAMES);
    for name in ["source.wav", "speech.flac", "speech.mka", "speech.oga"] {
        let Some(decoded) = load(name) else { return };
        assert_eq!(decoded.sample_rate, source.sample_rate, "{name}");
        assert_eq!(decoded.samples.len(), source.samples.len(), "{name}");
        let worst = decoded
            .samples
            .iter()
            .zip(&source.samples)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 1e-6, "{name}: worst delta {worst}");
    }
    let bytes = std::fs::read(fixture("speech.flac").unwrap()).unwrap();
    let in_memory = audio::decode(&bytes, DecodeLimits::default()).unwrap();
    assert_eq!(in_memory.samples.len(), SOURCE_FRAMES);
}

#[test]
fn listed_extensions_are_those_that_decode() {
    let decoded = [
        "source.wav",
        "speech.aac",
        "speech.flac",
        "speech.m4a",
        "speech.mka",
        "speech.mp3",
        "speech.oga",
        "speech.ogg",
        "av.mkv",
        "av.mp4",
    ];
    for name in decoded {
        let extension = name.rsplit('.').next().unwrap();
        assert!(EXTENSIONS.contains(&extension), "{name}");
        if load(name).is_none() {
            return;
        }
    }
    // tone.webm holds Opus, which `opus_webm_rejected` shows fails.
    for extension in ["webm", "opus"] {
        assert!(!EXTENSIONS.contains(&extension), "{extension}");
    }
    assert!(EXTENSIONS.is_sorted());
    assert!(
        EXTENSIONS
            .iter()
            .all(|e| *e == e.to_lowercase() && !e.starts_with('.'))
    );
}

#[test]
fn stereo_downmix_native_rate() {
    let Some(decoded) = load("speech_stereo.mp3") else {
        return;
    };
    assert_eq!(decoded.sample_rate.hz(), 44_100);
    let delta = decoded.samples.len().abs_diff(3 * 44_100);
    assert!(delta <= 44_100 / 5, "{}", decoded.samples.len());
}

#[test]
fn lossy_formats_keep_duration_and_level() {
    let Some(source_path) = fixture("source.wav") else {
        return;
    };
    let level = rms(&audio::read(source_path, DecodeLimits::default())
        .unwrap()
        .samples);
    for name in [
        "speech.mp3",
        "speech.m4a",
        "speech.aac",
        "speech.ogg",
        "av.mkv",
        "av.mp4",
    ] {
        let Some(decoded) = load(name) else { return };
        assert_eq!(decoded.sample_rate.hz(), 16_000, "{name}");
        let delta = decoded.samples.len().abs_diff(SOURCE_FRAMES);
        assert!(
            delta <= LOSSY_FRAMES_TOLERANCE,
            "{name}: {}",
            decoded.samples.len()
        );
        let ratio = rms(&decoded.samples) / level;
        assert!((0.4..2.0).contains(&ratio), "{name}: level ratio {ratio}");
    }
}
