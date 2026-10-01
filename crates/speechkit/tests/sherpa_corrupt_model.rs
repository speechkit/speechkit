//! What happens when a model directory has the right file names but
//! garbage content.
//!
//! Each case runs in a child process, because the native library may abort
//! the whole process instead of returning an error. The parent records the
//! outcome, so the behavior stays documented and a change is noticed.
#![cfg(feature = "sherpa")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test helpers fail the calling test"
)]

use std::{path::Path, process::Command};

use speechkit::sherpa::{AsrConfig, AsrFamily, PunctuationConfig, SileroVadConfig};

const CHILD: &str = "SPEECHKIT_CORRUPT_MODEL_CASE";

fn garbage(dir: &Path, names: &[&str]) {
    for (i, name) in names.iter().enumerate() {
        // A fixed pseudo-random byte pattern (xorshift), different per file.
        let mut state = 0x9E37_79B9_u32 ^ u32::try_from(i).unwrap_or(0);
        let bytes: Vec<u8> = (0..4096)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[0]
            })
            .collect();
        std::fs::write(dir.join(name), bytes).unwrap();
    }
}

/// Runs one case in this process and reports how it ended.
fn run_case(case: &str) {
    let dir = tempfile::tempdir().unwrap();
    let outcome = match case {
        "streaming" => {
            garbage(dir.path(), &["encoder.onnx", "decoder.onnx", "joiner.onnx"]);
            std::fs::write(dir.path().join("tokens.txt"), "a 0\nb 1\n").unwrap();
            AsrConfig::streaming(dir.path()).load().map(drop)
        }
        "sense-voice" => {
            garbage(dir.path(), &["model.onnx"]);
            std::fs::write(dir.path().join("tokens.txt"), "a 0\n").unwrap();
            // The recognizer loads first; the VAD is garbage too, in case.
            let vad = tempfile::tempdir().unwrap();
            garbage(vad.path(), &["silero_vad.onnx"]);
            let config = AsrConfig::offline(dir.path(), vad.path().join("silero_vad.onnx"))
                .with_family(AsrFamily::SenseVoice);
            config.load().map(drop)
        }
        "vad" => {
            garbage(dir.path(), &["silero_vad.onnx"]);
            SileroVadConfig::new(dir.path().join("silero_vad.onnx"))
                .load()
                .map(drop)
        }
        "punctuation" => {
            garbage(dir.path(), &["model.onnx"]);
            PunctuationConfig::new(dir.path()).load().map(drop)
        }
        other => panic!("unknown case {other}"),
    };
    match outcome {
        Ok(()) => println!("OUTCOME: loaded"),
        Err(error) => println!("OUTCOME: error: {error}"),
    }
}

/// The child entry point. It does nothing unless the parent set the case.
#[test]
fn child() {
    if let Ok(case) = std::env::var(CHILD) {
        run_case(&case);
    }
}

fn outcome(case: &str) -> String {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child", "--nocapture", "--test-threads=1"])
        .env(CHILD, case)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    match stdout
        .lines()
        .find_map(|line| line.strip_prefix("OUTCOME: "))
    {
        Some(outcome) => outcome.to_owned(),
        None => format!("process ended without a result ({})", output.status),
    }
}

#[test]
fn corrupt_models_never_load() {
    for case in ["streaming", "sense-voice", "vad", "punctuation"] {
        let outcome = outcome(case);
        eprintln!("{case}: {outcome}");
        assert_ne!(outcome, "loaded", "{case}");
    }
}
