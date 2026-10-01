//! The real worker binary reports configuration errors through the
//! protocol, before loading anything native.

use speechkit::SpeechError;
use speechkit::sherpa::{
    AsrConfig,
    process::{IsolatedAsr, WorkerCommand},
};

/// `IsolatedAsr::worker` checks the directory before the worker starts.
#[test]
fn bad_model_directory_is_reported() {
    let config = AsrConfig::streaming("/nonexistent/model");
    let error =
        IsolatedAsr::spawn(env!("CARGO_BIN_EXE_speechkit-worker"), &config).expect_err("no model");
    assert!(matches!(error, SpeechError::InvalidModel(_)), "{error}");
}

#[test]
fn missing_config_is_reported() {
    let error =
        IsolatedAsr::spawn_command(WorkerCommand::new(env!("CARGO_BIN_EXE_speechkit-worker")))
            .expect_err("no config");
    assert!(matches!(error, SpeechError::InvalidInput(_)), "{error}");
}

/// A real model transcribes through the worker, with settings that cross
/// the wire: a found family, VAD settings, and a language.
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn a_real_model_transcribes_through_the_worker() {
    use std::time::Duration;

    use speechkit::{
        asr::{AsrEngine, AsrOptions},
        audio::{self, DecodeLimits},
        sherpa::SileroVadConfig,
    };
    use speechkit_testkit::{gates::model_dir, secs};

    let (Some(model), Some(vad)) = (model_dir("sense-voice"), model_dir("silero-vad")) else {
        return;
    };
    let config = AsrConfig::offline(&model, &vad)
        .with_vad(SileroVadConfig::new(&vad).with_min_silence(Duration::from_millis(300)))
        .with_language("en");
    let backend = IsolatedAsr::spawn(env!("CARGO_BIN_EXE_speechkit-worker"), &config)
        .expect("the worker loads the model");
    let audio =
        audio::read(model.join("test_wavs/en.wav"), DecodeLimits::default()).expect("the sample");
    let transcript = AsrEngine::new(backend)
        .transcribe(&audio, AsrOptions::default(), secs(120))
        .expect("a transcript");
    let text = transcript.text();
    assert!(text.contains("tribal chieftain"), "{text}");
}
