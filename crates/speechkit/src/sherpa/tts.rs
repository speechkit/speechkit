//! Speech synthesis with sherpa-onnx: VITS and Piper,
//! Matcha, and Kokoro models.

use std::{path::PathBuf, sync::Arc};

use crate::{
    Flow, SampleRate, SpeechError,
    tts::{TtsBackend, TtsCapabilities, TtsOptions, TtsStream, Voice},
};
use sherpa_onnx::{GenerationConfig, OfflineTts, OfflineTtsConfig};

use crate::sherpa::{
    config::Inference,
    layout::{self, TtsFamily, TtsFiles},
    load_failed,
};

/// The speed range offered to sessions.
pub(crate) const SPEED_RANGE: std::ops::RangeInclusive<f32> = 0.5..=2.0;

/// The default longest chunk passed to one native call, in characters.
pub(crate) const DEFAULT_MAX_INPUT_CHARS: usize = 500;

/// A sherpa-onnx synthesis model and how to run it. [`load`](Self::load)
/// makes a [`Tts`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TtsConfig {
    /// The model directory.
    pub model: PathBuf,
    /// Where and how the model runs.
    pub inference: Inference,
    /// The longest chunk passed to one native call, in characters.
    /// Default: 500.
    pub max_input_chars: usize,
}

impl TtsConfig {
    /// Settings for the model in `model`.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            inference: Inference::default(),
            max_input_chars: DEFAULT_MAX_INPUT_CHARS,
        }
    }

    /// Sets the inference settings.
    #[must_use]
    pub fn with_inference(mut self, inference: Inference) -> Self {
        self.inference = inference;
        self
    }

    /// Sets the longest chunk passed to one native call.
    #[must_use]
    pub fn with_max_input_chars(mut self, chars: usize) -> Self {
        self.max_input_chars = chars;
        self
    }

    /// Checks the directory and the settings for [`load`](Self::load)
    /// without loading anything native, and returns the family it will
    /// load.
    ///
    /// # Errors
    ///
    /// As [`load`](Self::load), except for a model the native library
    /// rejects.
    pub fn validate(&self) -> Result<TtsFamily, SpeechError> {
        Ok(self.check()?.layout)
    }

    /// Validates, and returns the model's files.
    fn check(&self) -> Result<TtsFiles, SpeechError> {
        self.inference.validate()?;
        if self.max_input_chars == 0 {
            return Err(SpeechError::InvalidInput(
                "max_input_chars must be positive".into(),
            ));
        }
        layout::select_tts(&layout::list_dir(&self.model)?)
    }
}

/// A sherpa-onnx synthesis model.
///
/// Voices are the model's speaker IDs, `"0"` to `"n-1"`; the default is
/// `"0"`. sherpa-onnx returns a chunk's audio only once the chunk is done,
/// so audio arrives one chunk at a time (`streams_audio` is false), and
/// a cancelled session stops after the chunk being synthesized.
/// [`TtsConfig::load`] makes one.
pub struct Tts {
    native: Arc<OfflineTts>,
    layout: TtsFamily,
    caps: TtsCapabilities,
    voices: Vec<Voice>,
}

impl std::fmt::Debug for Tts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tts")
            .field("layout", &self.layout)
            .field("caps", &self.caps)
            .field("voices", &self.voices.len())
            .finish_non_exhaustive()
    }
}

fn native_config(config: &TtsConfig, files: &TtsFiles) -> OfflineTtsConfig {
    let path = |name: &String| Some(config.model.join(name).to_string_lossy().into_owned());
    let joined = |names: &[String]| {
        (!names.is_empty()).then(|| {
            names
                .iter()
                .map(|name| config.model.join(name).to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(",")
        })
    };
    let mut native = OfflineTtsConfig::default();
    let model = &mut native.model;
    let lexicon = joined(&files.lexicons);
    let data_dir = files.espeak_data.as_ref().and_then(path);
    let dict_dir = files.dict.as_ref().and_then(path);
    let tokens = path(&files.tokens);
    match files.layout {
        TtsFamily::Matcha => {
            model.matcha.acoustic_model = path(&files.model);
            model.matcha.vocoder = files.vocoder.as_ref().and_then(path);
            model.matcha.lexicon = lexicon;
            model.matcha.tokens = tokens;
            model.matcha.data_dir = data_dir;
            model.matcha.dict_dir = dict_dir;
        }
        TtsFamily::Kokoro => {
            model.kokoro.model = path(&files.model);
            model.kokoro.voices = files.voices.as_ref().and_then(path);
            model.kokoro.lexicon = lexicon;
            model.kokoro.tokens = tokens;
            model.kokoro.data_dir = data_dir;
            model.kokoro.dict_dir = dict_dir;
        }
        // VITS, and any layout added later that shares its files.
        _ => {
            model.vits.model = path(&files.model);
            model.vits.lexicon = lexicon;
            model.vits.tokens = tokens;
            model.vits.data_dir = data_dir;
            model.vits.dict_dir = dict_dir;
        }
    }
    model.num_threads = config.inference.threads_i32();
    model.provider = Some(config.inference.provider.as_str().into());
    native.rule_fsts = joined(&files.rule_fsts);
    native.rule_fars = joined(&files.rule_fars);
    native.max_num_sentences = 1;
    native.silence_scale = 0.2;
    native
}

impl TtsConfig {
    /// Loads the model. The directory and the settings are checked first,
    /// as [`validate`](Self::validate) does.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] or [`SpeechError::Unsupported`] for bad
    /// settings, [`SpeechError::InvalidModel`] for a bad directory or a
    /// model the native library rejects.
    pub fn load(&self) -> Result<Tts, SpeechError> {
        let config = self;
        let files = config.check()?;
        let tts = OfflineTts::create(&native_config(config, &files))
            .ok_or_else(|| load_failed(files.layout, &config.model))?;
        let rate = u32::try_from(tts.sample_rate())
            .map_err(|_| SpeechError::InvalidModel("negative sample rate".into()))
            .and_then(SampleRate::new)?;
        let speakers = tts.num_speakers().max(1);
        let voices = (0..speakers)
            .map(|sid| Voice::new(sid.to_string()))
            .collect();
        let mut caps = TtsCapabilities::new(rate, config.max_input_chars);
        caps.speed = Some(SPEED_RANGE);
        tracing::debug!(layout = %files.layout, rate = rate.hz(), speakers, "loaded sherpa-onnx TTS");
        Ok(Tts {
            native: Arc::new(tts),
            layout: files.layout,
            caps,
            voices,
        })
    }
}

impl TtsBackend for Tts {
    fn name(&self) -> &'static str {
        "sherpa-onnx"
    }

    fn capabilities(&self) -> &TtsCapabilities {
        &self.caps
    }

    fn voices(&self) -> &[Voice] {
        &self.voices
    }

    fn open(&self, opts: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError> {
        let sid = match &opts.voice {
            Some(id) => id
                .parse()
                .map_err(|_| SpeechError::InvalidInput(format!("unknown voice {id:?}")))?,
            None => 0,
        };
        Ok(Box::new(Stream {
            tts: self.native.clone(),
            generation: GenerationConfig {
                speed: opts.speed,
                sid,
                ..GenerationConfig::default()
            },
            cancelled: false,
        }))
    }
}

struct Stream {
    tts: Arc<OfflineTts>,
    generation: GenerationConfig,
    cancelled: bool,
}

type NoProgress = fn(&[f32], f32) -> bool;

impl TtsStream for Stream {
    fn synthesize(
        &mut self,
        chunk: &str,
        sink: &mut dyn FnMut(&[f32]) -> Flow,
    ) -> Result<(), SpeechError> {
        if self.cancelled || chunk.trim().is_empty() {
            return Ok(());
        }
        if chunk.contains('\0') {
            return Err(SpeechError::InvalidInput(
                "text contains a NUL character".into(),
            ));
        }
        let audio = self
            .tts
            .generate_with_config::<NoProgress>(chunk, &self.generation, None)
            .ok_or_else(|| SpeechError::backend("sherpa-onnx", false, "synthesis failed"))?;
        let samples = audio.samples();
        if !samples.is_empty() {
            sink(samples);
        }
        Ok(())
    }

    fn cancel(&mut self) {
        self.cancelled = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error of `load`, which `validate` gives too.
    fn load_error(config: &TtsConfig) -> SpeechError {
        let validated = config.validate().unwrap_err();
        match config.load() {
            Err(error) => {
                assert_eq!(error.to_string(), validated.to_string());
                error
            }
            Ok(tts) => panic!("expected an error, loaded {tts:?}"),
        }
    }

    #[test]
    fn validate_tells_the_model_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "model.onnx",
            "voices.bin",
            "tokens.txt",
            "espeak-ng-data/phontab",
        ] {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "x").unwrap();
        }
        let config = TtsConfig::new(dir.path());
        assert_eq!(config.validate().unwrap(), TtsFamily::Kokoro);
        assert_eq!(TtsFamily::Kokoro.to_string(), "Kokoro");
        std::fs::remove_file(dir.path().join("voices.bin")).unwrap();
        assert_eq!(config.validate().unwrap(), TtsFamily::Vits);
    }

    /// Every check here fails before the native library is called.
    #[test]
    fn bad_settings_fail_before_native_load() {
        let dir = tempfile::tempdir().unwrap();
        let missing = TtsConfig::new(dir.path().join("missing"));
        assert!(matches!(load_error(&missing), SpeechError::InvalidModel(_)));

        std::fs::write(dir.path().join("tokens.txt"), "a 0\n").unwrap();
        let no_model = TtsConfig::new(dir.path());
        assert!(matches!(
            load_error(&no_model),
            SpeechError::InvalidModel(_)
        ));

        std::fs::write(dir.path().join("model.onnx"), "").unwrap();
        std::fs::write(dir.path().join("lexicon.txt"), "a a\n").unwrap();
        let SpeechError::InvalidModel(message) = load_error(&no_model) else {
            panic!("expected InvalidModel");
        };
        assert!(message.contains("empty"), "{message}");

        let zero = TtsConfig::new(dir.path()).with_max_input_chars(0);
        assert!(matches!(load_error(&zero), SpeechError::InvalidInput(_)));

        let threads =
            TtsConfig::new(dir.path()).with_inference(Inference::default().with_threads(0));
        assert!(matches!(load_error(&threads), SpeechError::InvalidInput(_)));
    }
}
