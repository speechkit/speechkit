//! SenseVoice.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use crate::{
    SampleRate, SpeechError,
    asr::{AsrCapabilities, AsrOptions},
    vad::OfflineRecognizer,
};
use sherpa_onnx::{OfflineRecognizer as NativeRecognizer, OfflineRecognizerConfig};

use crate::sherpa::{
    config::{Inference, ModelFiles, SENSE_VOICE_LANGUAGES},
    layout::AsrFiles,
    load_failed,
};

/// The code of `value`, one of [`SENSE_VOICE_LANGUAGES`].
fn language(value: &str) -> Result<&'static str, SpeechError> {
    SENSE_VOICE_LANGUAGES
        .into_iter()
        .find(|code| *code == value)
        .ok_or_else(|| {
            SpeechError::InvalidInput(format!(
                "SenseVoice does not support language {value:?}; use one of {}",
                SENSE_VOICE_LANGUAGES.join(", ")
            ))
        })
}

/// Settings for [`SenseVoice`], checked by `AsrConfig`.
#[derive(Debug)]
pub(crate) struct SenseVoiceConfig<'a> {
    /// The model's files.
    pub files: ModelFiles,
    /// The language used when a session names none, such as `auto`.
    pub language: &'a str,
    /// Where and how the model runs.
    pub inference: Inference,
}

/// A recognizer for one language, built on first use.
type Slot = Arc<Mutex<Option<Arc<NativeRecognizer>>>>;

/// SenseVoice, an offline model for Chinese, English, Japanese, Korean,
/// and Cantonese that writes its own punctuation.
///
/// sherpa-onnx fixes the language when a recognizer is built, so a session
/// that overrides the language gets a recognizer built for it on first use
/// and cached for later sessions. Each costs as much memory as the first.
/// Building one holds only that language's slot, so sessions in other
/// languages keep recognizing meanwhile.
pub(crate) struct SenseVoice {
    files: ModelFiles,
    inference: Inference,
    /// The language used when a session names none.
    language: &'static str,
    caps: AsrCapabilities,
    recognizers: Mutex<HashMap<&'static str, Slot>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SenseVoice {
    /// Loads the model from the files `AsrConfig` checked.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a language SenseVoice lacks, or
    /// [`SpeechError::InvalidModel`] for a model the native library
    /// rejects.
    pub(crate) fn load(config: SenseVoiceConfig<'_>) -> Result<Self, SpeechError> {
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.accepts_language = true;
        caps.punctuated = true;
        let model = Self {
            files: config.files,
            inference: config.inference,
            language: language(config.language)?,
            caps,
            recognizers: Mutex::default(),
        };
        model.recognizer(model.language)?;
        Ok(model)
    }

    /// The recognizer for `language`, built on first use.
    fn recognizer(&self, language: &'static str) -> Result<Arc<NativeRecognizer>, SpeechError> {
        let slot = lock(&self.recognizers).entry(language).or_default().clone();
        let mut slot = lock(&slot);
        if let Some(recognizer) = &*slot {
            return Ok(recognizer.clone());
        }
        let recognizer = Arc::new(self.build(language)?);
        *slot = Some(recognizer.clone());
        Ok(recognizer)
    }

    fn build(&self, language: &'static str) -> Result<NativeRecognizer, SpeechError> {
        let files = &self.files;
        let AsrFiles::Flat { model, tokens } = &files.files else {
            return Err(SpeechError::InvalidModel(
                "not a flat model directory".into(),
            ));
        };
        let path = |name: &String| Some(files.path(name).to_string_lossy().into_owned());
        let mut native_config = OfflineRecognizerConfig::default();
        native_config.model_config.sense_voice.model = path(model);
        native_config.model_config.sense_voice.language = Some(language.into());
        native_config.model_config.sense_voice.use_itn = true;
        native_config.model_config.tokens = path(tokens);
        native_config.model_config.num_threads = self.inference.threads_i32();
        native_config.model_config.provider = Some(self.inference.provider.as_str().into());
        NativeRecognizer::create(&native_config)
            .ok_or_else(|| load_failed("SenseVoice", &files.root))
    }
}

impl OfflineRecognizer for SenseVoice {
    fn name(&self) -> &'static str {
        "sherpa-sense-voice"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn recognize(&self, samples: &[f32], opts: &AsrOptions) -> Result<String, SpeechError> {
        let language = match &opts.language {
            Some(code) => language(code)?,
            None => self.language,
        };
        let recognizer = self.recognizer(language)?;
        super::transcribe(&recognizer, &recognizer.create_stream(), samples)
    }

    fn check(&self, opts: &AsrOptions) -> Result<(), SpeechError> {
        match &opts.language {
            Some(code) => language(code).map(drop),
            None => Ok(()),
        }
    }
}

impl fmt::Debug for SenseVoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SenseVoice")
            .field("files", &self.files)
            .field("language", &self.language)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_parse() {
        for code in SENSE_VOICE_LANGUAGES {
            assert_eq!(language(code).unwrap(), code);
        }
        assert!(language("fr").is_err());
    }
}
