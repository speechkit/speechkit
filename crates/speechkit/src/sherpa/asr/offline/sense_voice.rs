//! SenseVoice.

use std::{
    collections::HashMap,
    fmt,
    str::FromStr,
    sync::{Arc, Mutex, PoisonError},
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

/// A language SenseVoice accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub(crate) enum SenseVoiceLanguage {
    /// Detect the language.
    #[default]
    Auto,
    /// Mandarin Chinese.
    Zh,
    /// English.
    En,
    /// Japanese.
    Ja,
    /// Korean.
    Ko,
    /// Cantonese.
    Yue,
}

impl SenseVoiceLanguage {
    /// Every language, in the order of [`SENSE_VOICE_LANGUAGES`].
    const ALL: [Self; 6] = [
        Self::Auto,
        Self::Zh,
        Self::En,
        Self::Ja,
        Self::Ko,
        Self::Yue,
    ];

    /// The code sherpa-onnx uses.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Zh => "zh",
            Self::En => "en",
            Self::Ja => "ja",
            Self::Ko => "ko",
            Self::Yue => "yue",
        }
    }
}

impl fmt::Display for SenseVoiceLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for SenseVoiceLanguage {
    type Err = SpeechError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|language| language.as_str() == value)
            .ok_or_else(|| {
                SpeechError::InvalidInput(format!(
                    "SenseVoice does not support language {value:?}; use one of {}",
                    SENSE_VOICE_LANGUAGES.join(", ")
                ))
            })
    }
}

/// Settings for [`SenseVoice`], checked by `AsrConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct SenseVoiceConfig {
    /// The model's files.
    pub files: ModelFiles,
    /// The language used when a session names none. Default: auto.
    pub language: SenseVoiceLanguage,
    /// Where and how the model runs.
    pub inference: Inference,
}

/// SenseVoice, an offline model for Chinese, English, Japanese, Korean,
/// and Cantonese that writes its own punctuation.
///
/// sherpa-onnx fixes the language when a recognizer is built, so a session
/// that overrides the language gets a recognizer built for it on first use
/// and cached for later sessions. Each costs as much memory as the first.
pub(crate) struct SenseVoice {
    config: SenseVoiceConfig,
    caps: AsrCapabilities,
    recognizers: Mutex<HashMap<SenseVoiceLanguage, Arc<NativeRecognizer>>>,
}

impl SenseVoice {
    /// Loads the model from the files `AsrConfig` checked.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for bad settings, or
    /// [`SpeechError::InvalidModel`] for a model the native library
    /// rejects.
    pub(crate) fn load(config: &SenseVoiceConfig) -> Result<Self, SpeechError> {
        config.inference.validate()?;
        let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
        caps.accepts_language = true;
        caps.punctuated = true;
        let model = Self {
            config: config.clone(),
            caps,
            recognizers: Mutex::default(),
        };
        model.recognizer(config.language)?;
        Ok(model)
    }

    fn recognizer(
        &self,
        language: SenseVoiceLanguage,
    ) -> Result<Arc<NativeRecognizer>, SpeechError> {
        let mut cache = self
            .recognizers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(recognizer) = cache.get(&language) {
            return Ok(recognizer.clone());
        }
        let files = &self.config.files;
        let AsrFiles::Flat { model, tokens } = &files.files else {
            return Err(SpeechError::InvalidModel(
                "not a flat model directory".into(),
            ));
        };
        let path = |name: &String| Some(files.path(name).to_string_lossy().into_owned());
        let mut native_config = OfflineRecognizerConfig::default();
        native_config.model_config.sense_voice.model = path(model);
        native_config.model_config.sense_voice.language = Some(language.as_str().into());
        native_config.model_config.sense_voice.use_itn = true;
        native_config.model_config.tokens = path(tokens);
        native_config.model_config.num_threads = self.config.inference.threads_i32();
        native_config.model_config.provider = Some(self.config.inference.provider.as_str().into());
        let recognizer = NativeRecognizer::create(&native_config)
            .ok_or_else(|| load_failed("SenseVoice", &files.root))?;
        let recognizer = Arc::new(recognizer);
        cache.insert(language, recognizer.clone());
        Ok(recognizer)
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
            Some(code) => code.parse()?,
            None => self.config.language,
        };
        let recognizer = self.recognizer(language)?;
        super::transcribe(&recognizer, &recognizer.create_stream(), samples)
    }
}

impl fmt::Debug for SenseVoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SenseVoice")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_parse() {
        for (language, code) in SenseVoiceLanguage::ALL.iter().zip(SENSE_VOICE_LANGUAGES) {
            assert_eq!(code.parse::<SenseVoiceLanguage>().unwrap(), *language);
            assert_eq!(language.to_string(), code);
        }
        assert!("fr".parse::<SenseVoiceLanguage>().is_err());
    }
}
