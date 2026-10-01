//! Punctuation restoration.

use std::path::{Path, PathBuf};

use crate::{SpeechError, asr::PostProcessor};
use sherpa_onnx::{
    OfflinePunctuation, OfflinePunctuationConfig, OnlinePunctuation, OnlinePunctuationConfig,
};

use crate::sherpa::{
    layout::{self, PunctFiles, PunctuationFamily},
    load_failed,
};

enum Model {
    CtTransformer(OfflinePunctuation),
    CnnBiLstm(OnlinePunctuation),
}

/// A sherpa-onnx punctuation model. [`load`](Self::load) makes a
/// [`Punctuation`].
///
/// The files decide the family, and [`validate`](Self::validate) tells
/// which it is without loading it:
///
/// - **CT-Transformer** (Chinese and English): writes full-width
///   punctuation (`，。？`), even for English text.
/// - **CNN-BiLSTM** (English, with `bpe.vocab`): writes ASCII punctuation
///   and restores capitalization. Input is lowercased first, since its
///   vocabulary is lowercase and streaming models often emit uppercase.
///
/// Punctuation always runs on the CPU with two threads.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PunctuationConfig {
    /// The model directory.
    pub model: PathBuf,
}

impl PunctuationConfig {
    /// The model in `model`.
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
        }
    }

    /// Checks the directory without loading anything native, and returns
    /// the family [`load`](Self::load) would load from it.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a missing, empty, or ambiguous
    /// file.
    pub fn validate(&self) -> Result<PunctuationFamily, SpeechError> {
        Ok(files(&self.model)?.layout)
    }

    /// Loads the model, detecting its family first.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a bad directory (see
    /// [`validate`](Self::validate)) or a model the native library
    /// rejects.
    pub fn load(&self) -> Result<Punctuation, SpeechError> {
        let dir = self.model.as_path();
        let files = files(dir)?;
        let model_path = dir.join(&files.model).to_string_lossy().into_owned();
        let failed = || load_failed(format_args!("{} punctuation", files.layout), dir);
        let model = match &files.vocab {
            None => {
                let mut config = OfflinePunctuationConfig::default();
                config.model.ct_transformer = Some(model_path);
                config.model.num_threads = 2;
                config.model.provider = Some("cpu".into());
                Model::CtTransformer(OfflinePunctuation::create(&config).ok_or_else(failed)?)
            }
            Some(vocab) => {
                let mut config = OnlinePunctuationConfig::default();
                config.model.cnn_bilstm = Some(model_path);
                config.model.bpe_vocab = Some(dir.join(vocab).to_string_lossy().into_owned());
                config.model.num_threads = 2;
                config.model.provider = Some("cpu".into());
                Model::CnnBiLstm(OnlinePunctuation::create(&config).ok_or_else(failed)?)
            }
        };
        Ok(Punctuation {
            model,
            layout: files.layout,
        })
    }
}

/// Adds punctuation to committed segments; pass it to
/// `AsrEngine::with_post_processor`. [`PunctuationConfig::load`] makes one.
pub struct Punctuation {
    model: Model,
    layout: PunctuationFamily,
}

/// Lists `dir` and picks the punctuation model's files.
fn files(dir: &Path) -> Result<PunctFiles, SpeechError> {
    layout::select_punct(&layout::list_dir(dir)?)
}

impl PostProcessor for Punctuation {
    fn process(&self, text: &str) -> Result<String, SpeechError> {
        let result = match &self.model {
            Model::CtTransformer(model) => model.add_punctuation(text),
            Model::CnnBiLstm(model) => model.add_punctuation(&text.to_lowercase()),
        };
        result
            .ok_or_else(|| SpeechError::backend("sherpa-punctuation", false, "punctuation failed"))
    }
}

impl std::fmt::Debug for Punctuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Punctuation")
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            std::fs::write(dir.path().join(name), content).unwrap();
        }
        dir
    }

    #[test]
    fn validate_tells_the_model_without_loading() {
        let ct = dir_with(&[("model.int8.onnx", "x")]);
        let cnn = dir_with(&[("model.onnx", "x"), ("bpe.vocab", "x")]);
        assert_eq!(
            PunctuationConfig::new(ct.path()).validate().unwrap(),
            PunctuationFamily::CtTransformer
        );
        assert_eq!(
            PunctuationConfig::new(cnn.path()).validate().unwrap(),
            PunctuationFamily::CnnBiLstm
        );
        assert_eq!(
            PunctuationFamily::CtTransformer.to_string(),
            "CT-Transformer"
        );
        assert_eq!(PunctuationFamily::CnnBiLstm.to_string(), "CNN-BiLSTM");
        for bad in [dir_with(&[]), dir_with(&[("model.onnx", "")])] {
            let error = PunctuationConfig::new(bad.path()).validate().unwrap_err();
            assert!(matches!(error, SpeechError::InvalidModel(_)), "{error}");
        }
    }
}
