//! Describing a model directory for a model picker.

use std::path::Path;

use crate::{
    SpeechError,
    sherpa::{AsrFamily, PunctuationFamily, TtsFamily, kws, layout},
};

/// What a model directory holds, as [`inspect`] finds it from the files.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct ModelInfo {
    /// Every recognition family the files fit, in the order of
    /// [`AsrFamily::ALL`]. Some families share files, so more than one can
    /// fit: a transducer is a streaming or an offline one, and a single
    /// model with `tokens.txt` is SenseVoice, Paraformer, or FireRedASR
    /// CTC unless its tokens hold SenseVoice's language markers. Name the
    /// one to load with [`AsrConfig::with_family`](crate::sherpa::AsrConfig::with_family).
    pub asr: Vec<AsrFamily>,
    /// The synthesis family, for a synthesis model.
    pub tts: Option<TtsFamily>,
    /// The punctuation family, for a punctuation model.
    pub punctuation: Option<PunctuationFamily>,
    /// Whether it is a keyword spotting model: a transducer shipped with a
    /// `keywords.txt`, as the published `kws-zipformer` models are.
    pub keyword_spotter: bool,
}

/// Describes the model in `dir` from its files, without loading anything
/// native, for a model picker.
///
/// A directory holds one model, so one kind is reported: a keyword
/// spotter, a synthesis model, a recognition model, or a punctuation
/// model, checked in that order, since the later kinds' files are a subset
/// of the earlier ones'.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] if `dir` cannot be read, holds an empty
/// model file, or fits no kind of model; the error then says what a
/// recognition model would be missing.
pub fn inspect(dir: impl AsRef<Path>) -> Result<ModelInfo, SpeechError> {
    let dir = dir.as_ref();
    let listing = layout::list_dir(dir)?;
    if is_keyword_spotter(dir, &listing) {
        return Ok(ModelInfo {
            keyword_spotter: true,
            ..ModelInfo::default()
        });
    }
    if let Ok(files) = layout::select_tts(&listing) {
        return Ok(ModelInfo {
            tts: Some(files.layout),
            ..ModelInfo::default()
        });
    }
    let asr = match AsrFamily::detect(dir, &listing) {
        Ok(asr) => asr,
        Err(error) => {
            let has_tokens = listing.iter().any(|name| file_name(name) == "tokens.txt");
            return match layout::select_punct(&listing) {
                Ok(files) if !has_tokens => Ok(ModelInfo {
                    punctuation: Some(files.layout),
                    ..ModelInfo::default()
                }),
                _ => Err(error),
            };
        }
    };
    Ok(ModelInfo {
        asr,
        ..ModelInfo::default()
    })
}

fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn is_keyword_spotter(dir: &Path, listing: &[String]) -> bool {
    listing
        .iter()
        .any(|name| matches!(file_name(name), "keywords.txt" | "keywords_raw.txt"))
        && kws::resolve(dir).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let content = if *name == "tokens.txt" { "a 1\n" } else { "x" };
            std::fs::write(path, content).unwrap();
        }
        dir
    }

    #[test]
    fn each_kind_is_told_apart() {
        let transducer = ["encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt"];
        let found = inspect(dir_with(&transducer).path()).unwrap();
        assert_eq!(
            found.asr,
            [AsrFamily::StreamingTransducer, AsrFamily::OfflineTransducer]
        );
        assert!(!found.keyword_spotter);

        let mut kws = transducer.to_vec();
        kws.push("test_wavs/keywords.txt");
        let found = inspect(dir_with(&kws).path()).unwrap();
        assert!(found.keyword_spotter);
        assert!(found.asr.is_empty());

        let vits = ["model.onnx", "tokens.txt", "lexicon.txt"];
        let found = inspect(dir_with(&vits).path()).unwrap();
        assert_eq!(found.tts, Some(TtsFamily::Vits));
        assert!(found.asr.is_empty());

        let punctuation = inspect(dir_with(&["model.onnx", "config.yaml"]).path()).unwrap();
        assert_eq!(
            punctuation.punctuation,
            Some(PunctuationFamily::CtTransformer)
        );
        assert!(punctuation.asr.is_empty());
    }

    #[test]
    fn nothing_found_says_what_is_missing() {
        let error = inspect(dir_with(&["README.md"]).path()).unwrap_err();
        assert!(matches!(error, SpeechError::InvalidModel(_)), "{error}");
        assert!(inspect("/nonexistent/speechkit").is_err());
    }
}
