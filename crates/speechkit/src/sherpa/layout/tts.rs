//! Speech synthesis model layouts.

use std::fmt;

use crate::SpeechError;

use super::{Listing, invalid, is_onnx, prefer};

/// A speech synthesis model family, which the files of its directory
/// tell. [`TtsConfig::validate`](crate::sherpa::TtsConfig::validate)
/// returns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TtsFamily {
    /// VITS, including Piper voices: one `.onnx` model, `tokens.txt`, and
    /// a `lexicon.txt` or an `espeak-ng-data/` directory.
    Vits,
    /// Matcha: an acoustic model (`model-steps-*.onnx`), a vocoder
    /// (`vocos*.onnx` or `hifigan*.onnx`), `tokens.txt`, and a lexicon or
    /// `espeak-ng-data/`.
    Matcha,
    /// Kokoro: `model*.onnx`, `voices.bin`, `tokens.txt`, and
    /// `espeak-ng-data/`.
    Kokoro,
}

impl fmt::Display for TtsFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Vits => "VITS",
            Self::Matcha => "Matcha",
            Self::Kokoro => "Kokoro",
        })
    }
}

/// The files of a synthesis model, relative to the listed directory.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct TtsFiles {
    /// The family.
    pub layout: TtsFamily,
    /// The model; for Matcha, the acoustic model.
    pub model: String,
    /// The Matcha vocoder.
    pub vocoder: Option<String>,
    /// `tokens.txt`.
    pub tokens: String,
    /// Lexicon files (`lexicon*.txt`), sorted.
    pub lexicons: Vec<String>,
    /// `espeak-ng-data/`, if present.
    pub espeak_data: Option<String>,
    /// `dict/`, the jieba dictionary some Chinese models use.
    pub dict: Option<String>,
    /// `voices.bin`, for Kokoro.
    pub voices: Option<String>,
    /// Text normalization rules (`*.fst`), sorted.
    pub rule_fsts: Vec<String>,
    /// Text normalization archives (`*.far`), sorted.
    pub rule_fars: Vec<String>,
}

const VOCODERS: [&str; 2] = ["vocos", "hifigan"];

fn has_extension(name: &str, extension: &str) -> bool {
    std::path::Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
}

fn is_vocoder(name: &str) -> bool {
    VOCODERS.iter().any(|prefix| name.starts_with(prefix))
}

impl Listing<'_> {
    /// Top-level files with `extension`, sorted, as full paths.
    fn with_extension(&self, extension: &str) -> Vec<String> {
        let mut names: Vec<&str> = self
            .files
            .iter()
            .copied()
            .filter(|f| !f.contains('/'))
            .filter(|f| has_extension(f, extension))
            .collect();
        names.sort_unstable();
        names.into_iter().map(|name| self.full(name)).collect()
    }

    fn has_dir(&self, dir: &str) -> bool {
        let prefix = format!("{dir}/");
        self.files.iter().any(|f| f.starts_with(&prefix))
    }

    fn dir(&self, dir: &str) -> Option<String> {
        self.has_dir(dir).then(|| self.full(dir))
    }

    /// Top-level `.onnx` files that are not vocoders.
    fn acoustic_models(&self) -> Vec<&str> {
        self.onnx("")
            .into_iter()
            .filter(|name| !is_vocoder(name))
            .collect()
    }
}

/// Picks the files of a synthesis model from a file list, as returned by
/// [`list_dir`](super::list_dir).
///
/// `voices.bin` means Kokoro, a vocoder or a `model-steps-*.onnx` means
/// Matcha, and anything else with one model and `tokens.txt` is VITS.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] naming what is missing or ambiguous.
pub(crate) fn select_tts(files: &[impl AsRef<str>]) -> Result<TtsFiles, SpeechError> {
    let l = Listing::new(files);
    let layout = if l.has("voices.bin") {
        TtsFamily::Kokoro
    } else if VOCODERS.iter().any(|prefix| l.has_onnx(prefix)) || l.has_onnx("model-steps") {
        TtsFamily::Matcha
    } else if l.onnx("").is_empty() {
        return Err(invalid(
            "unrecognized model directory layout; expected a VITS, Piper, Matcha, or \
             Kokoro model",
        ));
    } else {
        TtsFamily::Vits
    };
    let tokens = l.require("tokens.txt")?;
    let espeak_data = l.dir("espeak-ng-data");
    let lexicons: Vec<String> = l
        .with_extension("txt")
        .into_iter()
        .filter(|name| name[l.prefix.len()..].starts_with("lexicon"))
        .collect();
    let model = match layout {
        TtsFamily::Kokoro => l.pick("model", true)?,
        TtsFamily::Matcha => match l.onnx("model-steps")[..] {
            [] => one_acoustic_model(&l)?,
            _ => l.pick("model-steps", false)?,
        },
        TtsFamily::Vits => one_acoustic_model(&l)?,
    };
    let vocoder = match layout {
        TtsFamily::Matcha => Some(one_vocoder(&l)?),
        _ => None,
    };
    if layout == TtsFamily::Kokoro && espeak_data.is_none() {
        return Err(invalid(
            "the Kokoro model directory is missing espeak-ng-data/",
        ));
    }
    if espeak_data.is_none() && lexicons.is_empty() {
        return Err(invalid(
            "the model directory needs lexicon.txt or an espeak-ng-data/ directory",
        ));
    }
    Ok(TtsFiles {
        layout,
        model,
        vocoder,
        tokens,
        lexicons,
        espeak_data,
        dict: l.dir("dict"),
        voices: l.optional("voices.bin"),
        rule_fsts: l.with_extension("fst"),
        rule_fars: l.with_extension("far"),
    })
}

/// The one non-vocoder model, preferring int8 files if both exist.
fn one_acoustic_model(l: &Listing<'_>) -> Result<String, SpeechError> {
    l.only(
        prefer(l.acoustic_models(), |m| m.contains(".int8.")),
        "the model directory is missing the model .onnx file",
        "several model files",
    )
}

fn one_vocoder(l: &Listing<'_>) -> Result<String, SpeechError> {
    l.only(
        l.onnx("").into_iter().filter(|m| is_vocoder(m)).collect(),
        "the Matcha model directory is missing a vocoder (vocos*.onnx or hifigan*.onnx); \
         download one and put it next to the model",
        "several vocoders",
    )
}

/// Whether `name` is one of the files [`list_dir`](super::list_dir)
/// refuses to see empty.
pub(super) fn is_tts_model_file(name: &str) -> bool {
    is_onnx(name)
        || name == "voices.bin"
        || (name.starts_with("lexicon") && has_extension(name, "txt"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PIPER: &[&str] = &[
        "en_US-amy-low.onnx",
        "en_US-amy-low.onnx.json",
        "tokens.txt",
        "espeak-ng-data/phontab",
        "espeak-ng-data/voices/!v/Alex",
        "MODEL_CARD",
    ];

    const VITS_ZH: &[&str] = &[
        "vits-aishell3.onnx",
        "vits-aishell3.int8.onnx",
        "tokens.txt",
        "lexicon.txt",
        "date.fst",
        "number.fst",
        "phone.fst",
        "rule.far",
    ];

    const MATCHA: &[&str] = &[
        "model-steps-3.onnx",
        "vocos-22khz-univ.onnx",
        "tokens.txt",
        "lexicon.txt",
        "dict/jieba.dict.utf8",
        "number.fst",
    ];

    const KOKORO: &[&str] = &[
        "model.onnx",
        "voices.bin",
        "tokens.txt",
        "espeak-ng-data/phontab",
        "lexicon-us-en.txt",
        "lexicon-zh.txt",
        "dict/jieba.dict.utf8",
        "date-zh.fst",
    ];

    fn wrapped(files: &[&str]) -> Vec<String> {
        files.iter().map(|f| format!("archive/{f}")).collect()
    }

    fn message(result: Result<TtsFiles, SpeechError>) -> String {
        match result {
            Err(SpeechError::InvalidModel(message)) => message,
            other => panic!("expected InvalidModel, got {other:?}"),
        }
    }

    #[test]
    fn detection_table() {
        for (files, layout) in [
            (PIPER, TtsFamily::Vits),
            (VITS_ZH, TtsFamily::Vits),
            (MATCHA, TtsFamily::Matcha),
            (KOKORO, TtsFamily::Kokoro),
        ] {
            assert_eq!(select_tts(files).unwrap().layout, layout, "{files:?}");
            let wrapped = wrapped(files);
            assert_eq!(
                select_tts(&wrapped).unwrap().layout,
                layout,
                "wrapped {files:?}"
            );
        }
    }

    #[test]
    fn picks_files() {
        let piper = select_tts(PIPER).unwrap();
        assert_eq!(piper.model, "en_US-amy-low.onnx");
        assert_eq!(piper.espeak_data.as_deref(), Some("espeak-ng-data"));
        assert!(piper.lexicons.is_empty());

        let vits = select_tts(VITS_ZH).unwrap();
        assert_eq!(vits.model, "vits-aishell3.int8.onnx");
        assert_eq!(vits.lexicons, ["lexicon.txt"]);
        assert_eq!(vits.rule_fsts, ["date.fst", "number.fst", "phone.fst"]);
        assert_eq!(vits.rule_fars, ["rule.far"]);

        let matcha = select_tts(MATCHA).unwrap();
        assert_eq!(matcha.model, "model-steps-3.onnx");
        assert_eq!(matcha.vocoder.as_deref(), Some("vocos-22khz-univ.onnx"));
        assert_eq!(matcha.dict.as_deref(), Some("dict"));

        let wrapped = wrapped(KOKORO);
        let kokoro = select_tts(&wrapped).unwrap();
        assert_eq!(kokoro.model, "archive/model.onnx");
        assert_eq!(kokoro.voices.as_deref(), Some("archive/voices.bin"));
        assert_eq!(
            kokoro.lexicons,
            ["archive/lexicon-us-en.txt", "archive/lexicon-zh.txt"]
        );
    }

    #[test]
    fn incomplete_layouts_name_the_missing_file() {
        let cases: &[(&[&str], &str)] = &[
            (&["tokens.txt", "lexicon.txt"], "unrecognized"),
            (&["model.onnx", "lexicon.txt"], "tokens.txt"),
            (
                &["model.onnx", "tokens.txt"],
                "lexicon.txt or an espeak-ng-data",
            ),
            (
                &["model-steps-3.onnx", "tokens.txt", "lexicon.txt"],
                "vocoder",
            ),
            (
                &["model.onnx", "voices.bin", "tokens.txt", "lexicon.txt"],
                "espeak-ng-data",
            ),
            (
                &["a.onnx", "b.onnx", "tokens.txt", "lexicon.txt"],
                "several model files",
            ),
            (
                &[
                    "model-steps-3.onnx",
                    "vocos.onnx",
                    "hifigan_v2.onnx",
                    "tokens.txt",
                    "lexicon.txt",
                ],
                "several vocoders",
            ),
        ];
        for (files, expected) in cases {
            let message = message(select_tts(files));
            assert!(message.contains(expected), "{files:?}: {message}");
        }
    }
}
