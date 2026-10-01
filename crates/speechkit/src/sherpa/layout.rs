//! sherpa-onnx model directory layouts.
//!
//! Everything here is pure except [`list_dir`], which reads a directory
//! once. Detection works on the resulting list of relative file names, so
//! the rules are tested without touching disk or native code.
//!
//! Detection is one-way evidence: a marker file proves a family, but a
//! missing marker proves nothing. Streaming and offline transducer
//! archives share one layout, as do Paraformer, FireRed-CTC, and
//! markerless SenseVoice exports, so the caller names those explicitly.
//! Synthesis models are told apart by [`select_tts`].

use std::{fmt, path::Path};

use crate::SpeechError;

mod tts;

pub use tts::TtsFamily;
pub(crate) use tts::{TtsFiles, select_tts};

/// A speech recognition model family, as far as the files can tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub(crate) enum AsrModelLayout {
    /// `encoder*`, `decoder*`, `joiner*`, and `tokens.txt`: a streaming
    /// Zipformer or an offline transducer. The caller picks which.
    Transducer,
    /// `model*` and `tokens.txt` whose tokens carry SenseVoice language
    /// markers. See [`is_sense_voice_tokens`].
    SenseVoice,
    /// `model*` and `tokens.txt` without markers: Paraformer, FireRed-CTC,
    /// or a markerless SenseVoice variant. The caller names the family.
    Flat,
    /// Qwen3-ASR: `conv_frontend*`, `encoder*`, `decoder*`, and a
    /// `tokenizer/` directory with `merges.txt` and `vocab.json`.
    Qwen3Asr,
    /// FunASR-Nano: `encoder_adaptor*`, `llm*`, `embedding*`, and one
    /// tokenizer directory with `vocab.json`, `merges.txt`, and
    /// `tokenizer.json`.
    FunAsrNano,
    /// FireRed-AED: `encoder*`, `decoder*`, and `tokens.txt`, with no
    /// joiner.
    FireRedAed,
}

impl fmt::Display for AsrModelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Transducer => "transducer",
            Self::SenseVoice => "SenseVoice",
            Self::Flat => "flat (Paraformer, FireRed-CTC, or SenseVoice)",
            Self::Qwen3Asr => "Qwen3-ASR",
            Self::FunAsrNano => "FunASR-Nano",
            Self::FireRedAed => "FireRed-AED",
        })
    }
}

/// A punctuation model family, which the files of its directory tell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PunctuationFamily {
    /// Chinese and English CT-Transformer: `model*` only. It writes
    /// full-width punctuation, even for English text.
    CtTransformer,
    /// English CNN-BiLSTM: `model*` plus `bpe.vocab`. It also restores
    /// capitalization.
    CnnBiLstm,
}

impl fmt::Display for PunctuationFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CtTransformer => "CT-Transformer",
            Self::CnnBiLstm => "CNN-BiLSTM",
        })
    }
}

/// The files of a recognition model, relative to the listed directory.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) enum AsrFiles {
    /// A transducer.
    Transducer {
        /// The encoder.
        encoder: String,
        /// The decoder.
        decoder: String,
        /// The joiner.
        joiner: String,
        /// `tokens.txt`.
        tokens: String,
        /// `bpe.vocab`, if the archive has one.
        bpe_vocab: Option<String>,
    },
    /// A single model plus tokens.
    Flat {
        /// The model.
        model: String,
        /// `tokens.txt`.
        tokens: String,
    },
    /// Qwen3-ASR.
    Qwen3Asr {
        /// The convolutional front end.
        conv_frontend: String,
        /// The encoder.
        encoder: String,
        /// The decoder.
        decoder: String,
        /// The tokenizer directory.
        tokenizer: String,
    },
    /// FunASR-Nano.
    FunAsrNano {
        /// The encoder adaptor.
        encoder_adaptor: String,
        /// The language model.
        llm: String,
        /// The embedding model.
        embedding: String,
        /// The tokenizer directory.
        tokenizer: String,
    },
    /// FireRed-AED.
    FireRedAed {
        /// The encoder.
        encoder: String,
        /// The decoder.
        decoder: String,
        /// `tokens.txt`.
        tokens: String,
    },
}

/// The files of a punctuation model.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub(crate) struct PunctFiles {
    /// The family.
    pub layout: PunctuationFamily,
    /// The model.
    pub model: String,
    /// `bpe.vocab`, for CNN-BiLSTM.
    pub vocab: Option<String>,
}

fn invalid(message: impl Into<String>) -> SpeechError {
    SpeechError::InvalidModel(message.into())
}

/// The names `preferred` accepts, or all of them if it accepts none.
fn prefer(names: Vec<&str>, preferred: impl Fn(&str) -> bool) -> Vec<&str> {
    if names.iter().any(|name| preferred(name)) {
        names.into_iter().filter(|name| preferred(name)).collect()
    } else {
        names
    }
}

/// A file list with an optional single wrapper directory removed.
struct Listing<'a> {
    /// The wrapper, including its trailing `/`, or empty.
    prefix: &'a str,
    files: Vec<&'a str>,
}

impl<'a> Listing<'a> {
    /// Archives usually unpack into one top-level directory. If every file
    /// sits under the same one, look inside it.
    fn new(files: &'a [impl AsRef<str>]) -> Self {
        let files: Vec<&str> = files.iter().map(AsRef::as_ref).collect();
        let prefix = files
            .first()
            .and_then(|first| first.find('/').map(|end| &first[..=end]))
            .filter(|prefix| files.iter().all(|file| file.starts_with(prefix)))
            .unwrap_or("");
        Self {
            prefix,
            files: files.iter().map(|file| &file[prefix.len()..]).collect(),
        }
    }

    fn full(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    fn has(&self, name: &str) -> bool {
        self.files.contains(&name)
    }

    /// Top-level `.onnx` files whose name starts with `prefix`.
    fn onnx(&self, prefix: &str) -> Vec<&'a str> {
        self.files
            .iter()
            .copied()
            .filter(|f| !f.contains('/') && f.starts_with(prefix) && is_onnx(f))
            .collect()
    }

    fn has_onnx(&self, prefix: &str) -> bool {
        !self.onnx(prefix).is_empty()
    }

    /// Picks the one `prefix*.onnx` file, preferring `.int8.` names if
    /// `int8`, and plain names otherwise.
    fn pick(&self, prefix: &str, int8: bool) -> Result<String, SpeechError> {
        self.only(
            prefer(self.onnx(prefix), |name| name.contains(".int8.") == int8),
            &format!("the model directory is missing {prefix}*.onnx"),
            &format!("several equally preferred {prefix} files"),
        )
    }

    /// The full path of the one file in `names`: an error with `missing`
    /// when there is none, and one listing them after `several` otherwise.
    fn only(
        &self,
        mut names: Vec<&str>,
        missing: &str,
        several: &str,
    ) -> Result<String, SpeechError> {
        names.sort_unstable();
        match names[..] {
            [one] => Ok(self.full(one)),
            [] => Err(invalid(missing)),
            _ => Err(invalid(format!(
                "{several}: {}; keep exactly one",
                names.join(", ")
            ))),
        }
    }

    fn require(&self, name: &str) -> Result<String, SpeechError> {
        if self.has(name) {
            Ok(self.full(name))
        } else {
            Err(invalid(format!("the model directory is missing {name}")))
        }
    }

    fn optional(&self, name: &str) -> Option<String> {
        self.has(name).then(|| self.full(name))
    }

    /// Subdirectories that contain every file in `names`.
    fn dirs_with(&self, names: &[&str]) -> Vec<&'a str> {
        let mut dirs: Vec<&str> = self
            .files
            .iter()
            .filter_map(|f| f.split_once('/').map(|(dir, _)| dir))
            .filter(|dir| names.iter().all(|n| self.has(&format!("{dir}/{n}"))))
            .collect();
        dirs.sort_unstable();
        dirs.dedup();
        dirs
    }
}

/// Detects the recognition model family from a file list, as returned by
/// [`list_dir`].
///
/// File names alone cannot prove SenseVoice, so this reports
/// [`AsrModelLayout::Flat`] for it; [`is_sense_voice_tokens`] upgrades that
/// once the caller has read `tokens.txt`. Streaming and offline transducers
/// share one layout, and the caller names which one it is.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] naming the missing file when a family's
/// marker is present but its layout is incomplete, or a generic error when
/// no layout matches.
pub(crate) fn detect_asr(files: &[impl AsRef<str>]) -> Result<AsrModelLayout, SpeechError> {
    let listing = Listing::new(files);
    if listing.has_onnx("model") && listing.has("tokens.txt") {
        return Ok(AsrModelLayout::Flat);
    }
    let family = if listing.has_onnx("joiner") {
        AsrModelLayout::Transducer
    } else if listing.has_onnx("conv_frontend") {
        // Qwen3 also has an encoder and a decoder, so it is checked before
        // FireRed-AED.
        AsrModelLayout::Qwen3Asr
    } else if listing.has_onnx("encoder_adaptor") {
        AsrModelLayout::FunAsrNano
    } else if listing.has_onnx("encoder") {
        AsrModelLayout::FireRedAed
    } else {
        return Err(invalid(
            "unrecognized model directory layout; expected a transducer, SenseVoice, \
             Paraformer, FireRed, Qwen3-ASR, or FunASR-Nano model",
        ));
    };
    select_asr(files, family)?;
    Ok(family)
}

/// Picks the files of `family` from a file list. Use this when the caller
/// names the family, for example for a flat layout.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] naming what is missing or contradicts
/// the family.
pub(crate) fn select_asr(
    files: &[impl AsRef<str>],
    family: AsrModelLayout,
) -> Result<AsrFiles, SpeechError> {
    let l = Listing::new(files);
    match family {
        AsrModelLayout::Transducer => Ok(AsrFiles::Transducer {
            encoder: l.pick("encoder", true)?,
            // The float decoder is more accurate and barely slower.
            decoder: l.pick("decoder", false)?,
            joiner: l.pick("joiner", true)?,
            tokens: l.require("tokens.txt")?,
            bpe_vocab: l.optional("bpe.vocab"),
        }),
        AsrModelLayout::SenseVoice | AsrModelLayout::Flat => Ok(AsrFiles::Flat {
            model: l.pick("model", true)?,
            tokens: l.require("tokens.txt")?,
        }),
        AsrModelLayout::Qwen3Asr => {
            let conv_frontend = l.pick("conv_frontend", true)?;
            for name in ["merges.txt", "vocab.json"] {
                l.require(&format!("tokenizer/{name}"))?;
            }
            Ok(AsrFiles::Qwen3Asr {
                conv_frontend,
                encoder: l.pick("encoder", true)?,
                decoder: l.pick("decoder", true)?,
                tokenizer: l.full("tokenizer"),
            })
        }
        AsrModelLayout::FunAsrNano => {
            let encoder_adaptor = l.pick("encoder_adaptor", true)?;
            let llm = l.pick("llm", true)?;
            let embedding = l.pick("embedding", true)?;
            let tokenizer = match l.dirs_with(&["vocab.json", "merges.txt", "tokenizer.json"])[..] {
                [one] => l.full(one),
                [] => {
                    return Err(invalid(
                        "the model directory is missing a tokenizer directory with \
                         vocab.json, merges.txt, and tokenizer.json",
                    ));
                }
                _ => return Err(invalid("several tokenizer directories; keep exactly one")),
            };
            Ok(AsrFiles::FunAsrNano {
                encoder_adaptor,
                llm,
                embedding,
                tokenizer,
            })
        }
        AsrModelLayout::FireRedAed => {
            if l.has_onnx("joiner") {
                return Err(invalid(
                    "the model directory contains joiner*.onnx; this is a transducer layout, \
                     not FireRed-AED",
                ));
            }
            Ok(AsrFiles::FireRedAed {
                encoder: l.pick("encoder", true)?,
                decoder: l.pick("decoder", true)?,
                tokens: l.require("tokens.txt")?,
            })
        }
    }
}

/// Whether a `tokens.txt` carries SenseVoice language markers (`<|zh|>`).
///
/// A marker proves SenseVoice. Its absence proves nothing: markerless
/// SenseVoice variants and Paraformer have indistinguishable tokens.
pub(crate) fn is_sense_voice_tokens(tokens: &str) -> bool {
    tokens
        .lines()
        .any(|line| line.split_whitespace().next() == Some("<|zh|>"))
}

/// Picks the files of a punctuation model from a file list.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] if there is no single `model*.onnx`.
pub(crate) fn select_punct(files: &[impl AsRef<str>]) -> Result<PunctFiles, SpeechError> {
    let l = Listing::new(files);
    let model = l.pick("model", true)?;
    let vocab = l.optional("bpe.vocab");
    let layout = if vocab.is_some() {
        PunctuationFamily::CnnBiLstm
    } else {
        PunctuationFamily::CtTransformer
    };
    Ok(PunctFiles {
        layout,
        model,
        vocab,
    })
}

/// How deep [`list_dir`] looks: the wrapper, then a tokenizer directory.
const MAX_DEPTH: usize = 3;

/// Lists the files under `dir`, as `/`-separated paths relative to it,
/// sorted. This is the only function in this module that touches disk.
///
/// Hidden files are skipped, and the walk stops three levels down.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] if `dir` cannot be read, or if a model
/// file (`.onnx`, `tokens.txt`, `bpe.vocab`, a tokenizer file, a lexicon,
/// or `voices.bin`) is empty,
/// which usually means an interrupted download.
pub(crate) fn list_dir(dir: &Path) -> Result<Vec<String>, SpeechError> {
    let mut files = Vec::new();
    walk(dir, "", 0, &mut files)?;
    files.sort_unstable();
    Ok(files)
}

fn walk(dir: &Path, prefix: &str, depth: usize, out: &mut Vec<String>) -> Result<(), SpeechError> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        invalid(format!(
            "cannot read model directory {}: {e}",
            dir.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| invalid(format!("cannot read {}: {e}", dir.display())))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let path = entry.path();
        let relative = format!("{prefix}{name}");
        if path.is_dir() {
            if depth + 1 < MAX_DEPTH {
                walk(&path, &format!("{relative}/"), depth + 1, out)?;
            }
        } else if path.is_file() {
            let empty = path.metadata().map_or(true, |m| m.len() == 0);
            if empty && is_model_file(&name) {
                return Err(invalid(format!(
                    "{relative} is empty; the download may be incomplete"
                )));
            }
            out.push(relative);
        }
    }
    Ok(())
}

fn is_onnx(name: &str) -> bool {
    Path::new(name)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"))
}

fn is_model_file(name: &str) -> bool {
    tts::is_tts_model_file(name)
        || matches!(
            name,
            "tokens.txt" | "bpe.vocab" | "vocab.json" | "merges.txt" | "tokenizer.json"
        )
}

#[cfg(test)]
mod tests;
