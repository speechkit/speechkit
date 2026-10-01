//! Configuration shared by every sherpa-onnx backend, and model resolution.
//!
//! This module is std-only. Everything a constructor checks before it
//! calls into the native library lives here, so it is tested without
//! models or native code: a layout mistake always becomes an error, never
//! a native crash.

use std::{
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use crate::SpeechError;

use crate::sherpa::layout::{self, AsrFiles, AsrModelLayout};

/// The languages SenseVoice takes, as the codes sherpa-onnx reads.
pub(crate) const SENSE_VOICE_LANGUAGES: [&str; 6] = ["auto", "zh", "en", "ja", "ko", "yue"];

/// The default number of inference threads per model.
pub(crate) const DEFAULT_THREADS: usize = 2;

/// The largest accepted number of inference threads.
const MAX_THREADS: usize = 256;

/// Where the recognizer runs.
///
/// This is a request: the native library may fall back to the CPU if the
/// provider is unavailable. Voice activity detection and punctuation
/// always run on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
#[non_exhaustive]
pub enum Provider {
    /// The CPU.
    #[default]
    Cpu,
    /// An NVIDIA GPU, on Linux or Windows, with a CUDA-enabled sherpa-onnx
    /// build (feature `sherpa-shared`).
    Cuda,
    /// Apple CoreML, which may use the CPU, GPU, or Neural Engine.
    CoreMl,
}

impl Provider {
    /// The name sherpa-onnx uses.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::CoreMl => "coreml",
        }
    }

    /// Whether this platform can have the provider: the CPU everywhere,
    /// CUDA on Linux and Windows, and CoreML on Apple platforms. Loading a
    /// model with a provider for which this is false fails with
    /// [`SpeechError::Unsupported`], so an app can hide it.
    ///
    /// True does not mean the linked sherpa-onnx build has the provider: a
    /// build without it falls back to the CPU.
    pub const fn is_supported(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Cuda => cfg!(any(target_os = "linux", target_os = "windows")),
            Self::CoreMl => cfg!(target_vendor = "apple"),
        }
    }

    /// Checks that the provider exists on this platform.
    ///
    /// # Errors
    ///
    /// [`SpeechError::Unsupported`] unless [`is_supported`](Self::is_supported).
    pub(crate) fn check_platform(self) -> Result<(), SpeechError> {
        if self.is_supported() {
            Ok(())
        } else {
            Err(SpeechError::Unsupported(format!(
                "execution provider {self} is not available on this platform"
            )))
        }
    }
}

impl fmt::Display for Provider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Provider {
    type Err = SpeechError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "coreml" => Ok(Self::CoreMl),
            _ => Err(SpeechError::InvalidInput(format!(
                "unknown execution provider {value:?}; expected cpu, cuda, or coreml"
            ))),
        }
    }
}

/// Inference settings for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Inference {
    /// Where the model runs. Default: CPU.
    pub provider: Provider,
    /// Inference threads, 1 to 256. Default: 2.
    pub threads: usize,
}

impl Default for Inference {
    fn default() -> Self {
        Self {
            provider: Provider::Cpu,
            threads: DEFAULT_THREADS,
        }
    }
}

impl Inference {
    /// Sets the provider.
    #[must_use]
    pub const fn with_provider(mut self, provider: Provider) -> Self {
        self.provider = provider;
        self
    }

    /// Sets the thread count.
    #[must_use]
    pub const fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    /// Checks the settings.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a thread count outside 1 to 256,
    /// or [`SpeechError::Unsupported`] for a provider this platform lacks.
    pub(crate) fn validate(&self) -> Result<(), SpeechError> {
        if !(1..=MAX_THREADS).contains(&self.threads) {
            return Err(SpeechError::InvalidInput(format!(
                "threads must be between 1 and {MAX_THREADS}, got {}",
                self.threads
            )));
        }
        self.provider.check_platform()
    }

    pub(crate) fn threads_i32(&self) -> i32 {
        i32::try_from(self.threads).unwrap_or(i32::MAX)
    }
}

/// A recognition model family. Some families share a file layout, so the
/// family is named rather than guessed: file names cannot tell a streaming
/// transducer from an offline one, or Paraformer from FireRedASR CTC.
/// [`inspect`](crate::sherpa::inspect) lists the families a directory fits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AsrFamily {
    /// A streaming transducer, such as streaming Zipformer.
    StreamingTransducer,
    /// An offline transducer.
    OfflineTransducer,
    /// SenseVoice.
    SenseVoice,
    /// Paraformer.
    Paraformer,
    /// FireRedASR CTC.
    FireRedCtc,
    /// FireRedASR AED.
    FireRedAed,
    /// Qwen3-ASR.
    Qwen3Asr,
    /// FunASR-Nano.
    FunAsrNano,
}

impl AsrFamily {
    /// Every family, in a stable order.
    pub const ALL: [Self; 8] = [
        Self::StreamingTransducer,
        Self::OfflineTransducer,
        Self::SenseVoice,
        Self::Paraformer,
        Self::FireRedCtc,
        Self::FireRedAed,
        Self::Qwen3Asr,
        Self::FunAsrNano,
    ];

    /// The name used on the command line.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::StreamingTransducer => "streaming-transducer",
            Self::OfflineTransducer => "transducer",
            Self::SenseVoice => "sense-voice",
            Self::Paraformer => "paraformer",
            Self::FireRedCtc => "firered-ctc",
            Self::FireRedAed => "firered-aed",
            Self::Qwen3Asr => "qwen3-asr",
            Self::FunAsrNano => "funasr-nano",
        }
    }

    /// Whether the family takes [`AsrConfig::with_hotwords`]: the
    /// transducers, as decoding bias, and Qwen3-ASR and FunASR-Nano, as a
    /// prompt. Loading any other family with hotwords fails with
    /// [`SpeechError::Unsupported`].
    ///
    /// [`AsrConfig::with_hotwords`]: crate::sherpa::AsrConfig::with_hotwords
    pub const fn supports_hotwords(self) -> bool {
        self.takes_bias() || matches!(self, Self::Qwen3Asr | Self::FunAsrNano)
    }

    /// Whether the family takes hotwords as decoding bias: only the
    /// transducers. Qwen3-ASR and FunASR-Nano take them as a prompt.
    pub(crate) const fn takes_bias(self) -> bool {
        matches!(self, Self::StreamingTransducer | Self::OfflineTransducer)
    }

    /// The languages the family takes, as codes for
    /// [`AsrConfig::with_language`] and
    /// [`AsrOptions::with_language`]; empty for a family that takes
    /// none. SenseVoice takes `auto`, which detects the language, `zh`,
    /// `en`, `ja`, `ko`, and `yue`.
    ///
    /// [`AsrConfig::with_language`]: crate::sherpa::AsrConfig::with_language
    /// [`AsrOptions::with_language`]: crate::asr::AsrOptions::with_language
    pub const fn languages(self) -> &'static [&'static str] {
        match self {
            Self::SenseVoice => &SENSE_VOICE_LANGUAGES,
            _ => &[],
        }
    }

    /// The families the model directory `dir` fits, found from its files
    /// without loading anything native, in the order of [`ALL`](Self::ALL).
    ///
    /// Some families share files, so more than one can fit: a transducer
    /// directory is a streaming or an offline transducer, and a single
    /// model with `tokens.txt` is SenseVoice, Paraformer, or FireRedASR CTC.
    /// Tokens holding SenseVoice's language markers make it SenseVoice
    /// alone, but some SenseVoice exports lack them, so their absence rules
    /// nothing out.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a directory that fits no family,
    /// naming the missing, empty, or ambiguous file when one family's files
    /// are partly there.
    pub(crate) fn detect(dir: &Path, listing: &[String]) -> Result<Vec<Self>, SpeechError> {
        let fits: &[Self] = match layout::detect_asr(listing)? {
            AsrModelLayout::Transducer => &[Self::StreamingTransducer, Self::OfflineTransducer],
            AsrModelLayout::Qwen3Asr => &[Self::Qwen3Asr],
            AsrModelLayout::FunAsrNano => &[Self::FunAsrNano],
            AsrModelLayout::FireRedAed => &[Self::FireRedAed],
            AsrModelLayout::Flat | AsrModelLayout::SenseVoice => {
                let files = resolve_listed(dir, listing, Self::SenseVoice)?;
                let AsrFiles::Flat { tokens, .. } = &files.files else {
                    return Err(SpeechError::InvalidModel("not a flat layout".into()));
                };
                if layout::is_sense_voice_tokens(&read_tokens(&files.path(tokens), tokens)?) {
                    &[Self::SenseVoice]
                } else {
                    &[Self::SenseVoice, Self::Paraformer, Self::FireRedCtc]
                }
            }
        };
        for &family in fits {
            resolve_listed(dir, listing, family)?;
        }
        Ok(fits.to_vec())
    }

    /// The file layout the family uses.
    pub(crate) const fn layout(self) -> AsrModelLayout {
        match self {
            Self::StreamingTransducer | Self::OfflineTransducer => AsrModelLayout::Transducer,
            Self::SenseVoice => AsrModelLayout::SenseVoice,
            Self::Paraformer | Self::FireRedCtc => AsrModelLayout::Flat,
            Self::FireRedAed => AsrModelLayout::FireRedAed,
            Self::Qwen3Asr => AsrModelLayout::Qwen3Asr,
            Self::FunAsrNano => AsrModelLayout::FunAsrNano,
        }
    }
}

impl fmt::Display for AsrFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for AsrFamily {
    type Err = SpeechError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|family| family.as_str() == value)
            .ok_or_else(|| {
                let names: Vec<_> = Self::ALL.iter().map(|f| f.as_str()).collect();
                SpeechError::InvalidInput(format!(
                    "unknown model family {value:?}; expected one of {}",
                    names.join(", ")
                ))
            })
    }
}

/// A model directory whose files were found and checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelFiles {
    /// The directory that was listed.
    pub(crate) root: PathBuf,
    /// The chosen files, relative to `root`.
    pub(crate) files: AsrFiles,
}

impl ModelFiles {
    /// The absolute path of a file named in `files`.
    pub(crate) fn path(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }
}

/// Lists `dir` and picks the files of `family`, before any native call.
///
/// For the flat families, `tokens.txt` is also read: SenseVoice language
/// markers contradict every family except SenseVoice.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] naming the missing, empty, ambiguous, or
/// contradicting file.
pub(crate) fn resolve(dir: &Path, family: AsrFamily) -> Result<ModelFiles, SpeechError> {
    resolve_listed(dir, &layout::list_dir(dir)?, family)
}

/// As [`resolve`], from the `listing` of `dir`.
fn resolve_listed(
    dir: &Path,
    listing: &[String],
    family: AsrFamily,
) -> Result<ModelFiles, SpeechError> {
    let files = layout::select_asr(listing, family.layout())?;
    if let AsrFiles::Flat { tokens, .. } = &files
        && family != AsrFamily::SenseVoice
        && layout::is_sense_voice_tokens(&read_tokens(&dir.join(tokens), tokens)?)
    {
        return Err(SpeechError::InvalidModel(format!(
            "{tokens} contains SenseVoice language markers, but the family is {family}"
        )));
    }
    Ok(ModelFiles {
        root: dir.to_path_buf(),
        files,
    })
}

fn read_tokens(path: &Path, name: &str) -> Result<String, SpeechError> {
    std::fs::read_to_string(path)
        .map_err(|e| SpeechError::InvalidModel(format!("cannot read {name}: {e}")))
}

/// Checks that `path` is a non-empty file.
///
/// # Errors
///
/// [`SpeechError::InvalidModel`] otherwise.
pub(crate) fn require_file(path: &Path, what: &str) -> Result<(), SpeechError> {
    match path.metadata() {
        Ok(meta) if meta.is_file() && meta.len() > 0 => Ok(()),
        Ok(meta) if meta.is_file() => Err(SpeechError::InvalidModel(format!(
            "the {what} {} is empty",
            path.display()
        ))),
        _ => Err(SpeechError::InvalidModel(format!(
            "the {what} {} is not a file",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, content).unwrap();
        }
        dir
    }

    #[test]
    fn providers_parse_and_display() {
        for provider in [Provider::Cpu, Provider::Cuda, Provider::CoreMl] {
            assert_eq!(provider.to_string().parse::<Provider>().unwrap(), provider);
        }
        assert!("gpu".parse::<Provider>().is_err());
        assert!(Provider::Cpu.check_platform().is_ok());
        for provider in [Provider::Cpu, Provider::Cuda, Provider::CoreMl] {
            assert_eq!(provider.is_supported(), provider.check_platform().is_ok());
        }
        let cuda = Provider::Cuda.check_platform();
        assert_eq!(
            cuda.is_ok(),
            cfg!(any(target_os = "linux", target_os = "windows"))
        );
        let coreml = Provider::CoreMl.check_platform();
        assert_eq!(coreml.is_ok(), cfg!(target_vendor = "apple"));
        if let Err(error) = cuda {
            assert!(matches!(error, SpeechError::Unsupported(_)));
        }
    }

    #[test]
    fn thread_counts_are_bounded() {
        assert!(Inference::default().validate().is_ok());
        assert_eq!(Inference::default().threads, 2);
        for bad in [0, 257] {
            let error = Inference::default()
                .with_threads(bad)
                .validate()
                .unwrap_err();
            assert!(matches!(error, SpeechError::InvalidInput(_)), "{error}");
        }
        assert!(Inference::default().with_threads(256).validate().is_ok());
        let cpu = Inference::default().with_provider(Provider::Cpu);
        assert_eq!(cpu.provider, Provider::Cpu);
    }

    #[test]
    fn families_parse() {
        for family in AsrFamily::ALL {
            assert_eq!(family.as_str().parse::<AsrFamily>().unwrap(), family);
            let _ = family.layout();
        }
        let error = "whisper".parse::<AsrFamily>().unwrap_err();
        assert!(error.to_string().contains("sense-voice"), "{error}");
    }

    #[test]
    fn resolve_finds_files_before_any_native_call() {
        let dir = dir_with(&[
            ("m/encoder.int8.onnx", "x"),
            ("m/decoder.onnx", "x"),
            ("m/joiner.int8.onnx", "x"),
            ("m/tokens.txt", "a 1"),
        ]);
        let files = resolve(dir.path(), AsrFamily::StreamingTransducer).unwrap();
        let AsrFiles::Transducer { encoder, .. } = &files.files else {
            panic!("{files:?}");
        };
        assert!(files.path(encoder).is_file());
    }

    #[test]
    fn invalid_directories_are_invalid_models() {
        let empty_encoder = dir_with(&[
            ("encoder.onnx", ""),
            ("decoder.onnx", "x"),
            ("joiner.onnx", "x"),
            ("tokens.txt", "a 1"),
        ]);
        let missing_joiner = dir_with(&[
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("tokens.txt", "a"),
        ]);
        let sense_voice = dir_with(&[("model.onnx", "x"), ("tokens.txt", "<|zh|> 3\n")]);
        let cases: [(&Path, AsrFamily, &str); 4] = [
            (
                empty_encoder.path(),
                AsrFamily::StreamingTransducer,
                "is empty",
            ),
            (
                missing_joiner.path(),
                AsrFamily::StreamingTransducer,
                "joiner",
            ),
            (
                sense_voice.path(),
                AsrFamily::Paraformer,
                "SenseVoice language markers",
            ),
            (
                Path::new("/nonexistent/speechkit"),
                AsrFamily::SenseVoice,
                "cannot read",
            ),
        ];
        for (dir, family, fragment) in cases {
            match resolve(dir, family) {
                Err(SpeechError::InvalidModel(message)) => {
                    assert!(message.contains(fragment), "{message}");
                }
                other => panic!("expected InvalidModel, got {other:?}"),
            }
        }
        assert!(resolve(sense_voice.path(), AsrFamily::SenseVoice).is_ok());
    }

    fn detect_dir(dir: &Path) -> Result<Vec<AsrFamily>, SpeechError> {
        AsrFamily::detect(dir, &layout::list_dir(dir)?)
    }

    #[test]
    fn detect_lists_every_family_the_files_fit() {
        use AsrFamily as F;
        let transducer = dir_with(&[
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("joiner.onnx", "x"),
            ("tokens.txt", "a 1"),
        ]);
        let sense_voice = dir_with(&[("model.int8.onnx", "x"), ("tokens.txt", "<|zh|> 3\n")]);
        let flat = dir_with(&[("model.onnx", "x"), ("tokens.txt", "a 1\n")]);
        let qwen = dir_with(&[
            ("conv_frontend.onnx", "x"),
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("tokenizer/merges.txt", "x"),
            ("tokenizer/vocab.json", "x"),
        ]);
        let nano = dir_with(&[
            ("encoder_adaptor.onnx", "x"),
            ("llm.onnx", "x"),
            ("embedding.onnx", "x"),
            ("Qwen3-0.6B/vocab.json", "x"),
            ("Qwen3-0.6B/merges.txt", "x"),
            ("Qwen3-0.6B/tokenizer.json", "x"),
        ]);
        let aed = dir_with(&[
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("tokens.txt", "a 1"),
        ]);
        let cases: [(&tempfile::TempDir, &[AsrFamily]); 6] = [
            (&transducer, &[F::StreamingTransducer, F::OfflineTransducer]),
            (&sense_voice, &[F::SenseVoice]),
            (&flat, &[F::SenseVoice, F::Paraformer, F::FireRedCtc]),
            (&qwen, &[F::Qwen3Asr]),
            (&nano, &[F::FunAsrNano]),
            (&aed, &[F::FireRedAed]),
        ];
        for (dir, families) in cases {
            assert_eq!(detect_dir(dir.path()).unwrap(), families);
        }
    }

    #[test]
    fn detect_names_what_is_wrong() {
        let empty = dir_with(&[]);
        let no_tokens = dir_with(&[
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("joiner.onnx", "x"),
        ]);
        let empty_model = dir_with(&[("model.onnx", ""), ("tokens.txt", "a 1\n")]);
        let cases: [(&Path, &str); 4] = [
            (empty.path(), "unrecognized"),
            (no_tokens.path(), "tokens.txt"),
            (empty_model.path(), "is empty"),
            (Path::new("/nonexistent/speechkit"), "cannot read"),
        ];
        for (dir, fragment) in cases {
            match detect_dir(dir) {
                Err(SpeechError::InvalidModel(message)) => {
                    assert!(message.contains(fragment), "{message}");
                }
                other => panic!("expected InvalidModel, got {other:?}"),
            }
        }
    }

    #[test]
    fn require_file_rejects_empty_and_missing() {
        let dir = dir_with(&[("vad.onnx", "x"), ("empty.onnx", "")]);
        assert!(require_file(&dir.path().join("vad.onnx"), "VAD model").is_ok());
        let empty = require_file(&dir.path().join("empty.onnx"), "VAD model").unwrap_err();
        assert!(empty.to_string().contains("is empty"), "{empty}");
        let missing = require_file(dir.path(), "VAD model").unwrap_err();
        assert!(missing.to_string().contains("not a file"), "{missing}");
    }
}
