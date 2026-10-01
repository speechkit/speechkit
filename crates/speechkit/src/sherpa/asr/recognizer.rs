//! One recognizer for every sherpa-onnx family.

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use crate::{
    SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvents, AsrOptions, AsrStream},
    sherpa::{
        AsrFamily, Inference, SileroVad, SileroVadConfig,
        asr::{
            PreparedBias, SherpaStreaming, StreamingConfig,
            offline::{
                GenericOffline, OfflineConfig, SenseVoice, SenseVoiceConfig, SenseVoiceLanguage,
            },
            prepare_bias,
        },
        bias::{self, BiasPhrase, DEFAULT_SCORE, TransducerBias},
        config::{ModelFiles, resolve},
        layout::list_dir,
        vad::MAX_SPEECH,
    },
    vad::{OfflineRecognizer, VadBackend},
};

/// A sherpa-onnx recognition model and how to run it.
///
/// Start from [`streaming`](Self::streaming) for a streaming transducer,
/// such as streaming Zipformer, or from [`offline`](Self::offline) for a
/// model that transcribes one utterance at a time behind a silero VAD:
/// SenseVoice, Paraformer, an offline transducer, FireRedASR, Qwen3-ASR,
/// or FunASR-Nano. [`load`](Self::load) loads it.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AsrConfig {
    /// The model directory.
    pub model: PathBuf,
    /// The family, or `None` to find it from the files: a transducer
    /// layout is an offline transducer, and a single model with
    /// `tokens.txt` is SenseVoice if the tokens hold its language markers.
    /// Paraformer and FireRedASR CTC share that layout, so name them.
    pub family: Option<AsrFamily>,
    /// The voice activity detector that cuts audio into utterances, for
    /// the offline families.
    pub vad: Option<SileroVadConfig>,
    /// Where and how the model runs.
    pub inference: Inference,
    /// Phrases to favor: decoding bias for transducers, which also lets
    /// sessions add their own phrases, or a prompt for Qwen3-ASR and
    /// FunASR-Nano. `None` means none; other families reject them.
    pub hotwords: Option<Vec<String>>,
    /// The language SenseVoice assumes when a session names none: `auto`
    /// (the default), `zh`, `en`, `ja`, `ko`, or `yue`.
    pub language: Option<String>,
    /// For a streaming transducer, the pause after speech that ends an
    /// utterance. The offline families end one at the VAD's
    /// `min_silence`. Default: 1 s.
    pub endpoint_silence: Duration,
    /// Where an unfinished utterance is cut and committed, as a segment
    /// boundary while the speech goes on. Default: 20 s.
    pub max_utterance: Duration,
}

/// The longest pause `with_endpoint_silence` takes.
const MAX_ENDPOINT_SILENCE: Duration = Duration::from_secs(60);

impl AsrConfig {
    /// A streaming transducer, such as streaming Zipformer, in `model`.
    pub fn streaming(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            family: Some(AsrFamily::StreamingTransducer),
            vad: None,
            inference: Inference::default(),
            hotwords: None,
            language: None,
            endpoint_silence: Duration::from_secs(1),
            max_utterance: Duration::from_secs(20),
        }
    }

    /// An offline model in `model`, behind the silero VAD model file
    /// `vad` with default settings.
    pub fn offline(model: impl Into<PathBuf>, vad: impl Into<PathBuf>) -> Self {
        Self {
            vad: Some(SileroVadConfig::new(vad)),
            family: None,
            ..Self::streaming(model)
        }
    }

    /// Names the family.
    #[must_use]
    pub fn with_family(mut self, family: AsrFamily) -> Self {
        self.family = Some(family);
        self
    }

    /// Sets the VAD and its settings.
    #[must_use]
    pub fn with_vad(mut self, vad: SileroVadConfig) -> Self {
        self.vad = Some(vad);
        self
    }

    /// Sets the inference settings.
    #[must_use]
    pub fn with_inference(mut self, inference: Inference) -> Self {
        self.inference = inference;
        self
    }

    /// Sets the phrases to favor. Even with no phrases, this switches a
    /// transducer to modified beam search, 2 to 4 times slower than greedy
    /// decoding, so that sessions can add their own phrases.
    #[must_use]
    pub fn with_hotwords(mut self, hotwords: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.hotwords = Some(hotwords.into_iter().map(Into::into).collect());
        self
    }

    /// Sets the language SenseVoice assumes when a session names none.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Sets the pause after speech that ends an utterance of a streaming
    /// transducer: positive, and at most 60 s.
    #[must_use]
    pub fn with_endpoint_silence(mut self, silence: Duration) -> Self {
        self.endpoint_silence = silence;
        self
    }

    /// Sets where an unfinished utterance is cut: positive, and at most
    /// 300 s. For an offline family it must also be longer than the VAD's
    /// `min_speech`.
    #[must_use]
    pub fn with_max_utterance(mut self, max: Duration) -> Self {
        self.max_utterance = max;
        self
    }

    /// Checks the directory and the settings for [`load`](Self::load)
    /// without loading anything native, and returns the family the model
    /// will load as.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] for a missing, empty, or ambiguous
    /// file; [`SpeechError::InvalidInput`] for bad settings or hotwords, a
    /// family that cannot be found from the files, a VAD missing for an
    /// offline family or given to a streaming one; and
    /// [`SpeechError::Unsupported`] for hotwords or a language the family
    /// does not take.
    pub fn validate(&self) -> Result<AsrFamily, SpeechError> {
        Ok(self.check()?.0)
    }

    /// Checks `self` (see [`validate`](Self::validate)), then loads the
    /// model and, for an offline family, the VAD.
    ///
    /// # Errors
    ///
    /// The errors of [`validate`](Self::validate), or
    /// [`SpeechError::InvalidModel`] for a model the native library
    /// rejects.
    pub fn load(&self) -> Result<Asr, SpeechError> {
        let (family, files, bias) = self.check()?;
        let recognizer = match family {
            AsrFamily::StreamingTransducer => {
                Recognizer::Streaming(SherpaStreaming::load(StreamingConfig {
                    files,
                    inference: self.inference,
                    bias,
                    endpoint_silence: self.endpoint_silence,
                    max_utterance: self.max_utterance,
                })?)
            }
            family => {
                let model = Offline::open(self, family, files, bias)?;
                let vad = match &self.vad {
                    Some(vad) => vad.load()?,
                    None => return Err(SpeechError::InvalidInput("no VAD".into())),
                };
                Recognizer::Offline(
                    VadBackend::new(model, vad)?.with_max_utterance(self.max_utterance),
                )
            }
        };
        Ok(Asr { recognizer, family })
    }

    /// Validates, and returns the family, its files, and the prepared
    /// decoding bias, which the loader uses as they are.
    fn check(&self) -> Result<(AsrFamily, ModelFiles, Option<PreparedBias>), SpeechError> {
        self.inference.validate()?;
        let family = match self.family {
            Some(family) => family,
            None => detect(&self.model)?,
        };
        let files = resolve(&self.model, family)?;
        self.check_vad(family)?;
        self.check_endpoints()?;
        let unsupported = |what: &str| {
            Err(SpeechError::Unsupported(format!(
                "the {family} family takes no {what}"
            )))
        };
        let bias = match &self.hotwords {
            Some(hotwords) if family.takes_bias() => {
                let bias = TransducerBias {
                    phrases: hotwords
                        .iter()
                        .map(|text| BiasPhrase {
                            text: text.clone(),
                            score: None,
                        })
                        .collect(),
                    default_score: DEFAULT_SCORE,
                };
                Some(prepare_bias(&bias, &files)?)
            }
            Some(_) if family.supports_hotwords() => {
                bias::prompt(&self.prompt(family), family == AsrFamily::FunAsrNano)?;
                None
            }
            Some(_) => return unsupported("hotwords"),
            None => None,
        };
        if let Some(language) = &self.language {
            let languages = family.languages();
            if languages.is_empty() {
                return unsupported("language");
            }
            if !languages.contains(&language.as_str()) {
                return Err(SpeechError::InvalidInput(format!(
                    "the {family} family does not support language {language:?}; use one of {}",
                    languages.join(", ")
                )));
            }
        }
        Ok((family, files, bias))
    }

    /// Whether the VAD fits the family: an offline family runs behind the
    /// VAD, and a streaming transducer without one.
    fn check_vad(&self, family: AsrFamily) -> Result<(), SpeechError> {
        let invalid = |message: String| Err(SpeechError::InvalidInput(message));
        let streaming = family == AsrFamily::StreamingTransducer;
        match (&self.vad, streaming) {
            (Some(_), true) => {
                invalid("a streaming transducer finds its own endpoints and takes no VAD".into())
            }
            (None, false) => invalid(format!(
                "the {family} family transcribes one utterance at a time and needs a VAD; \
                 start from AsrConfig::offline"
            )),
            (Some(vad), false) => vad.validate(),
            (None, true) => Ok(()),
        }
    }

    /// Whether the endpoint settings are in range.
    fn check_endpoints(&self) -> Result<(), SpeechError> {
        let silence = self.endpoint_silence;
        if silence.is_zero() || silence > MAX_ENDPOINT_SILENCE {
            return Err(SpeechError::InvalidInput(
                "the endpoint silence must be positive and at most 60 s".into(),
            ));
        }
        match &self.vad {
            Some(vad) => vad.check_max_speech(self.max_utterance),
            None if self.max_utterance.is_zero() || self.max_utterance > MAX_SPEECH => {
                Err(SpeechError::InvalidInput(format!(
                    "the longest utterance must be positive and at most {} s",
                    MAX_SPEECH.as_secs()
                )))
            }
            None => Ok(()),
        }
    }

    /// The hotwords as a prompt for the families that take hotwords but
    /// no boost (Qwen3-ASR and FunASR-Nano), and none for the others: a
    /// transducer takes them as decoding bias instead.
    fn prompt(&self, family: AsrFamily) -> Vec<String> {
        if !family.supports_hotwords() || family.takes_bias() {
            return Vec::new();
        }
        self.hotwords.iter().flatten().cloned().collect()
    }
}

/// The family of the offline model in `dir`, found from its files: the one
/// family they fit, or an offline transducer for a transducer layout.
fn detect(dir: &Path) -> Result<AsrFamily, SpeechError> {
    match AsrFamily::detect(dir, &list_dir(dir)?)?.as_slice() {
        [family] => Ok(*family),
        [AsrFamily::StreamingTransducer, AsrFamily::OfflineTransducer] => {
            Ok(AsrFamily::OfflineTransducer)
        }
        families => {
            let names: Vec<_> = families.iter().map(ToString::to_string).collect();
            Err(SpeechError::InvalidInput(format!(
                "{} can be any of {}; name the family",
                dir.display(),
                names.join(", ")
            )))
        }
    }
}

/// The recognizer a family runs on.
enum Recognizer {
    Streaming(SherpaStreaming),
    Offline(VadBackend<Offline, SileroVad>),
}

/// A loaded sherpa-onnx recognizer, for any family.
///
/// A streaming transducer reports partial results and commits a segment at
/// each pause; the offline families commit one segment per utterance the
/// VAD finds. [`AsrConfig::load`] makes one.
pub struct Asr {
    recognizer: Recognizer,
    family: AsrFamily,
}

impl Asr {
    /// The family that was loaded.
    pub fn family(&self) -> AsrFamily {
        self.family
    }

    fn backend(&self) -> &dyn AsrBackend {
        match &self.recognizer {
            Recognizer::Streaming(backend) => backend,
            Recognizer::Offline(backend) => backend,
        }
    }
}

impl AsrBackend for Asr {
    fn name(&self) -> &str {
        self.backend().name()
    }

    fn capabilities(&self) -> &AsrCapabilities {
        self.backend().capabilities()
    }

    fn open(
        &self,
        opts: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        self.backend().open(opts, events)
    }
}

impl std::fmt::Debug for Asr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Asr")
            .field("family", &self.family)
            .field("capabilities", self.capabilities())
            .finish_non_exhaustive()
    }
}

/// A sherpa-onnx offline model: SenseVoice, Paraformer, an offline
/// transducer, FireRedASR, Qwen3-ASR, or FunASR-Nano, as an
/// [`OfflineRecognizer`] that recognizes the audio of each call as one
/// utterance. [`Asr`] runs it behind silero VAD.
pub(crate) struct Offline {
    model: OfflineModel,
    family: AsrFamily,
}

/// The model an offline family runs on.
enum OfflineModel {
    SenseVoice(SenseVoice),
    Other(GenericOffline),
}

impl Offline {
    /// Loads the offline model `config` describes, from its checked files
    /// and bias.
    fn open(
        config: &AsrConfig,
        family: AsrFamily,
        files: ModelFiles,
        bias: Option<PreparedBias>,
    ) -> Result<Self, SpeechError> {
        let inference = config.inference;
        let model = if family == AsrFamily::SenseVoice {
            let language = match &config.language {
                Some(language) => language.parse()?,
                None => SenseVoiceLanguage::Auto,
            };
            OfflineModel::SenseVoice(SenseVoice::load(&SenseVoiceConfig {
                files,
                language,
                inference,
            })?)
        } else {
            OfflineModel::Other(GenericOffline::load(OfflineConfig {
                files,
                family,
                inference,
                bias,
                prompt_hints: config.prompt(family),
            })?)
        };
        Ok(Self { model, family })
    }

    fn recognizer(&self) -> &dyn OfflineRecognizer {
        match &self.model {
            OfflineModel::SenseVoice(model) => model,
            OfflineModel::Other(model) => model,
        }
    }
}

impl OfflineRecognizer for Offline {
    fn name(&self) -> &str {
        self.recognizer().name()
    }

    fn capabilities(&self) -> &AsrCapabilities {
        self.recognizer().capabilities()
    }

    fn recognize(&self, samples: &[f32], opts: &AsrOptions) -> Result<String, SpeechError> {
        self.recognizer().recognize(samples, opts)
    }
}

impl std::fmt::Debug for Offline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Offline")
            .field("family", &self.family)
            .field("capabilities", self.capabilities())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory holding `files`, and a non-empty VAD model file.
    fn model(files: &[(&str, &str)]) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        for (name, content) in files {
            std::fs::write(dir.path().join(name), content).unwrap();
        }
        let vad = dir.path().join("vad").join("silero_vad.onnx");
        std::fs::create_dir(vad.parent().unwrap()).unwrap();
        std::fs::write(&vad, "x").unwrap();
        (dir, vad)
    }

    fn transducer() -> (tempfile::TempDir, PathBuf) {
        model(&[
            ("encoder.onnx", "x"),
            ("decoder.onnx", "x"),
            ("joiner.onnx", "x"),
            ("tokens.txt", "中 1\n文 2\n"),
        ])
    }

    #[test]
    fn families_are_found_from_the_files() {
        let (dir, vad) = transducer();
        let found = AsrConfig::offline(dir.path(), &vad).validate();
        assert_eq!(found.unwrap(), AsrFamily::OfflineTransducer);
        let (dir, vad) = model(&[("model.onnx", "x"), ("tokens.txt", "<|en|> 1\n<|zh|> 2\n")]);
        let found = AsrConfig::offline(dir.path(), &vad).validate();
        assert_eq!(found.unwrap(), AsrFamily::SenseVoice);
        let (dir, vad) = model(&[("model.onnx", "x"), ("tokens.txt", "a 1\n")]);
        let error = AsrConfig::offline(dir.path(), &vad).validate().unwrap_err();
        assert!(error.to_string().contains("name the family"), "{error}");
        assert!(
            error
                .to_string()
                .contains("sense-voice, paraformer, firered-ctc"),
            "{error}"
        );
        let named = AsrConfig::offline(dir.path(), &vad).with_family(AsrFamily::Paraformer);
        assert_eq!(named.validate().unwrap(), AsrFamily::Paraformer);
    }

    /// A directory holding a model of `family`, and a VAD model file.
    fn model_of(family: AsrFamily) -> (tempfile::TempDir, PathBuf) {
        let files: &[&str] = match family {
            AsrFamily::StreamingTransducer | AsrFamily::OfflineTransducer => {
                return transducer();
            }
            AsrFamily::SenseVoice => {
                return model(&[("model.onnx", "x"), ("tokens.txt", "<|zh|> 1\n")]);
            }
            AsrFamily::Paraformer | AsrFamily::FireRedCtc => &["model.onnx"],
            AsrFamily::FireRedAed => &["encoder.onnx", "decoder.onnx"],
            AsrFamily::Qwen3Asr => &[
                "conv_frontend.onnx",
                "encoder.onnx",
                "decoder.onnx",
                "tokenizer/merges.txt",
                "tokenizer/vocab.json",
            ],
            AsrFamily::FunAsrNano => &[
                "encoder_adaptor.onnx",
                "llm.onnx",
                "embedding.onnx",
                "Qwen3-0.6B/vocab.json",
                "Qwen3-0.6B/merges.txt",
                "Qwen3-0.6B/tokenizer.json",
            ],
        };
        let (dir, vad) = model(&[("tokens.txt", "中 1\n文 2\n")]);
        for name in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "x").unwrap();
        }
        (dir, vad)
    }

    #[test]
    fn families_say_what_validate_accepts() {
        for family in AsrFamily::ALL {
            let (dir, vad) = model_of(family);
            let config = if family == AsrFamily::StreamingTransducer {
                AsrConfig::streaming(dir.path())
            } else {
                AsrConfig::offline(dir.path(), &vad).with_family(family)
            };
            assert_eq!(config.validate().unwrap(), family);
            let with_hotwords = config.clone().with_hotwords(["中文"]);
            let checked = with_hotwords.check();
            assert_eq!(checked.is_ok(), family.supports_hotwords(), "{family}");
            // Accepted hotwords reach the model, as bias or as a prompt.
            if let Ok((_, _, bias)) = checked {
                let prompt = with_hotwords.prompt(family);
                assert_eq!(bias.is_some(), family.takes_bias(), "{family}");
                assert_eq!(prompt.is_empty(), bias.is_some(), "{family}");
            }
            for language in family.languages() {
                let config = config.clone().with_language(*language);
                assert!(config.validate().is_ok(), "{family} {language}");
            }
            let other = config.clone().with_language("fr").validate();
            assert!(other.is_err(), "{family}");
        }
        assert_eq!(
            AsrFamily::SenseVoice.languages(),
            ["auto", "zh", "en", "ja", "ko", "yue"]
        );
    }

    #[test]
    fn a_vad_goes_with_the_offline_families_only() {
        let (dir, vad) = transducer();
        let streaming = AsrConfig::streaming(dir.path());
        assert_eq!(
            streaming.validate().unwrap(),
            AsrFamily::StreamingTransducer
        );
        let with_vad = streaming.with_vad(SileroVadConfig::new(&vad));
        assert!(matches!(
            with_vad.validate(),
            Err(SpeechError::InvalidInput(_))
        ));
        let mut without = AsrConfig::offline(dir.path(), &vad);
        without.vad = None;
        assert!(matches!(
            without.validate(),
            Err(SpeechError::InvalidInput(_))
        ));
    }

    #[test]
    fn only_llm_families_take_hotwords_as_a_prompt() {
        let (dir, vad) = transducer();
        let config = AsrConfig::offline(dir.path(), &vad).with_hotwords(["中文"]);
        assert!(config.prompt(AsrFamily::OfflineTransducer).is_empty());
        assert_eq!(config.prompt(AsrFamily::Qwen3Asr), ["中文"]);
        assert_eq!(config.prompt(AsrFamily::FunAsrNano), ["中文"]);
    }

    #[test]
    fn hotwords_follow_the_family() {
        let (dir, vad) = transducer();
        let biased = AsrConfig::streaming(dir.path()).with_hotwords(["中文"]);
        assert!(biased.validate().is_ok());
        let unknown = AsrConfig::streaming(dir.path()).with_hotwords(["语音"]);
        assert!(
            unknown
                .validate()
                .unwrap_err()
                .to_string()
                .contains("does not know")
        );
        let (dir, vad2) = model(&[("model.onnx", "x"), ("tokens.txt", "<|zh|> 1\n")]);
        let sense = AsrConfig::offline(dir.path(), &vad2).with_hotwords(["x"]);
        assert!(matches!(sense.validate(), Err(SpeechError::Unsupported(_))));
        let language = AsrConfig::offline(dir.path(), &vad2).with_language("fr");
        assert!(matches!(
            language.validate(),
            Err(SpeechError::InvalidInput(_))
        ));
        let (dir, _) = transducer();
        let paraformer = AsrConfig::offline(dir.path(), &vad)
            .with_family(AsrFamily::OfflineTransducer)
            .with_language("zh");
        assert!(matches!(
            paraformer.validate(),
            Err(SpeechError::Unsupported(_))
        ));
    }

    #[test]
    fn endpoint_settings_are_checked() {
        let (dir, vad) = transducer();
        let streaming = AsrConfig::streaming(dir.path());
        assert_eq!(streaming.endpoint_silence, Duration::from_secs(1));
        assert_eq!(streaming.max_utterance, Duration::from_secs(20));
        let offline = AsrConfig::offline(dir.path(), &vad);
        let bad = [
            streaming.clone().with_endpoint_silence(Duration::ZERO),
            streaming
                .clone()
                .with_endpoint_silence(Duration::from_secs(61)),
            streaming.clone().with_max_utterance(Duration::ZERO),
            streaming
                .clone()
                .with_max_utterance(Duration::from_secs(301)),
            offline
                .clone()
                .with_max_utterance(Duration::from_millis(100)),
        ];
        for config in bad {
            assert!(
                matches!(config.validate(), Err(SpeechError::InvalidInput(_))),
                "{config:?}"
            );
        }
        let good = offline.with_max_utterance(Duration::from_secs(300));
        assert!(good.validate().is_ok());
    }

    #[test]
    fn bad_settings_fail_before_native_code() {
        let (dir, vad) = transducer();
        let error = AsrConfig::offline(dir.path(), &vad)
            .with_family(AsrFamily::FireRedAed)
            .load()
            .unwrap_err();
        assert!(error.to_string().contains("transducer layout"), "{error}");
        let (dir, vad) = model(&[("model.onnx", ""), ("tokens.txt", "<|zh|> 1\n")]);
        let empty = AsrConfig::offline(dir.path(), &vad).with_family(AsrFamily::SenseVoice);
        assert!(matches!(empty.load(), Err(SpeechError::InvalidModel(_))));
        let threads = empty.with_inference(Inference::default().with_threads(0));
        assert!(matches!(threads.load(), Err(SpeechError::InvalidInput(_))));
        let (dir, vad) = model(&[]);
        let prompt = AsrConfig::offline(dir.path(), &vad)
            .with_family(AsrFamily::FunAsrNano)
            .with_hotwords(["a;b"]);
        assert!(matches!(prompt.load(), Err(SpeechError::InvalidModel(_))));
    }
}
