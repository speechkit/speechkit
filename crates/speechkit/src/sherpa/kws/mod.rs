//! Keyword spotting with sherpa-onnx, for wake words.
//!
//! [`KeywordSpotter`] listens for a list of phrases, such as "Hi Jarvis" or
//! "小爱同学", with one of the sherpa-onnx `kws-zipformer` models: English
//! (`gigaspeech`), Chinese (`wenetspeech`), or both (`zh-en`). The phrases
//! are spelled in the model's tokens: `SentencePiece` pieces for English,
//! toned pinyin for Chinese, and `en.phone` phones plus pinyin for both.

mod tokenize;

use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use sherpa_onnx::{
    KeywordSpotter as NativeSpotter, KeywordSpotterConfig as NativeConfig, OnlineStream,
};

use tokenize::KeywordTokenizer;

use crate::{
    SampleRate, SpeechError,
    sherpa::{
        asr::{RATE, TAIL_PADDING},
        config::{Inference, ModelFiles},
        layout::{self, AsrFiles, AsrModelLayout},
    },
    wake::{WakeEvent, WakeWordDetector, WakeWordModel},
};

/// One wake word.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Keyword {
    /// The phrase, such as "Hi Jarvis" or "你好，小爱同学". Detections
    /// report it as [`WakeEvent::keyword`].
    pub text: String,
    /// The phrase in model tokens, such as `x iǎo y ì x iǎo y ì`, instead
    /// of the spelling found from the phrase. Use it for a Chinese
    /// character read the wrong way, or for a model the tokenizer cannot
    /// spell for.
    pub tokens: Option<String>,
    /// Boost for this keyword, instead of [`KeywordSpotterConfig::score`].
    pub score: Option<f32>,
    /// Threshold for this keyword, instead of
    /// [`KeywordSpotterConfig::threshold`].
    pub threshold: Option<f32>,
}

impl Keyword {
    /// The phrase `text`, spelled by the tokenizer, with the default score
    /// and threshold.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: None,
            score: None,
            threshold: None,
        }
    }

    /// Sets the spelling in model tokens.
    #[must_use]
    pub fn with_tokens(mut self, tokens: impl Into<String>) -> Self {
        self.tokens = Some(tokens.into());
        self
    }

    /// Sets the boost.
    #[must_use]
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }

    /// Sets the threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }
}

impl From<&str> for Keyword {
    fn from(text: &str) -> Self {
        Self::new(text)
    }
}

impl From<String> for Keyword {
    fn from(text: String) -> Self {
        Self::new(text)
    }
}

/// A sherpa-onnx keyword spotting model and the keywords to listen for.
/// [`load`](Self::load) makes a [`KeywordSpotter`].
///
/// When the directory holds several model variants, one is picked by
/// preference: `chunk-16`, then a checkpoint other than `epoch-99`. How
/// the model spells keywords is found from the files next to
/// `tokens.txt`: `bpe.model` means `SentencePiece` pieces, `en.phone`
/// means phones plus pinyin, and toned pinyin finals (`ǎo`) in
/// `tokens.txt` mean pinyin.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KeywordSpotterConfig {
    /// The model directory, such as
    /// `sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20`.
    pub model: PathBuf,
    /// The keywords. Default: none, which [`load`](Self::load) rejects.
    pub keywords: Vec<Keyword>,
    /// Where and how the model runs.
    pub inference: Inference,
    /// Boost for keyword paths during the search. Larger values trigger
    /// more easily. Default: 1.0.
    pub score: f32,
    /// Probability above which a keyword triggers, in (0, 1). Lower values
    /// trigger more easily, including falsely. Default: 0.25.
    pub threshold: f32,
}

impl KeywordSpotterConfig {
    /// Default settings for the model in `model`. Add keywords with
    /// [`with_keywords`](Self::with_keywords).
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            keywords: Vec::new(),
            inference: Inference::default(),
            score: 1.0,
            threshold: 0.25,
        }
    }

    /// Listens for `keywords`, such as `["Hi Jarvis", "小艺小艺"]`.
    #[must_use]
    pub fn with_keywords<K: Into<Keyword>>(
        mut self,
        keywords: impl IntoIterator<Item = K>,
    ) -> Self {
        self.keywords = keywords.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the inference settings.
    #[must_use]
    pub fn with_inference(mut self, inference: Inference) -> Self {
        self.inference = inference;
        self
    }

    /// Sets the default boost.
    #[must_use]
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Sets the default threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Checks the settings, the model directory, and the keywords for
    /// [`load`](Self::load) without loading anything native, so an app can
    /// tell at once whether the model can spell a phrase a user typed.
    ///
    /// # Errors
    ///
    /// As [`load`](Self::load), except for a model the native library
    /// rejects.
    pub fn validate(&self) -> Result<(), SpeechError> {
        self.check().map(drop)
    }

    /// Validates, and returns the model's files and the keywords in the
    /// form sherpa-onnx reads.
    fn check(&self) -> Result<(KwsFiles, Compiled), SpeechError> {
        self.inference.validate()?;
        check_score(self.score)?;
        check_threshold(self.threshold)?;
        let files = resolve(&self.model)?;
        let tokens_text = read(&files.tokens, "tokens file")?;
        // A model the tokenizer cannot read still works with keywords
        // given as tokens, so its error counts only for a phrase.
        let tokenizer = KeywordTokenizer::detect(&files.tokens, &tokens_text);
        let compiled = compile(&self.keywords, &tokens_text, |phrase| match &tokenizer {
            Ok(tokenizer) => tokenizer.tokenize(phrase),
            Err(error) => Err(error.clone()),
        })?;
        Ok((files, compiled))
    }
}

fn check_score(score: f32) -> Result<(), SpeechError> {
    if score.is_finite() && score > 0.0 {
        Ok(())
    } else {
        Err(SpeechError::InvalidInput(format!(
            "the keyword score must be positive, got {score}"
        )))
    }
}

fn check_threshold(threshold: f32) -> Result<(), SpeechError> {
    if threshold > 0.0 && threshold < 1.0 {
        Ok(())
    } else {
        Err(SpeechError::InvalidInput(format!(
            "the keyword threshold must be in (0, 1), got {threshold}"
        )))
    }
}

fn vocabulary(tokens: &str) -> HashSet<&str> {
    tokens
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .collect()
}

fn read(path: &Path, what: &str) -> Result<String, SpeechError> {
    std::fs::read_to_string(path).map_err(|e| {
        SpeechError::InvalidModel(format!("cannot read {what} {}: {e}", path.display()))
    })
}

/// Keywords in the form sherpa-onnx reads, and the phrase each `@name`
/// stands for.
#[derive(Debug, PartialEq)]
struct Compiled {
    buffer: String,
    names: HashMap<String, String>,
}

/// Writes `keywords` in the form sherpa-onnx reads, each named
/// `@k<index>`, and checks every token against `tokens` (the text of
/// `tokens.txt`): sherpa-onnx ends the process on an unknown token.
/// `spell` spells keywords given without tokens.
fn compile(
    keywords: &[Keyword],
    tokens: &str,
    spell: impl Fn(&str) -> Result<Vec<String>, SpeechError>,
) -> Result<Compiled, SpeechError> {
    if keywords.is_empty() {
        return Err(SpeechError::InvalidInput(
            "no keywords; list them with KeywordSpotterConfig::with_keywords".into(),
        ));
    }
    let vocabulary = vocabulary(tokens);
    let mut buffer = String::new();
    let mut names = HashMap::new();
    for (index, keyword) in keywords.iter().enumerate() {
        keyword.score.map(check_score).transpose()?;
        keyword.threshold.map(check_threshold).transpose()?;
        let mut line = match &keyword.tokens {
            Some(tokens) => tokens.split_whitespace().map(str::to_owned).collect(),
            None => spell(&keyword.text)?,
        };
        if line.is_empty() {
            return Err(SpeechError::InvalidInput(format!(
                "the keyword {:?} has no tokens",
                keyword.text
            )));
        }
        if let Some(unknown) = line.iter().find(|t| !vocabulary.contains(t.as_str())) {
            return Err(SpeechError::InvalidInput(format!(
                "the keyword {:?} is spelled with {unknown:?}, which is not a token of the model",
                keyword.text
            )));
        }
        let name = format!("k{index}");
        line.extend(keyword.score.map(|score| format!(":{score}")));
        line.extend(keyword.threshold.map(|threshold| format!("#{threshold}")));
        line.push(format!("@{name}"));
        buffer.push_str(&line.join(" "));
        buffer.push('\n');
        names.insert(name, keyword.text.clone());
    }
    Ok(Compiled { buffer, names })
}

/// What distinguishes the model files of one variant: the file name
/// without its role prefix and extensions, for example
/// `epoch-12-avg-2-chunk-16-left-64` for
/// `encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx`. `None` for files
/// that are not encoder, decoder, or joiner models.
fn variant_of(path: &str) -> Option<&str> {
    let name = path.rsplit('/').next()?;
    let stem = name.strip_suffix(".onnx")?;
    let stem = stem.strip_suffix(".int8").unwrap_or(stem);
    ["encoder", "decoder", "joiner"]
        .iter()
        .find_map(|role| stem.strip_prefix(role))
        .map(|rest| rest.trim_start_matches(['-', '_', '.']))
}

/// Keeps the candidates `good` accepts, if it accepts any.
fn prefer(candidates: &mut Vec<&str>, good: impl Fn(&str) -> bool) {
    if candidates.iter().any(|v| good(v)) {
        candidates.retain(|v| good(v));
    }
}

/// Picks the preferred variant from those in `listing`: `chunk-16`, then
/// a checkpoint other than `epoch-99`. A listing without model files
/// yields `""`, and layout selection reports what is missing.
fn pick_variant(listing: &[String]) -> Result<&str, SpeechError> {
    let mut candidates: Vec<&str> = listing.iter().filter_map(|f| variant_of(f)).collect();
    candidates.sort_unstable();
    candidates.dedup();
    prefer(&mut candidates, |v| v.contains("chunk-16"));
    prefer(&mut candidates, |v| !v.contains("epoch-99"));
    match candidates[..] {
        [] => Ok(""),
        [one] => Ok(one),
        _ => Err(SpeechError::InvalidModel(format!(
            "the model directory holds several variants: {}; keep one of them",
            candidates.join(", ")
        ))),
    }
}

/// The files of a keyword spotting model.
pub(crate) struct KwsFiles {
    encoder: PathBuf,
    decoder: PathBuf,
    joiner: PathBuf,
    tokens: PathBuf,
}

/// Lists `dir` and picks the transducer files of one variant.
pub(crate) fn resolve(dir: &Path) -> Result<KwsFiles, SpeechError> {
    let mut listing = layout::list_dir(dir)?;
    let chosen = pick_variant(&listing)?.to_owned();
    listing.retain(|name| variant_of(name).is_none_or(|v| v == chosen));
    let model = ModelFiles {
        root: dir.to_path_buf(),
        files: layout::select_asr(&listing, AsrModelLayout::Transducer)?,
    };
    let AsrFiles::Transducer {
        encoder,
        decoder,
        joiner,
        tokens,
        ..
    } = &model.files
    else {
        return Err(SpeechError::InvalidModel("not a transducer".into()));
    };
    Ok(KwsFiles {
        encoder: model.path(encoder),
        decoder: model.path(decoder),
        joiner: model.path(joiner),
        tokens: model.path(tokens),
    })
}

/// A sherpa-onnx keyword spotter: a small streaming transducer that
/// listens for wake words.
///
/// The model is loaded once and shared; each [`WakeWordModel::create`]
/// makes a new stream.
///
/// ```no_run
/// use speechkit::sherpa::{Keyword, KeywordSpotterConfig};
/// use speechkit::wake::{WakeWordDetector, WakeWordModel};
///
/// let config = KeywordSpotterConfig::new("models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
///     .with_keywords([
///         Keyword::new("Hi Jarvis"),
///         Keyword::new("你好，小爱同学"),
///         Keyword::new("小艺小艺").with_threshold(0.2),
///     ]);
/// let kws = config.load()?;
/// let mut detector = kws.create()?;
/// # let samples = vec![0.0; 16_000];
/// for event in detector.accept(&samples) {
///     println!("{} from {:?} to {:?}", event.keyword, event.start, event.end);
/// }
/// # Ok::<(), speechkit::SpeechError>(())
/// ```
pub struct KeywordSpotter {
    spotter: Arc<NativeSpotter>,
    names: Arc<HashMap<String, String>>,
    keywords: usize,
}

impl KeywordSpotterConfig {
    /// Checks the settings, the model directory, and the keywords, as
    /// [`validate`](Self::validate) does, then loads the model.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] or [`SpeechError::Unsupported`] for bad
    /// settings or keywords, including no keywords, a phrase the model
    /// cannot spell, such as an English word in a Chinese-only model or a
    /// word missing from `en.phone`, and tokens the
    /// model does not have;
    /// [`SpeechError::InvalidModel`] for a bad directory or a model the
    /// native library rejects.
    pub fn load(&self) -> Result<KeywordSpotter, SpeechError> {
        let config = self;
        let (files, compiled) = config.check()?;
        let path = |path: &Path| Some(path.to_string_lossy().into_owned());
        let mut native = NativeConfig {
            keywords_buf: Some(compiled.buffer),
            keywords_score: config.score,
            keywords_threshold: config.threshold,
            ..NativeConfig::default()
        };
        native.model_config.transducer.encoder = path(&files.encoder);
        native.model_config.transducer.decoder = path(&files.decoder);
        native.model_config.transducer.joiner = path(&files.joiner);
        native.model_config.tokens = path(&files.tokens);
        native.model_config.num_threads = config.inference.threads_i32();
        native.model_config.provider = Some(config.inference.provider.as_str().into());
        let spotter = NativeSpotter::create(&native).ok_or_else(|| {
            SpeechError::InvalidModel(format!(
                "sherpa-onnx could not load the keyword spotting model in {} with provider {}",
                config.model.display(),
                config.inference.provider
            ))
        })?;
        Ok(KeywordSpotter {
            spotter: Arc::new(spotter),
            names: Arc::new(compiled.names),
            keywords: config.keywords.len(),
        })
    }
}

impl std::fmt::Debug for KeywordSpotter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeywordSpotter")
            .field("keywords", &self.keywords)
            .finish_non_exhaustive()
    }
}

impl WakeWordModel for KeywordSpotter {
    fn sample_rate(&self) -> SampleRate {
        SampleRate::HZ_16000
    }

    fn create(&self) -> Result<Box<dyn WakeWordDetector>, SpeechError> {
        Ok(Box::new(Detector {
            stream: self.spotter.create_stream(),
            spotter: self.spotter.clone(),
            names: self.names.clone(),
            fed: 0,
            finished: false,
        }))
    }
}

struct Detector {
    spotter: Arc<NativeSpotter>,
    stream: OnlineStream,
    names: Arc<HashMap<String, String>>,
    /// Samples fed so far, excluding padding.
    fed: u64,
    finished: bool,
}

/// Seconds from sherpa-onnx as a duration, clamped to `[0, limit]`.
fn seconds(value: f32, limit: Duration) -> Duration {
    Duration::try_from_secs_f32(value.max(0.0))
        .unwrap_or(limit)
        .min(limit)
}

impl Detector {
    /// Decodes what is ready. A detection needs no reset: sherpa-onnx
    /// starts looking for the next keyword on its own, and token times
    /// stay measured from the start of the stream (a reset would restart
    /// them).
    fn decode(&self) -> Vec<WakeEvent> {
        let mut out = Vec::new();
        let heard = SampleRate::HZ_16000.duration_of(self.fed);
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
            let Some(result) = self.spotter.get_result(&self.stream) else {
                continue;
            };
            if result.keyword.is_empty() {
                continue;
            }
            let (start, end) = match (result.timestamps.first(), result.timestamps.last()) {
                (Some(&first), Some(&last)) => (seconds(first, heard), seconds(last, heard)),
                _ => (heard, heard),
            };
            let keyword = match self.names.get(&result.keyword) {
                Some(text) => text.clone(),
                None => result.keyword,
            };
            out.push(WakeEvent {
                keyword,
                start,
                end,
            });
        }
        out
    }
}

impl WakeWordDetector for Detector {
    fn accept(&mut self, samples: &[f32]) -> Vec<WakeEvent> {
        if self.finished || samples.is_empty() {
            return Vec::new();
        }
        self.stream.accept_waveform(RATE, samples);
        self.fed += samples.len() as u64;
        self.decode()
    }

    fn flush(&mut self) -> Vec<WakeEvent> {
        let finished = std::mem::replace(&mut self.finished, true);
        if finished || self.fed == 0 {
            return Vec::new();
        }
        self.stream.accept_waveform(RATE, &vec![0.0; TAIL_PADDING]);
        self.stream.input_finished();
        self.decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKENS: &str = "<blk> 0\n▁HE 1\nY 2\n▁S 3\nI 4\nRI 5\nx 6\niǎo 7\ny 8\nì 9\n";

    /// A speller for tests whose keywords all have tokens.
    fn no_speller(phrase: &str) -> Result<Vec<String>, SpeechError> {
        panic!("{phrase:?} should not need spelling")
    }

    #[test]
    fn keywords_compile_to_named_lines() {
        let keywords = [
            Keyword::new("Hey Siri").with_threshold(0.3),
            Keyword::new("小艺")
                .with_tokens("x iǎo y ì")
                .with_score(2.0),
        ];
        let spell = |phrase: &str| {
            assert_eq!(phrase, "Hey Siri");
            Ok(vec![
                "▁HE".into(),
                "Y".into(),
                "▁S".into(),
                "I".into(),
                "RI".into(),
            ])
        };
        let compiled = compile(&keywords, TOKENS, spell).unwrap();
        assert_eq!(
            compiled.buffer,
            "▁HE Y ▁S I RI #0.3 @k0\nx iǎo y ì :2 @k1\n"
        );
        assert_eq!(compiled.names["k0"], "Hey Siri");
        assert_eq!(compiled.names["k1"], "小艺");
    }

    #[test]
    fn bad_keywords_are_rejected() {
        let cases: [(Vec<Keyword>, &str); 6] = [
            (vec![], "no keywords"),
            (vec![Keyword::new("hi").with_tokens("▁HI")], "\"▁HI\""),
            (vec![Keyword::new("hi").with_tokens(" ")], "no tokens"),
            (
                vec![Keyword::new("hey").with_tokens("▁HE Y").with_threshold(0.0)],
                "threshold",
            ),
            (
                vec![Keyword::new("hey").with_tokens("▁HE Y").with_score(-1.0)],
                "score",
            ),
            (vec![Keyword::new("hey").with_tokens("hey")], "not a token"),
        ];
        for (keywords, expected) in cases {
            let error = compile(&keywords, TOKENS, no_speller).unwrap_err();
            assert!(matches!(error, SpeechError::InvalidInput(_)), "{error}");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn variants_are_told_apart_by_file_name() {
        assert_eq!(
            variant_of("m/encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
            Some("epoch-12-avg-2-chunk-16-left-64")
        );
        assert_eq!(variant_of("joiner.onnx"), Some(""));
        assert_eq!(variant_of("tokens.txt"), None);
        assert_eq!(variant_of("silero_vad.onnx"), None);

        let listing =
            |names: &[&str]| -> Vec<String> { names.iter().map(|n| format!("{n}.onnx")).collect() };
        let zh_en = listing(&[
            "encoder-epoch-13-avg-2-chunk-16-left-64",
            "encoder-epoch-13-avg-2-chunk-8-left-64",
        ]);
        assert_eq!(
            pick_variant(&zh_en).unwrap(),
            "epoch-13-avg-2-chunk-16-left-64"
        );
        let zh = listing(&[
            "encoder-epoch-12-avg-2-chunk-16-left-64",
            "encoder-epoch-99-avg-1-chunk-16-left-64",
        ]);
        assert_eq!(
            pick_variant(&zh).unwrap(),
            "epoch-12-avg-2-chunk-16-left-64"
        );
        assert_eq!(pick_variant(&[]).unwrap(), "");
        let two = listing(&["encoder-a", "encoder-b"]);
        let error = pick_variant(&two).unwrap_err();
        assert!(error.to_string().contains("several variants"), "{error}");
    }

    #[test]
    fn sherpa_seconds_are_clamped() {
        let limit = Duration::from_secs(2);
        assert_eq!(seconds(1.5, limit), Duration::from_millis(1500));
        assert_eq!(seconds(-1.0, limit), Duration::ZERO);
        assert_eq!(seconds(3.0, limit), limit);
        assert_eq!(seconds(f32::NAN, limit), Duration::ZERO);
    }

    fn model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in ["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        std::fs::write(dir.path().join("tokens.txt"), TOKENS).unwrap();
        dir
    }

    #[test]
    fn validate_accepts_keywords_the_model_can_spell() {
        let dir = model_dir();
        let config = KeywordSpotterConfig::new(dir.path())
            .with_keywords([Keyword::new("hey siri").with_tokens("▁HE Y ▁S I RI")]);
        config.validate().unwrap();
    }

    #[test]
    fn load_rejects_problems_before_any_native_call() {
        let dir = model_dir();
        let path = dir.path();
        let two = model_dir();
        std::fs::write(two.path().join("encoder-x.int8.onnx"), "x").unwrap();
        std::fs::write(two.path().join("encoder-y.int8.onnx"), "x").unwrap();
        let cases = [
            (KeywordSpotterConfig::new(path), "no keywords"),
            (
                KeywordSpotterConfig::new(path).with_threshold(1.0),
                "threshold",
            ),
            (KeywordSpotterConfig::new(path).with_score(0.0), "score"),
            (
                KeywordSpotterConfig::new(path)
                    .with_keywords([Keyword::new("hey siri").with_tokens("hey siri")]),
                "not a token",
            ),
            (
                KeywordSpotterConfig::new(path).with_keywords(["hey siri"]),
                "no bpe.model or en.phone",
            ),
            (
                KeywordSpotterConfig::new(two.path()).with_keywords(["hey"]),
                "several variants",
            ),
        ];
        for (config, expected) in cases {
            let error = config.load().unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert_eq!(
                config.validate().unwrap_err().to_string(),
                error.to_string()
            );
        }
    }
}
