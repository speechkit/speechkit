//! Transducer decoding bias (hotwords) and prompt hints for LLM families.
//!
//! Everything here is std-only and pure, apart from reading `tokens.txt`
//! through the caller.

use std::collections::HashSet;

use crate::SpeechError;

/// The longest phrase, in characters.
pub(crate) const MAX_PHRASE_CHARS: usize = 64;
/// The most phrases, engine and session combined.
pub(crate) const MAX_PHRASES: usize = 256;
/// The longest prompt, in characters, for Qwen3-ASR and FunASR-Nano.
pub(crate) const MAX_PROMPT_CHARS: usize = 64;
/// The default boost for bias phrases without their own score.
pub(crate) const DEFAULT_SCORE: f32 = 2.0;

fn invalid(message: impl Into<String>) -> SpeechError {
    SpeechError::InvalidInput(message.into())
}

/// A phrase to favor, with an optional score overriding the default.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct BiasPhrase {
    /// The phrase.
    pub text: String,
    /// Its boost, or `None` for the default.
    pub score: Option<f32>,
}

/// How phrases are split into model tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ModelingUnit {
    /// Chinese characters.
    CjkChar,
    /// BPE pieces.
    Bpe,
    /// Chinese characters and BPE pieces, for bilingual models.
    CjkCharBpe,
}

impl ModelingUnit {
    /// The name sherpa-onnx uses.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CjkChar => "cjkchar",
            Self::Bpe => "bpe",
            Self::CjkCharBpe => "cjkchar+bpe",
        }
    }

    /// Infers the unit: with `bpe.vocab`, `cjkchar+bpe` if the tokens hold
    /// CJK characters and `bpe` otherwise; without it, `cjkchar`.
    pub(crate) fn infer(has_bpe_vocab: bool, tokens: &str) -> Self {
        match (has_bpe_vocab, tokens.chars().any(is_cjk)) {
            (false, _) => Self::CjkChar,
            (true, true) => Self::CjkCharBpe,
            (true, false) => Self::Bpe,
        }
    }

    /// Whether the unit needs `bpe.vocab`.
    pub(crate) const fn needs_bpe_vocab(self) -> bool {
        !matches!(self, Self::CjkChar)
    }
}

/// A CJK ideograph, including the extension and compatibility blocks.
/// CJK punctuation does not count.
pub(crate) fn is_cjk(c: char) -> bool {
    matches!(u32::from(c),
        0x3400..=0x4DBF
        | 0x4E00..=0x9FFF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B739
        | 0x2B740..=0x2B81D
        | 0x2B820..=0x2CEA1
        | 0x2CEB0..=0x2EBE0
        | 0x2EBF0..=0x2EE5D
        | 0x2F800..=0x2FA1F
        | 0x30000..=0x3134A
        | 0x31350..=0x323AF)
}

/// Decoding bias for transducer models.
///
/// Configuring it, even with no phrases, switches decoding to
/// `modified_beam_search`, which costs roughly 2–4× greedy decoding, and
/// lets sessions pass their own hints in
/// [`AsrOptions::hints`](crate::asr::AsrOptions::hints).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) struct TransducerBias {
    /// Phrases for every session.
    pub phrases: Vec<BiasPhrase>,
    /// The boost for phrases without a score. Default: 2.0.
    pub default_score: f32,
}

impl TransducerBias {
    /// Checks the phrases and scores.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] for a bad phrase (see
    /// [`check_transducer_phrase`]), a non-finite score, or more than 256
    /// phrases.
    pub(crate) fn validate(&self) -> Result<(), SpeechError> {
        if !self.default_score.is_finite() {
            return Err(invalid("the default bias score must be finite"));
        }
        check_count(self.phrases.len())?;
        for phrase in &self.phrases {
            check_transducer_phrase(&phrase.text)?;
            if phrase.score.is_some_and(|score| !score.is_finite()) {
                return Err(invalid(format!(
                    "the score of {:?} must be finite",
                    phrase.text
                )));
            }
        }
        Ok(())
    }

    /// The phrases in sherpa-onnx's format: `a :3.5/b`.
    pub(crate) fn render(&self) -> Option<String> {
        let rendered: Vec<String> = self
            .phrases
            .iter()
            .map(|phrase| match phrase.score {
                Some(score) => format!("{} :{score}", phrase.text),
                None => phrase.text.clone(),
            })
            .collect();
        (!rendered.is_empty()).then(|| rendered.join("/"))
    }
}

fn check_count(count: usize) -> Result<(), SpeechError> {
    if count > MAX_PHRASES {
        return Err(invalid(format!(
            "at most {MAX_PHRASES} hint phrases are allowed"
        )));
    }
    Ok(())
}

fn check_phrase(phrase: &str) -> Result<(), SpeechError> {
    if phrase.trim().is_empty() {
        return Err(invalid("hint phrases must not be empty"));
    }
    if phrase.chars().count() > MAX_PHRASE_CHARS {
        return Err(invalid(format!(
            "hint phrase {phrase:?} is longer than {MAX_PHRASE_CHARS} characters"
        )));
    }
    if phrase.chars().any(char::is_control) {
        return Err(invalid("hint phrases must not contain control characters"));
    }
    Ok(())
}

/// Checks one transducer phrase: not empty, at most 64 characters, and no
/// `/`, `:`, `,`, `#`, `@`, newline, or other control character, which
/// sherpa-onnx's hotword syntax reserves.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] describing the problem.
pub(crate) fn check_transducer_phrase(phrase: &str) -> Result<(), SpeechError> {
    check_phrase(phrase)?;
    if phrase
        .chars()
        .any(|c| matches!(c, '/' | ':' | ',' | '#' | '@'))
    {
        return Err(invalid(format!(
            "hint phrase {phrase:?} must not contain '/', ':', ',', '#', or '@'"
        )));
    }
    Ok(())
}

/// Checks session hints for a transducer.
///
/// # Errors
///
/// As [`check_transducer_phrase`], or too many phrases.
pub(crate) fn check_session_hints(hints: &[String]) -> Result<(), SpeechError> {
    check_count(hints.len())?;
    hints
        .iter()
        .try_for_each(|phrase| check_transducer_phrase(phrase))
}

/// Joins the engine's rendered phrases with a session's hints.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] if the two together exceed 256 phrases.
pub(crate) fn merge(
    engine: Option<&str>,
    session: &[String],
) -> Result<Option<String>, SpeechError> {
    let session = (!session.is_empty()).then(|| session.join("/"));
    match (engine, session) {
        (None, None) => Ok(None),
        (Some(engine), None) => Ok(Some(engine.to_owned())),
        (None, Some(session)) => Ok(Some(session)),
        (Some(engine), Some(session)) => {
            let total = engine.split('/').count() + session.split('/').count();
            check_count(total)?;
            Ok(Some(format!("{engine}/{session}")))
        }
    }
}

/// The token strings of a `tokens.txt`, for the out-of-vocabulary check.
#[derive(Debug, Clone, Default)]
pub(crate) struct Vocabulary {
    unit: Option<ModelingUnit>,
    known: HashSet<String>,
}

impl Vocabulary {
    /// The vocabulary of `tokens` for `unit`. Only character units are
    /// checked; for pure BPE every phrase passes.
    pub(crate) fn new(unit: ModelingUnit, tokens: &str) -> Self {
        let known = if unit == ModelingUnit::Bpe {
            HashSet::new()
        } else {
            tokens
                .lines()
                .filter_map(|line| line.split_whitespace().next())
                .map(str::to_owned)
                .collect()
        };
        Self {
            unit: Some(unit),
            known,
        }
    }

    /// Checks that every character of `phrase` is a model token. Latin
    /// letters pass under `cjkchar+bpe`, where BPE covers them.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] listing up to eight missing characters.
    pub(crate) fn check(&self, phrase: &str) -> Result<(), SpeechError> {
        let Some(unit) = self.unit else { return Ok(()) };
        if unit == ModelingUnit::Bpe {
            return Ok(());
        }
        let missing: String = phrase
            .chars()
            .filter(|c| !c.is_whitespace())
            .filter(|c| !(unit == ModelingUnit::CjkCharBpe && c.is_ascii_alphabetic()))
            .filter(|c| !self.known.contains(c.to_string().as_str()))
            .take(8)
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(invalid(format!(
                "hint phrase {phrase:?} uses characters the model does not know: {missing}"
            )))
        }
    }
}

/// Checks prompt hints for Qwen3-ASR or FunASR-Nano and joins them with
/// commas. They may total at most 64 characters; FunASR-Nano also rejects
/// `;`, `；`, and `，`, its own separators.
///
/// # Errors
///
/// [`SpeechError::InvalidInput`] describing the problem.
pub(crate) fn prompt(hints: &[String], funasr: bool) -> Result<Option<String>, SpeechError> {
    check_count(hints.len())?;
    for phrase in hints {
        check_phrase(phrase)?;
        if phrase.contains(',') {
            return Err(invalid(format!(
                "prompt phrase {phrase:?} must not contain ','"
            )));
        }
        if funasr && phrase.chars().any(|c| matches!(c, ';' | '；' | '，')) {
            return Err(invalid(format!(
                "FunASR-Nano prompt phrase {phrase:?} must not contain ';', '；', or '，'"
            )));
        }
    }
    let joined = hints.join(",");
    let total = joined.chars().count();
    if total > MAX_PROMPT_CHARS {
        return Err(invalid(format!(
            "prompt hints total {total} characters; keep them within {MAX_PROMPT_CHARS}"
        )));
    }
    Ok((!joined.is_empty()).then_some(joined))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hints(phrases: &[&str]) -> Vec<String> {
        phrases.iter().map(|&phrase| phrase.to_owned()).collect()
    }

    #[test]
    fn transducer_phrase_table() {
        let long = "x".repeat(65);
        let cases: &[(&str, bool)] = &[
            ("语音识别", true),
            ("hello world", true),
            (&"y".repeat(64), true),
            (&long, false),
            ("", false),
            ("  ", false),
            ("a/b", false),
            ("a:b", false),
            ("a,b", false),
            ("a#b", false),
            ("a@b", false),
            ("a\nb", false),
            ("a\u{7}b", false),
        ];
        for (phrase, ok) in cases {
            assert_eq!(check_transducer_phrase(phrase).is_ok(), *ok, "{phrase:?}");
        }
        let many: Vec<String> = (0..257).map(|i| format!("p{i}")).collect();
        assert!(check_session_hints(&many).is_err());
        assert!(check_session_hints(&hints(&["ok"])).is_ok());
    }

    fn bias(phrases: &[(&str, Option<f32>)]) -> TransducerBias {
        TransducerBias {
            phrases: phrases
                .iter()
                .map(|&(text, score)| BiasPhrase {
                    text: text.into(),
                    score,
                })
                .collect(),
            default_score: DEFAULT_SCORE,
        }
    }

    #[test]
    fn bias_validation_and_rendering() {
        let named = bias(&[("语音识别", None), ("张三", Some(3.5))]);
        named.validate().unwrap();
        assert_eq!(named.render().as_deref(), Some("语音识别/张三 :3.5"));
        assert_eq!(bias(&[]).render(), None);
        let nan = TransducerBias {
            default_score: f32::NAN,
            ..bias(&[])
        };
        assert!(nan.validate().is_err());
        assert!(bias(&[("a", Some(f32::INFINITY))]).validate().is_err());
        assert!(bias(&[("a/b", None)]).validate().is_err());
    }

    #[test]
    fn merging() {
        assert_eq!(merge(None, &[]).unwrap(), None);
        assert_eq!(merge(Some("a"), &[]).unwrap().as_deref(), Some("a"));
        assert_eq!(
            merge(None, &hints(&["b", "c"])).unwrap().as_deref(),
            Some("b/c")
        );
        assert_eq!(
            merge(Some("a :2"), &hints(&["b"])).unwrap().as_deref(),
            Some("a :2/b")
        );
        let engine = vec!["e"; 200].join("/");
        let session: Vec<String> = (0..57).map(|i| i.to_string()).collect();
        assert!(merge(Some(&engine), &session).is_err());
    }

    #[test]
    fn modeling_units() {
        assert_eq!(
            ModelingUnit::infer(true, "中 1\n▁hello 2\n"),
            ModelingUnit::CjkCharBpe
        );
        assert_eq!(ModelingUnit::infer(true, "▁hello 1\n"), ModelingUnit::Bpe);
        assert_eq!(ModelingUnit::infer(false, "中 1\n"), ModelingUnit::CjkChar);
        assert_eq!(
            ModelingUnit::infer(true, "𠀀 1\n"),
            ModelingUnit::CjkCharBpe
        );
        assert!(ModelingUnit::Bpe.needs_bpe_vocab());
        assert!(!ModelingUnit::CjkChar.needs_bpe_vocab());
        assert_eq!(ModelingUnit::CjkCharBpe.as_str(), "cjkchar+bpe");
        for c in [
            '\u{20000}',
            '\u{2A700}',
            '\u{2CEB0}',
            '\u{31350}',
            '\u{F900}',
            '\u{2F800}',
            '中',
        ] {
            assert!(is_cjk(c), "{c}");
        }
        assert!(!is_cjk('a'));
        assert!(!is_cjk('。'));
    }

    #[test]
    fn out_of_vocabulary() {
        let tokens = "中 1\n文 2\n▁hello 3\n";
        let chars = Vocabulary::new(ModelingUnit::CjkChar, tokens);
        chars.check("中文").unwrap();
        let error = chars.check("中国").unwrap_err();
        assert!(error.to_string().contains('国'), "{error}");
        assert!(chars.check("中 hello").is_err());
        let bilingual = Vocabulary::new(ModelingUnit::CjkCharBpe, tokens);
        bilingual.check("中 hello").unwrap();
        Vocabulary::new(ModelingUnit::Bpe, tokens)
            .check("anything 国")
            .unwrap();
        Vocabulary::default().check("unchecked").unwrap();
    }

    #[test]
    fn prompts() {
        assert_eq!(
            prompt(&hints(&["张三", "李四"]), false).unwrap().as_deref(),
            Some("张三,李四")
        );
        assert_eq!(prompt(&hints(&[]), true).unwrap(), None);
        assert!(prompt(&hints(&["a,b"]), false).is_err());
        assert!(prompt(&hints(&["a;b"]), true).is_err());
        assert!(prompt(&hints(&["a；b"]), true).is_err());
        assert!(prompt(&hints(&["a，b"]), true).is_err());
        assert!(prompt(&hints(&["a;b"]), false).is_ok());
        let long = "x".repeat(40);
        assert!(prompt(&hints(&[&long, &long]), false).is_err());
        assert!(prompt(&hints(&[&"x".repeat(64)]), false).is_ok());
    }
}
