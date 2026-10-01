//! Turning keyword phrases into the tokens of a keyword spotting model.
//!
//! sherpa-onnx only reads keywords already written as model tokens, as
//! `sherpa-onnx-cli text2token` writes them. This module does the same
//! in Rust for the three kinds of published models.

use std::{collections::HashMap, path::Path};

use pinyin::ToPinyin;

use crate::SpeechError;

/// Pinyin initials, two-letter ones first. `y` and `w` count as initials,
/// as in the models' training labels.
const INITIALS: [&str; 23] = [
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r",
    "z", "c", "s", "y", "w",
];

/// How a model spells keywords.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum KeywordUnit {
    /// `SentencePiece` pieces from `bpe.model`, for English models such as
    /// `kws-zipformer-gigaspeech`.
    Bpe,
    /// Pinyin initials and toned finals, for Chinese models such as
    /// `kws-zipformer-wenetspeech`.
    Pinyin,
    /// Pinyin for Chinese characters and `ARPAbet` phones from `en.phone`
    /// for English words, for `kws-zipformer-zh-en`.
    PhonePinyin,
}

/// Spells keyword phrases, such as "Hi Jarvis" or "小爱同学", in the tokens
/// of one keyword spotting model.
///
/// Chinese characters are read one at a time with their most common
/// reading, so a phrase with a polyphonic character may be misspelled;
/// give such a keyword its tokens with
/// [`Keyword::with_tokens`](super::Keyword::with_tokens).
pub(crate) struct KeywordTokenizer(Speller);

/// A [`KeywordUnit`] with what it needs to spell.
enum Speller {
    Bpe(Unigram),
    Pinyin,
    /// The lexicon maps an uppercased word to its phones.
    PhonePinyin(HashMap<String, Vec<String>>),
}

impl std::fmt::Debug for KeywordTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("KeywordTokenizer")
            .field(&self.unit())
            .finish()
    }
}

fn invalid_model(message: String) -> SpeechError {
    SpeechError::InvalidModel(message)
}

fn read(path: &Path) -> Result<String, SpeechError> {
    std::fs::read_to_string(path)
        .map_err(|e| invalid_model(format!("cannot read {}: {e}", path.display())))
}

/// Whether `tokens.txt` has toned pinyin finals.
fn has_pinyin(tokens: &str) -> bool {
    tokens
        .lines()
        .any(|line| matches!(line.split_whitespace().next(), Some("ǎo" | "ǐ")))
}

impl KeywordTokenizer {
    /// Reads the tokenizer of the model whose tokens are in `tokens`
    /// (`tokens.txt`). The kind is found from the files: `bpe.model` next
    /// to `tokens.txt` means [`KeywordUnit::Bpe`],
    /// `en.phone` means [`KeywordUnit::PhonePinyin`], and toned pinyin
    /// finals (`ǎo`) in `tokens.txt` mean [`KeywordUnit::Pinyin`].
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidModel`] if a needed file cannot be read or
    /// parsed, or [`SpeechError::Unsupported`] if the model spells keywords
    /// some other way; such a model needs keywords given as tokens.
    #[cfg(test)]
    pub(crate) fn load(tokens: &Path) -> Result<Self, SpeechError> {
        Self::detect(tokens, &read(tokens)?)
    }

    /// As [`load`](Self::load), with `tokens.txt` already read.
    pub(crate) fn detect(path: &Path, tokens: &str) -> Result<Self, SpeechError> {
        let bpe = path.with_file_name("bpe.model");
        let phones = path.with_file_name("en.phone");
        let unit = if bpe.is_file() {
            KeywordUnit::Bpe
        } else if phones.is_file() {
            KeywordUnit::PhonePinyin
        } else if has_pinyin(tokens) {
            KeywordUnit::Pinyin
        } else {
            return Err(SpeechError::Unsupported(format!(
                "cannot spell keyword phrases for the model with {}: it has no bpe.model or \
                 en.phone, and no pinyin tokens; give each keyword its tokens with \
                 Keyword::with_tokens, as `sherpa-onnx-cli text2token` writes them",
                path.display()
            )));
        };
        let speller = match unit {
            KeywordUnit::Bpe => {
                let bytes = std::fs::read(&bpe)
                    .map_err(|e| invalid_model(format!("cannot read {}: {e}", bpe.display())))?;
                Speller::Bpe(Unigram::parse(&bytes)?)
            }
            KeywordUnit::PhonePinyin => Speller::PhonePinyin(parse_lexicon(&read(&phones)?)),
            KeywordUnit::Pinyin => Speller::Pinyin,
        };
        Ok(Self(speller))
    }

    /// How the model spells keywords.
    pub(crate) fn unit(&self) -> KeywordUnit {
        match self.0 {
            Speller::Bpe(_) => KeywordUnit::Bpe,
            Speller::Pinyin => KeywordUnit::Pinyin,
            Speller::PhonePinyin(_) => KeywordUnit::PhonePinyin,
        }
    }

    /// The spelling of `phrase` in model tokens. Punctuation is ignored,
    /// and case does not matter.
    ///
    /// # Errors
    ///
    /// [`SpeechError::InvalidInput`] naming a word the model cannot spell,
    /// such as an English word in a Chinese-only model or a word missing
    /// from `en.phone`, or if the phrase has no words.
    pub(crate) fn tokenize(&self, phrase: &str) -> Result<Vec<String>, SpeechError> {
        let words = split_words(phrase);
        if words.is_empty() {
            return Err(SpeechError::InvalidInput(format!(
                "the keyword {phrase:?} has no words"
            )));
        }
        if let Speller::Bpe(pieces) = &self.0 {
            let mut text = String::new();
            for word in &words {
                let Word::Latin(word) = word else {
                    return Err(SpeechError::InvalidInput(format!(
                        "the keyword {phrase:?} has Chinese characters, but the model is \
                         English only"
                    )));
                };
                text.push('▁');
                text.push_str(word);
            }
            return pieces.encode(&text).ok_or_else(|| {
                SpeechError::InvalidInput(format!("the model cannot spell the keyword {phrase:?}"))
            });
        }
        let mut tokens = Vec::new();
        for word in &words {
            match (word, &self.0) {
                (Word::Han(pinyin), _) => tokens.extend(split_pinyin(pinyin)),
                (Word::Latin(word), Speller::PhonePinyin(lexicon)) => {
                    let phones = lexicon.get(word).ok_or_else(|| {
                        SpeechError::InvalidInput(format!(
                            "the word {word:?} in the keyword {phrase:?} is not in the \
                             model's en.phone"
                        ))
                    })?;
                    tokens.extend(phones.iter().cloned());
                }
                (Word::Latin(word), _) => {
                    return Err(SpeechError::InvalidInput(format!(
                        "the word {word:?} in the keyword {phrase:?} is not Chinese, and the \
                         model is Chinese only"
                    )));
                }
            }
        }
        Ok(tokens)
    }
}

/// A word of a phrase.
#[derive(Debug, PartialEq)]
enum Word {
    /// An uppercased Latin word, with digits and apostrophes.
    Latin(String),
    /// The toned pinyin of one Chinese character.
    Han(String),
}

fn split_words(phrase: &str) -> Vec<Word> {
    let mut words = Vec::new();
    let mut latin = String::new();
    for ch in phrase.chars() {
        let han = ch.to_pinyin();
        if han.is_none() && (ch.is_alphanumeric() || ch == '\'') {
            latin.extend(ch.to_uppercase());
            continue;
        }
        if !latin.is_empty() {
            words.push(Word::Latin(std::mem::take(&mut latin)));
        }
        if let Some(pinyin) = han {
            words.push(Word::Han(pinyin.with_tone().to_owned()));
        }
    }
    if !latin.is_empty() {
        words.push(Word::Latin(latin));
    }
    words
}

/// Splits toned pinyin into its initial and final: `xiǎo` → `x iǎo`.
fn split_pinyin(pinyin: &str) -> Vec<String> {
    for initial in INITIALS {
        if let Some(rest) = pinyin.strip_prefix(initial)
            && !rest.is_empty()
        {
            return vec![initial.to_owned(), rest.to_owned()];
        }
    }
    vec![pinyin.to_owned()]
}

/// `WORD PH1 PH2 ...` lines. A word listed twice keeps its first
/// pronunciation; the published `en.phone` lists each word once.
fn parse_lexicon(text: &str) -> HashMap<String, Vec<String>> {
    let mut lexicon = HashMap::new();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(word) = fields.next() else {
            continue;
        };
        let phones: Vec<String> = fields.map(str::to_owned).collect();
        if !phones.is_empty() {
            lexicon.entry(word.to_uppercase()).or_insert(phones);
        }
    }
    lexicon
}

/// A `SentencePiece` unigram model: pieces and their log probabilities.
#[derive(Debug)]
struct Unigram {
    scores: HashMap<String, f32>,
    /// The longest piece, in characters.
    longest: usize,
}

impl Unigram {
    /// Reads the pieces of a serialized `SentencePiece` `ModelProto`.
    fn parse(bytes: &[u8]) -> Result<Self, SpeechError> {
        let bad = || invalid_model("bpe.model is not a SentencePiece model".into());
        let mut scores = HashMap::new();
        let mut model_type = 1;
        for (number, value) in fields(bytes).ok_or_else(bad)? {
            match (number, value) {
                // pieces: { 1: piece, 2: score, 3: type }
                (1, Value::Bytes(piece)) => {
                    let (mut text, mut score, mut kind) = (None, 0.0, 1);
                    for field in fields(piece).ok_or_else(bad)? {
                        match field {
                            (1, Value::Bytes(s)) => {
                                text = Some(std::str::from_utf8(s).map_err(|_| bad())?);
                            }
                            (2, Value::Fixed32(bits)) => score = f32::from_bits(bits),
                            (3, Value::Varint(t)) => kind = t,
                            _ => {}
                        }
                    }
                    // Only normal pieces take part in segmentation.
                    if let (Some(text), 1) = (text, kind) {
                        scores.insert(text.to_owned(), score);
                    }
                }
                // trainer_spec: { 3: model_type }
                (2, Value::Bytes(spec)) => {
                    for field in fields(spec).ok_or_else(bad)? {
                        if let (3, Value::Varint(t)) = field {
                            model_type = t;
                        }
                    }
                }
                _ => {}
            }
        }
        if model_type != 1 {
            return Err(SpeechError::Unsupported(format!(
                "bpe.model has SentencePiece model type {model_type}; only unigram models \
                 (type 1) are supported; give each keyword its tokens with \
                 Keyword::with_tokens"
            )));
        }
        if scores.is_empty() {
            return Err(bad());
        }
        let longest = scores.keys().map(|p| p.chars().count()).max().unwrap_or(1);
        Ok(Self { scores, longest })
    }

    /// The most likely segmentation of `text`, or `None` if the pieces
    /// cannot cover it.
    fn encode(&self, text: &str) -> Option<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        // best[end] = (score, start) of the best segmentation of chars[..end].
        let mut best: Vec<Option<(f32, usize)>> = vec![None; chars.len() + 1];
        best[0] = Some((0.0, 0));
        for end in 1..=chars.len() {
            for start in end.saturating_sub(self.longest)..end {
                let Some((before, _)) = best[start] else {
                    continue;
                };
                let piece: String = chars[start..end].iter().collect();
                if let Some(score) = self.scores.get(&piece) {
                    let total = before + score;
                    if best[end].is_none_or(|(current, _)| total > current) {
                        best[end] = Some((total, start));
                    }
                }
            }
        }
        let mut tokens = Vec::new();
        let mut end = chars.len();
        while end > 0 {
            let (_, start) = best[end]?;
            tokens.push(chars[start..end].iter().collect());
            end = start;
        }
        tokens.reverse();
        Some(tokens)
    }
}

enum Value<'a> {
    Varint(u64),
    Fixed32(u32),
    Bytes(&'a [u8]),
    Other,
}

/// The fields of a protobuf message, or `None` if it is malformed.
fn fields(mut bytes: &[u8]) -> Option<Vec<(u64, Value<'_>)>> {
    fn varint(bytes: &mut &[u8]) -> Option<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = bytes.split_first()?;
            *bytes = rest;
            value |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Some(value);
            }
        }
        None
    }
    fn take<'a>(bytes: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        let (head, rest) = bytes.split_at_checked(n)?;
        *bytes = rest;
        Some(head)
    }
    let mut out = Vec::new();
    while !bytes.is_empty() {
        let key = varint(&mut bytes)?;
        let value = match key & 7 {
            0 => Value::Varint(varint(&mut bytes)?),
            1 => {
                take(&mut bytes, 8)?;
                Value::Other
            }
            2 => {
                let n = usize::try_from(varint(&mut bytes)?).ok()?;
                Value::Bytes(take(&mut bytes, n)?)
            }
            5 => Value::Fixed32(u32::from_le_bytes(take(&mut bytes, 4)?.try_into().ok()?)),
            _ => return None,
        };
        out.push((key >> 3, value));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bpe(pieces: &[(&str, f32)]) -> KeywordTokenizer {
        KeywordTokenizer(Speller::Bpe(Unigram {
            scores: pieces.iter().map(|(p, s)| ((*p).to_owned(), *s)).collect(),
            longest: pieces.iter().map(|(p, _)| p.chars().count()).max().unwrap(),
        }))
    }

    fn spell(tokenizer: &KeywordTokenizer, phrase: &str) -> String {
        tokenizer.tokenize(phrase).unwrap().join(" ")
    }

    #[test]
    fn chinese_becomes_initials_and_toned_finals() {
        let zh = KeywordTokenizer(Speller::Pinyin);
        assert_eq!(
            spell(&zh, "你好，小爱同学！"),
            "n ǐ h ǎo x iǎo ài t óng x ué"
        );
        assert_eq!(spell(&zh, "小艺小艺"), "x iǎo y ì x iǎo y ì");
        let error = zh.tokenize("hi 小艺").unwrap_err();
        assert!(error.to_string().contains("Chinese only"), "{error}");
    }

    #[test]
    fn english_words_use_their_first_pronunciation() {
        let lexicon = "HI HH AY1\nJARVIS JH AA1 R V AH0 S\nREAD R IY1 D\nREAD R EH1 D\n";
        let zh_en = KeywordTokenizer(Speller::PhonePinyin(parse_lexicon(lexicon)));
        assert_eq!(spell(&zh_en, "Hi, Jarvis"), "HH AY1 JH AA1 R V AH0 S");
        assert_eq!(spell(&zh_en, "read 小艺"), "R IY1 D x iǎo y ì");
        let error = zh_en.tokenize("hi speechkit").unwrap_err();
        assert!(error.to_string().contains("\"SPEECHKIT\""), "{error}");
        assert!(zh_en.tokenize(" ,!").is_err());
    }

    #[test]
    fn unigram_picks_the_most_likely_pieces() {
        let pieces = [
            ("▁HE", -5.0),
            ("Y", -4.0),
            ("▁", -3.0),
            ("H", -6.0),
            ("E", -4.0),
            ("▁HEY", -12.0),
        ];
        let en = bpe(&pieces);
        // ▁HE + Y = -9 beats ▁HEY = -12 and ▁ + H + E + Y = -17.
        assert_eq!(spell(&en, "hey"), "▁HE Y");
        assert!(en.tokenize("hex").is_err());
        assert!(en.tokenize("你好").is_err());
    }

    #[test]
    fn malformed_models_are_errors() {
        assert!(Unigram::parse(&[0x0a, 0x05, 0x01]).is_err());
        assert!(Unigram::parse(&[]).is_err());
        // One piece "A" (score -1) in a BPE (type 2) model.
        let bpe = [
            0x0a, 0x08, 0x0a, 0x01, b'A', 0x15, 0x00, 0x00, 0x80, 0xbf, 0x12, 0x02, 0x18, 0x02,
        ];
        let error = Unigram::parse(&bpe).unwrap_err();
        assert!(matches!(error, SpeechError::Unsupported(_)), "{error}");
    }
}

/// Spellings checked against each model's own example keywords, the plain
/// phrases next to their `text2token` output. Ignored by default; run them
/// after `cargo xtask fetch-fixtures`, which sets `SPEECHKIT_MODEL_*`.
#[cfg(test)]
mod model_tests {
    use std::path::{Path, PathBuf};

    use super::*;

    /// The directory of model `id`, from `SPEECHKIT_MODEL_<ID>`.
    fn model_dir(id: &str) -> Option<PathBuf> {
        let var = format!("SPEECHKIT_MODEL_{}", id.to_uppercase().replace('-', "_"));
        std::env::var_os(var).map(PathBuf::from)
    }

    /// `(phrase, tokens)` pairs from a model's plain and tokenized keyword
    /// files, whose lines may end in `@name`. With `raw` set to `None`, the
    /// name is the phrase.
    fn examples(dir: &Path, raw: Option<&str>, tokenized: &str) -> Vec<(String, String)> {
        let tokenized = std::fs::read_to_string(dir.join(tokenized)).unwrap();
        let raw = raw.map(|raw| std::fs::read_to_string(dir.join(raw)).unwrap());
        let before_name = |line: &str| line.split(" @").next().unwrap().trim().to_owned();
        let pairs: Vec<_> = tokenized
            .lines()
            .enumerate()
            .map(|(index, line)| {
                let phrase = match &raw {
                    Some(raw) => before_name(raw.lines().nth(index).unwrap()),
                    None => line.rsplit_once('@').unwrap().1.trim().to_owned(),
                };
                (phrase, before_name(line))
            })
            .collect();
        assert!(!pairs.is_empty());
        pairs
    }

    fn tokenizer(dir: &Path) -> KeywordTokenizer {
        KeywordTokenizer::load(&dir.join("tokens.txt")).unwrap()
    }

    /// The spelling of `phrase` as space-separated tokens, or the error.
    fn spell(tokenizer: &KeywordTokenizer, phrase: &str) -> String {
        match tokenizer.tokenize(phrase) {
            Ok(tokens) => tokens.join(" "),
            Err(error) => error.to_string(),
        }
    }

    fn check_spellings(dir: &Path, unit: KeywordUnit, raw: Option<&str>, tokenized: &str) {
        let loaded = tokenizer(dir);
        assert_eq!(loaded.unit(), unit);
        for (phrase, expected) in examples(dir, raw, tokenized) {
            assert_eq!(spell(&loaded, &phrase), expected, "{phrase}");
        }
    }

    #[test]
    #[ignore = "needs the kws-en model"]
    fn english_spellings_match_text2token() {
        let Some(dir) = model_dir("kws-en") else {
            return;
        };
        check_spellings(
            &dir,
            KeywordUnit::Bpe,
            Some("keywords_raw.txt"),
            "keywords.txt",
        );
    }

    #[test]
    #[ignore = "needs the kws-zh model"]
    fn chinese_spellings_match_text2token() {
        let Some(dir) = model_dir("kws-zh") else {
            return;
        };
        check_spellings(&dir, KeywordUnit::Pinyin, None, "keywords.txt");
        check_spellings(
            &dir,
            KeywordUnit::Pinyin,
            None,
            "test_wavs/test_keywords.txt",
        );
    }

    #[test]
    #[ignore = "needs the kws-zh-en model"]
    fn mixed_spellings_match_text2token() {
        let Some(dir) = model_dir("kws-zh-en") else {
            return;
        };
        check_spellings(
            &dir,
            KeywordUnit::PhonePinyin,
            Some("test_wavs/keywords_raw.txt"),
            "test_wavs/keywords.txt",
        );
    }

    #[test]
    #[ignore = "needs the kws-zh and kws-zh-en models"]
    fn common_wake_phrases_are_spelled() {
        let (Some(zh), Some(zh_en)) = (model_dir("kws-zh"), model_dir("kws-zh-en")) else {
            return;
        };
        let mixed = tokenizer(&zh_en);
        for tokenizer in [&tokenizer(&zh), &mixed] {
            assert_eq!(
                spell(tokenizer, "你好，小爱同学"),
                "n ǐ h ǎo x iǎo ài t óng x ué"
            );
            assert_eq!(spell(tokenizer, "小艺小艺"), "x iǎo y ì x iǎo y ì");
        }
        assert_eq!(spell(&mixed, "Hi Jarvis"), "HH AY1 JH AA1 R V AH0 S");
        let unknown = spell(&mixed, "Hello Speechkit");
        assert!(
            unknown.contains("\"SPEECHKIT\" in the keyword"),
            "{unknown:?}"
        );
    }
}
