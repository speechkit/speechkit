//! Error-rate metrics for ASR accuracy evaluation.
//!
//! Two rates, both an edit distance divided by the reference length:
//! [`cer`] over normalized characters (Chinese, and mixed
//! Chinese-English text) and [`wer`] over normalized words
//! (English-only text; Chinese is not space-separated, so on Chinese
//! input `wer` scores whole utterances as single words and is
//! meaningless). [`Errors`] breaks a distance down into
//! substitutions, deletions, and insertions.
//!
//! # Normalization policy
//!
//! Both rates fold case and drop every character that is not a
//! letter or a digit, so punctuation, spacing, and symbols never
//! count as errors. Backends differ on punctuation and output
//! spacing arbitrarily, and the rates must not measure that. Two
//! consequences when reading numbers:
//!
//! - Digits compare as written: "123" and "一百二十三" are wholly
//!   different to [`cer`]. No inverse text normalization is applied;
//!   normalize the references before scoring if a corpus spells
//!   numbers out and the backend does not.
//! - Full-width and half-width alphanumerics stay distinct after
//!   case folding ("ａ" is not "a"). Sherpa backends emit half-width,
//!   and the corpora listed in `fixtures/evals.json` use half-width
//!   transcripts, so this only matters for hand-written references.
//!
//! # Reading the breakdown
//!
//! [`Errors`] says where accuracy was lost: deletions clustered at
//! the end of utterances point at endpointing or VAD cutting speech
//! off early, insertions at utterance boundaries point at VAD
//! merging two utterances, and substitutions spread across the
//! utterance are acoustic or model errors. Compare `Errors` across
//! backends to see how their failures differ, not only how large
//! they are.

/// Where the edits of an alignment fell: reference units replaced by
/// a different hypothesis unit ([`Errors::substitutions`]), reference
/// units with no counterpart ([`Errors::deletions`]), and hypothesis
/// units with no counterpart ([`Errors::insertions`]). A unit is a
/// character for [`char_errors`] and a word for [`word_errors`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Errors {
    /// Reference units replaced by a different hypothesis unit.
    pub substitutions: usize,
    /// Reference units with no counterpart in the hypothesis.
    pub deletions: usize,
    /// Hypothesis units with no counterpart in the reference.
    pub insertions: usize,
}

impl Errors {
    /// The three counts summed.
    pub fn total(&self) -> usize {
        self.substitutions + self.deletions + self.insertions
    }

    /// The error rate over a reference of `reference_len` units. An
    /// empty reference scores 0.0 for an empty hypothesis and 1.0
    /// otherwise.
    pub fn rate(&self, reference_len: usize) -> f64 {
        if reference_len == 0 {
            return if self.total() == 0 { 0.0 } else { 1.0 };
        }
        #[expect(clippy::cast_precision_loss, reason = "texts are short")]
        let rate = self.total() as f64 / reference_len as f64;
        rate
    }
}

/// Lowercases `text` and keeps only letters and digits, so
/// punctuation, spacing, and case do not count as errors.
pub fn normalize(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// [`normalize`] as words: every run of letters and digits between
/// dropped characters is one word.
pub fn normalize_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    // Filter before lowercasing, as `normalize` does: a lowercase form
    // can hold a combining mark ("İ" is "i" and U+0307) that must not
    // split the word.
    for c in text.chars() {
        if c.is_alphanumeric() {
            current.extend(c.to_lowercase());
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// The edits of the cheapest alignment of `hypothesis` to
/// `reference`. Ties prefer substitution over deletion over
/// insertion, so the counts are deterministic. O(n·m) time and
/// space; utterance-sized texts only.
fn align<T: PartialEq>(reference: &[T], hypothesis: &[T]) -> Errors {
    let mut costs = vec![vec![0_usize; hypothesis.len() + 1]; reference.len() + 1];
    for (i, row) in costs.iter_mut().enumerate().skip(1) {
        row[0] = i;
    }
    for (j, cost) in costs[0].iter_mut().enumerate().skip(1) {
        *cost = j;
    }
    for (i, r) in reference.iter().enumerate() {
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = costs[i][j] + usize::from(r != h);
            costs[i + 1][j + 1] = substitution
                .min(costs[i][j + 1] + 1)
                .min(costs[i + 1][j] + 1);
        }
    }
    let mut errors = Errors::default();
    let (mut i, mut j) = (reference.len(), hypothesis.len());
    while i > 0 || j > 0 {
        if i > 0
            && j > 0
            && costs[i][j]
                == costs[i - 1][j - 1] + usize::from(reference[i - 1] != hypothesis[j - 1])
        {
            errors.substitutions += usize::from(reference[i - 1] != hypothesis[j - 1]);
            i -= 1;
            j -= 1;
        } else if i > 0 && costs[i][j] == costs[i - 1][j] + 1 {
            errors.deletions += 1;
            i -= 1;
        } else {
            errors.insertions += 1;
            j -= 1;
        }
    }
    errors
}

/// The character-level [`Errors`] between `reference` and
/// `hypothesis`, after [`normalize`].
///
/// # Examples
///
/// ```
/// use speechkit_testkit::metrics::char_errors;
///
/// let e = char_errors("今天天气很好。", "今天天汽很好");
/// assert_eq!((e.substitutions, e.deletions, e.insertions), (1, 0, 0));
/// ```
pub fn char_errors(reference: &str, hypothesis: &str) -> Errors {
    align(&normalize(reference), &normalize(hypothesis))
}

/// The word-level [`Errors`] between `reference` and `hypothesis`,
/// after [`normalize_words`].
///
/// # Examples
///
/// ```
/// use speechkit_testkit::metrics::word_errors;
///
/// let e = word_errors("the cat sat", "cat sat on");
/// assert_eq!((e.substitutions, e.deletions, e.insertions), (0, 1, 1));
/// ```
pub fn word_errors(reference: &str, hypothesis: &str) -> Errors {
    align(&normalize_words(reference), &normalize_words(hypothesis))
}

/// The edit distance between `reference` and `hypothesis`, divided
/// by the reference length, after [`normalize`]. An empty reference
/// gives 0.0 for an empty hypothesis and 1.0 otherwise.
///
/// # Examples
///
/// ```
/// use speechkit_testkit::metrics::cer;
///
/// assert!((cer("abcd", "abxd") - 0.25).abs() < f64::EPSILON);
/// ```
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize(reference);
    let hypothesis = normalize(hypothesis);
    align(&reference, &hypothesis).rate(reference.len())
}

/// The word error rate between `reference` and `hypothesis`:
/// [`word_errors`] over the reference word count. An empty reference
/// gives 0.0 for an empty hypothesis and 1.0 otherwise. For Chinese,
/// use [`cer`].
///
/// # Examples
///
/// ```
/// use speechkit_testkit::metrics::wer;
///
/// assert!((wer("the cat sat", "cat sat on") - 2.0 / 3.0).abs() < f64::EPSILON);
/// ```
pub fn wer(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize_words(reference);
    let hypothesis = normalize_words(hypothesis);
    align(&reference, &hypothesis).rate(reference.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates() {
        assert!(cer("Hello, world!", "hello world").abs() < f64::EPSILON);
        assert!((cer("abcd", "abxd") - 0.25).abs() < f64::EPSILON);
        assert!((cer("abcd", "") - 1.0).abs() < f64::EPSILON);
        assert!((cer("ab", "abcd") - 1.0).abs() < f64::EPSILON);
        assert!((cer("今天天气很好。", "今天天汽很好") - 1.0 / 6.0).abs() < 1e-9);
        assert!(cer("", "").abs() < f64::EPSILON);
        assert!((cer("", "x") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn breakdowns() {
        let e = char_errors("abcd", "abxd");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (1, 0, 0));
        let e = char_errors("abcd", "abd");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (0, 1, 0));
        let e = char_errors("abd", "abcd");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (0, 0, 1));
        let e = char_errors("", "xyz");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (0, 0, 3));
        let e = char_errors("xyz", "");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (0, 3, 0));
        // Substitution plus deletions: e has no counterpart.
        let e = char_errors("abcde", "axc");
        assert_eq!((e.substitutions, e.deletions, e.insertions), (1, 2, 0));
    }

    #[test]
    fn word_normalization() {
        assert_eq!(normalize_words("Hello, world!"), ["hello", "world"]);
        assert_eq!(normalize_words("今天天气 很好"), ["今天天气", "很好"]);
        assert_eq!(normalize_words("don't stop"), ["don", "t", "stop"]);
        assert_eq!(normalize_words(""), Vec::<String>::new());
        assert_eq!(normalize_words("İstanbul"), ["i\u{307}stanbul"]);
        assert_eq!(
            normalize_words("İstanbul")
                .concat()
                .chars()
                .collect::<Vec<_>>(),
            normalize("İstanbul")
        );
    }

    #[test]
    fn word_rates() {
        assert!((wer("the cat sat", "the cat sat") - 0.0).abs() < f64::EPSILON);
        assert!((wer("the cat sat", "the cat sat on") - 1.0 / 3.0).abs() < f64::EPSILON);
        assert!(wer("", "").abs() < f64::EPSILON);
        assert!((wer("", "x") - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rates_match_their_breakdowns() {
        let (reference, hypothesis) = ("嗯今天weather不错", "今天天气不错");
        let e = char_errors(reference, hypothesis);
        assert!((e.rate(normalize(reference).len()) - cer(reference, hypothesis)).abs() < 1e-12);
        let e = word_errors(reference, hypothesis);
        assert!(
            (e.rate(normalize_words(reference).len()) - wer(reference, hypothesis)).abs() < 1e-12
        );
    }
}
