//! Character error rate, for round-trip tests.

/// Lowercases `text` and keeps only letters and digits, so punctuation,
/// spacing, and case do not count as errors.
pub fn normalize(text: &str) -> Vec<char> {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// The edit distance between `reference` and `hypothesis`, divided by the
/// reference length, after [`normalize`]. An empty reference gives 0.0
/// for an empty hypothesis and 1.0 otherwise.
pub fn cer(reference: &str, hypothesis: &str) -> f64 {
    let reference = normalize(reference);
    let hypothesis = normalize(hypothesis);
    if reference.is_empty() {
        return if hypothesis.is_empty() { 0.0 } else { 1.0 };
    }
    let mut previous: Vec<usize> = (0..=hypothesis.len()).collect();
    let mut current = vec![0; hypothesis.len() + 1];
    for (i, r) in reference.iter().enumerate() {
        current[0] = i + 1;
        for (j, h) in hypothesis.iter().enumerate() {
            let substitution = previous[j] + usize::from(r != h);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
        }
        std::mem::swap(&mut previous, &mut current);
    }
    #[expect(clippy::cast_precision_loss, reason = "texts are short")]
    let rate = previous[hypothesis.len()] as f64 / reference.len() as f64;
    rate
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
}
