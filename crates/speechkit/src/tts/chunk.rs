//! Splitting text into chunks for synthesis.

use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

/// A piece of text to synthesize, with its byte range in all text pushed.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TextChunk {
    /// The text, including trailing whitespace.
    pub text: String,
    /// Its byte range in the whole input.
    pub range: Range<usize>,
}

impl TextChunk {
    /// Whether the chunk has anything to say.
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Splits text into sentence-sized chunks, incrementally.
///
/// - A chunk ends after `。！？；…` or a newline, or after `.`, `!`, or `?`
///   followed by whitespace or the end of input.
/// - `.` does not end a chunk in common abbreviations (`Mr.`, `Dr.`,
///   `e.g.`, `i.e.`, `U.S.`, single initials), decimals (`3.14`), or
///   ellipses (`...`).
/// - A chunk longer than the limit is split at its last comma (`，,、`),
///   then its last space, and only then at a character boundary. A split
///   never falls inside a grapheme cluster.
///
/// Concatenating the chunks gives back the input exactly.
#[derive(Debug, Clone)]
pub struct Chunker {
    max_chars: usize,
    buffer: String,
    offset: usize,
}

const STRONG_ENDS: &[char] = &['。', '！', '？', '；', '…', '\n'];
const CLOSERS: &[char] = &['"', '\'', '”', '’', '」', '』', '）', ')', ']', '》'];
const COMMAS: &[char] = &['，', ',', '、'];
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "u.s", "u.k", "a.m",
    "p.m", "no",
];

impl Chunker {
    /// A chunker for chunks of at most `max_chars` characters (at least 1).
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars: max_chars.max(1),
            buffer: String::new(),
            offset: 0,
        }
    }

    /// Adds text and returns the chunks it completes.
    pub fn push(&mut self, text: &str) -> Vec<TextChunk> {
        self.buffer.push_str(text);
        self.drain(false)
    }

    /// Ends the input and returns the remaining chunks.
    pub fn flush(&mut self) -> Vec<TextChunk> {
        self.drain(true)
    }

    /// Bytes of input already returned in chunks.
    pub fn consumed(&self) -> usize {
        self.offset
    }

    fn drain(&mut self, last: bool) -> Vec<TextChunk> {
        let mut out = Vec::new();
        loop {
            let end = match boundary(&self.buffer, last) {
                Some(end) => end,
                None if last && !self.buffer.is_empty() => self.buffer.len(),
                None if self.buffer.chars().count() > self.max_chars => self.buffer.len(),
                None => break,
            };
            // An over-long sentence gives up its first piece; the rest
            // stays buffered and is split on the next pass.
            let piece = &self.buffer[..end];
            let end = if piece.chars().count() > self.max_chars {
                split_point(piece, self.max_chars)
            } else {
                end
            };
            out.push(self.take(end));
        }
        out
    }

    fn take(&mut self, bytes: usize) -> TextChunk {
        let text: String = self.buffer.drain(..bytes).collect();
        let range = self.offset..self.offset + bytes;
        self.offset += bytes;
        TextChunk { text, range }
    }
}

/// The byte index just after the first sentence end, including closers
/// and whitespace after it; `None` if there is none yet (or, before the
/// end of input, if it cannot be decided yet).
fn boundary(text: &str, last: bool) -> Option<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    for (i, &(_, c)) in chars.iter().enumerate() {
        let ends = if STRONG_ENDS.contains(&c) {
            true
        } else if matches!(c, '.' | '!' | '?') {
            let mut j = i + 1;
            while j < chars.len()
                && (CLOSERS.contains(&chars[j].1) || matches!(chars[j].1, '!' | '?'))
            {
                j += 1;
            }
            match chars.get(j) {
                None if !last => return None,
                None => c != '.' || !is_abbreviation(&chars, i),
                Some(&(_, next)) => {
                    next.is_whitespace() && (c != '.' || !is_abbreviation(&chars, i))
                }
            }
        } else {
            false
        };
        if ends {
            let mut j = i + 1;
            while j < chars.len()
                && (CLOSERS.contains(&chars[j].1)
                    || matches!(chars[j].1, '.' | '!' | '?' | '！' | '？' | '。' | '…')
                    || chars[j].1.is_whitespace())
            {
                j += 1;
            }
            if j == chars.len() && !last && !STRONG_ENDS.contains(&c) {
                // More closers or whitespace may follow.
                return None;
            }
            return Some(chars.get(j).map_or(text.len(), |&(at, _)| at));
        }
    }
    None
}

/// Whether the `.` at `chars[i]` belongs to an abbreviation, an initial,
/// or an ellipsis.
fn is_abbreviation(chars: &[(usize, char)], i: usize) -> bool {
    let before = i.checked_sub(1).map(|k| chars[k].1);
    let after = chars.get(i + 1).map(|&(_, c)| c);
    if before == Some('.') || after == Some('.') {
        return true;
    }
    let start = chars[..i]
        .iter()
        .rposition(|&(_, c)| c.is_whitespace() || c == '(' || c == '"')
        .map_or(0, |k| k + 1);
    let word: String = chars[start..i].iter().map(|&(_, c)| c).collect();
    let lower = word.to_lowercase();
    ABBREVIATIONS.contains(&lower.as_str())
        || (word.chars().count() == 1 && word.chars().all(char::is_uppercase))
}

/// Where to split `text`, which is longer than `max` characters: after the
/// last comma in the first `max` characters, then after the last space,
/// then at the last grapheme boundary. Always at least one grapheme.
fn split_point(text: &str, max: usize) -> usize {
    let limit = text
        .char_indices()
        .nth(max)
        .map_or(text.len(), |(at, _)| at);
    let window = &text[..limit];
    if let Some((at, c)) = window.char_indices().rfind(|&(_, c)| COMMAS.contains(&c)) {
        return at + c.len_utf8();
    }
    if let Some((at, c)) = window.char_indices().rfind(|&(_, c)| c.is_whitespace())
        && at > 0
    {
        return at + c.len_utf8();
    }
    let cut = text
        .grapheme_indices(true)
        .map(|(at, _)| at)
        .take_while(|&at| at <= limit)
        .last()
        .unwrap_or(0);
    if cut > 0 {
        cut
    } else {
        text.graphemes(true).next().map_or(text.len(), str::len)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn split(text: &str, max: usize) -> Vec<String> {
        let mut chunker = Chunker::new(max);
        let mut out: Vec<String> = chunker.push(text).into_iter().map(|c| c.text).collect();
        out.extend(chunker.flush().into_iter().map(|c| c.text));
        out
    }

    #[test]
    #[expect(clippy::too_many_lines, reason = "one row per chunking case")]
    fn table() {
        let cases: &[(&str, usize, &[&str])] = &[
            ("你好。世界。", 300, &["你好。", "世界。"]),
            (
                "你好！你是谁？我很好；谢谢…",
                300,
                &["你好！", "你是谁？", "我很好；", "谢谢…"],
            ),
            (
                "Hello world. How are you?",
                300,
                &["Hello world. ", "How are you?"],
            ),
            (
                "Hello world.How are you?",
                300,
                &["Hello world.How are you?"],
            ),
            ("Wait! Really? Yes.", 300, &["Wait! ", "Really? ", "Yes."]),
            ("What?! No.", 300, &["What?! ", "No."]),
            (
                "Mr. Smith went home. Dr. Who came.",
                300,
                &["Mr. Smith went home. ", "Dr. Who came."],
            ),
            (
                "Use e.g. apples, i.e. fruit. Then eat.",
                300,
                &["Use e.g. apples, i.e. fruit. ", "Then eat."],
            ),
            (
                "The U.S. economy grew. It rained.",
                300,
                &["The U.S. economy grew. ", "It rained."],
            ),
            (
                "J. R. R. Tolkien wrote. The end.",
                300,
                &["J. R. R. Tolkien wrote. ", "The end."],
            ),
            (
                "Pi is 3.14 roughly. OK.",
                300,
                &["Pi is 3.14 roughly. ", "OK."],
            ),
            ("Wait... what? Fine.", 300, &["Wait... what? ", "Fine."]),
            (
                "He said \"stop.\" Then left.",
                300,
                &["He said \"stop.\" ", "Then left."],
            ),
            (
                "她说：“好。”然后走了。",
                300,
                &["她说：“好。”", "然后走了。"],
            ),
            ("line one\nline two", 300, &["line one\n", "line two"]),
            (
                "我用Rust写代码。Rust很快.",
                300,
                &["我用Rust写代码。", "Rust很快."],
            ),
            ("", 300, &[]),
            ("   ", 300, &["   "]),
            ("no ending", 300, &["no ending"]),
            (
                "Trailing spaces.   Next.",
                300,
                &["Trailing spaces.   ", "Next."],
            ),
            ("a, b, c, d", 5, &["a, b,", " c, d"]),
            ("one two three four", 9, &["one two ", "three ", "four"]),
            ("一二三四五六七八", 3, &["一二三", "四五六", "七八"]),
            ("甲，乙，丙丁戊己", 4, &["甲，乙，", "丙丁戊己"]),
            ("abcdefgh", 3, &["abc", "def", "gh"]),
            (
                "Hello, world, again. Bye.",
                8,
                &["Hello,", " world,", " again. ", "Bye."],
            ),
            (
                "e\u{301}e\u{301}e\u{301}",
                2,
                &["e\u{301}", "e\u{301}", "e\u{301}"],
            ),
            ("👍🏽👍🏽", 1, &["👍🏽", "👍🏽"]),
            ("Stop.", 300, &["Stop."]),
            (
                "第一句。第二句，很长很长。",
                5,
                &["第一句。", "第二句，", "很长很长。"],
            ),
            ("A. B.", 300, &["A. B."]),
            ("ok. no. 5 items.", 300, &["ok. ", "no. 5 items."]),
            ("Hi (Mr. X). Bye.", 300, &["Hi (Mr. X). ", "Bye."]),
            ("…！", 300, &["…！"]),
            ("你好,world. 再见。", 300, &["你好,world. ", "再见。"]),
            ("Tab\tend.\tNext.", 300, &["Tab\tend.\t", "Next."]),
            ("Ends with question?", 300, &["Ends with question?"]),
            ("Q? A!", 300, &["Q? ", "A!"]),
            ("数字3.5和4.2。", 300, &["数字3.5和4.2。"]),
            ("a.b.c. Next.", 300, &["a.b.c. ", "Next."]),
        ];
        for (text, max, want) in cases {
            assert_eq!(split(text, *max), *want, "{text:?} max {max}");
        }
    }

    #[test]
    fn incremental_waits_for_what_follows_a_period() {
        let mut chunker = Chunker::new(300);
        assert!(chunker.push("Hello world.").is_empty());
        let chunks = chunker.push(" Next");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world. ");
        assert_eq!(chunks[0].range, 0..13);
        assert_eq!(chunker.consumed(), 13);
        assert_eq!(chunker.push("你好。")[0].text, "Next你好。");
        let rest = chunker.flush();
        assert!(rest.is_empty());
        assert!(
            TextChunk {
                text: " ".into(),
                range: 0..1
            }
            .is_blank()
        );
    }

    proptest! {
        #[test]
        fn chunks_rebuild_the_input_within_the_limit(
            parts in prop::collection::vec("[a-z ,.!?。，、…\\n你好e\u{301}]{0,12}", 0..12),
            max in 1_usize..20,
        ) {
            let mut chunker = Chunker::new(max);
            let mut chunks = Vec::new();
            for part in &parts {
                chunks.extend(chunker.push(part));
            }
            chunks.extend(chunker.flush());
            let input: String = parts.concat();
            let rebuilt: String = chunks.iter().map(|c| c.text.as_str()).collect();
            prop_assert_eq!(&rebuilt, &input);
            let mut offset = 0;
            for chunk in &chunks {
                prop_assert_eq!(chunk.range.start, offset);
                prop_assert_eq!(&input[chunk.range.clone()], chunk.text.as_str());
                offset = chunk.range.end;
                let graphemes = chunk.text.graphemes(true).count();
                prop_assert!(chunk.text.chars().count() <= max || graphemes == 1, "{:?} over {}", chunk.text, max);
            }
        }
    }
}
