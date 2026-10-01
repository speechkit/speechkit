//! Printing transcripts.

use std::{io::Write, time::Duration};

use serde_json::json;
use speechkit::asr::Transcript;

use crate::OutputFormat;

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// `HH:MM:SS,mmm`.
pub(crate) fn srt_time(duration: Duration) -> String {
    let ms = millis(duration);
    format!(
        "{:02}:{:02}:{:02},{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1_000 % 60,
        ms % 1_000
    )
}

/// `MM:SS.mm`, for progress lines.
pub(crate) fn short_time(duration: Duration) -> String {
    let cs = millis(duration) / 10;
    format!("{:02}:{:02}.{:02}", cs / 6_000, cs / 100 % 60, cs % 100)
}

pub(crate) fn print(
    transcript: &Transcript,
    format: OutputFormat,
    out: &mut dyn Write,
) -> std::io::Result<()> {
    match format {
        OutputFormat::Text => writeln!(out, "{}", transcript.text()),
        OutputFormat::Json => {
            let segments: Vec<_> = transcript
                .segments
                .iter()
                .map(|s| {
                    json!({
                        "text": s.text,
                        "start_ms": millis(s.start),
                        "end_ms": millis(s.end),
                    })
                })
                .collect();
            let value = json!({ "text": transcript.text(), "segments": segments });
            writeln!(out, "{value}")
        }
        OutputFormat::Srt => {
            let segments = transcript
                .segments
                .iter()
                .filter(|s| !s.text.trim().is_empty());
            for (index, segment) in segments.enumerate() {
                writeln!(
                    out,
                    "{}\n{} --> {}\n{}\n",
                    index + 1,
                    srt_time(segment.start),
                    srt_time(segment.end),
                    segment.text.trim()
                )?;
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use speechkit::asr::{Segment, UtteranceId};

    use super::*;

    fn transcript() -> Transcript {
        Transcript::new(
            vec![
                Segment {
                    utterance: UtteranceId(0),
                    text: "hello".into(),
                    start: Duration::from_millis(120),
                    end: Duration::from_millis(3_723_456),
                },
                Segment {
                    utterance: UtteranceId(1),
                    text: " ".into(),
                    start: Duration::ZERO,
                    end: Duration::ZERO,
                },
            ],
            Duration::from_secs(4),
        )
    }

    #[test]
    fn formats() {
        let mut text = Vec::new();
        print(&transcript(), OutputFormat::Text, &mut text).unwrap();
        assert_eq!(text, b"hello\n");
        let mut json = Vec::new();
        print(&transcript(), OutputFormat::Json, &mut json).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&json).unwrap();
        assert_eq!(value["segments"][0]["end_ms"], 3_723_456);
        let mut srt = Vec::new();
        print(&transcript(), OutputFormat::Srt, &mut srt).unwrap();
        assert_eq!(
            String::from_utf8(srt).unwrap(),
            "1\n00:00:00,120 --> 01:02:03,456\nhello\n\n"
        );
        assert_eq!(short_time(Duration::from_millis(61_230)), "01:01.23");
    }
}
