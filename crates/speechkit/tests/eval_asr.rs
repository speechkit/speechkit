#![cfg(feature = "sherpa")]
//! Recognition accuracy on the AISHELL-1 test split, against the human
//! transcripts, through the full `AsrEngine` pipeline. This is the
//! evaluation `docs/eval.md` describes; the golden tests in
//! `sherpa_models.rs` only catch change on the models' own clips.
//!
//! Fetch the corpus and models first, then run with output shown:
//!
//! ```sh
//! cargo xtask fetch-evals
//! cargo xtask fetch-fixtures --only silero-vad,sense-voice,paraformer-zh,transducer-zh,streaming-bilingual
//! cargo test -p speechkit --all-features --test eval_asr -- --ignored --nocapture
//! ```
//!
//! The large tier (funasr-nano, firered-aed, firered-ctc, qwen3-asr)
//! is not in the manifest; point `SPEECHKIT_MODEL_<ID>` at its
//! directory to include it, as the large-model tests in
//! `sherpa_models.rs` do. `docs/eval.md` lists where each archive
//! comes from.
//!
//! Every model whose `SPEECHKIT_MODEL_<ID>` is set is evaluated on the
//! first `SPEECHKIT_EVAL_UTTERANCES` utterances of each test speaker
//! (15 by default; raise it toward 359 for the full split when a number
//! matters). The subset is deterministic, so runs are comparable. The
//! report prints the corpus CER with its substitution, deletion, and
//! insertion shares, the per-speaker CERs, and the worst utterances.
//! The test fails only on a session error, never on a rate.
//! Streaming backends are fed each utterance as one buffered push,
//! not at real-time pace, so their endpointing differs from
//! production: read their CER as a fast-push control, not a deployed
//! figure. Real-time pacing is future work (docs/eval.md).
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

use speechkit::{
    asr::{AsrEngine, AsrOptions},
    audio::{self, DecodeLimits},
    sherpa::{AsrConfig, AsrFamily},
};
use speechkit_testkit::{
    gates::{eval_dir, model_dir},
    metrics::{Errors, char_errors, normalize},
    secs,
};

/// One utterance to score: its ID, its wav, and its transcript line.
struct Case {
    id: String,
    wav: PathBuf,
    reference: String,
}

/// The evaluation subset: the first utterances of every test speaker,
/// in utterance order, skipping wavs missing on disk.
fn cases() -> Option<Vec<Case>> {
    let per_speaker = match std::env::var("SPEECHKIT_EVAL_UTTERANCES") {
        Err(_) => 15,
        Ok(value) => value
            .parse()
            .expect("SPEECHKIT_EVAL_UTTERANCES must be a number"),
    };
    let root = eval_dir("aishell-1")?;
    let transcript = std::fs::read_to_string(root.join("transcript/aishell_transcript_v0.8.txt"))
        .expect("the corpus entry lists the transcript, so it is present");
    let mut by_speaker: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
    for line in transcript.lines() {
        let Some((id, reference)) = line.split_once(' ') else {
            continue;
        };
        // BAC009S0764W0121: corpus code, speaker, utterance.
        assert!(id.starts_with("BAC009") && id.len() == 16, "{id}");
        let speaker = id[6..11].to_string();
        by_speaker
            .entry(speaker)
            .or_default()
            .push((id.to_string(), reference.to_string()));
    }
    let mut cases = Vec::new();
    for (speaker, mut utterances) in by_speaker {
        utterances.sort_by(|a, b| a.0.cmp(&b.0));
        for (id, reference) in utterances.into_iter().take(per_speaker) {
            let wav = root.join(format!("wav/test/{speaker}/{id}.wav"));
            if wav.exists() {
                cases.push(Case { id, wav, reference });
            }
        }
    }
    Some(cases)
}

/// The engine for `id`, or `None` (with a note) when it is not fetched.
/// Offline backends run behind the silero VAD; streaming ones alone.
/// The family is named because a directory layout can match more than
/// one (a Paraformer directory also parses as SenseVoice).
fn engine(id: &str, family: AsrFamily) -> Option<AsrEngine> {
    let dir = model_dir(id)?;
    let backend = if family == AsrFamily::StreamingTransducer {
        AsrConfig::streaming(&dir).load().unwrap()
    } else {
        let Some(vad) = model_dir("silero-vad") else {
            eprintln!("[{id}] skipped: offline backends run behind silero-vad");
            return None;
        };
        AsrConfig::offline(&dir, &vad)
            .with_family(family)
            .load()
            .unwrap()
    };
    Some(AsrEngine::new(backend))
}

/// The transcript of one wav: the product path a caller gets, including
/// VAD segmentation and endpointing. A backend failure is reported, not
/// panicked on, so one bad utterance cannot end a model's evaluation.
fn transcribe(engine: &AsrEngine, audio: &speechkit::AudioBuffer) -> Result<String, String> {
    engine
        .transcribe(audio, AsrOptions::default(), secs(600))
        .map(|outcome| outcome.text())
        .map_err(|error| error.to_string())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "statistics over thousands of characters"
)]
fn evaluate(id: &str, engine: &AsrEngine, cases: &[Case]) {
    let started = Instant::now();
    let mut totals = Errors::default();
    let mut reference_chars = 0_usize;
    let mut audio_time = Duration::ZERO;
    let mut empty = 0_usize;
    let mut failures = 0_usize;
    let mut per_speaker: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut worst: Vec<(usize, &Case, String)> = Vec::new();
    for (n, case) in cases.iter().enumerate() {
        // A failed decode or transcription scores as an empty
        // hypothesis: every reference character becomes a deletion.
        // Only a successful but empty transcript counts as `empty`,
        // so the two reported counts stay disjoint.
        let hypothesis = match audio::read(&case.wav, DecodeLimits::default())
            .map_err(|e| e.to_string())
            .and_then(|audio| {
                audio_time += audio.sample_rate.duration_of(audio.samples.len() as u64);
                transcribe(engine, &audio)
            }) {
            Ok(text) => {
                if text.trim().is_empty() {
                    empty += 1;
                }
                text
            }
            Err(error) => {
                failures += 1;
                if failures <= 3 {
                    eprintln!("[{id}] {}: {error}", case.id);
                }
                String::new()
            }
        };
        let scored = char_errors(&case.reference, &hypothesis);
        let reference_len = normalize(&case.reference).len();
        reference_chars += reference_len;
        totals.substitutions += scored.substitutions;
        totals.deletions += scored.deletions;
        totals.insertions += scored.insertions;
        let speaker_score = per_speaker.entry(&case.id[6..11]).or_default();
        speaker_score.0 += scored.total();
        speaker_score.1 += reference_len;
        worst.push((scored.total(), case, hypothesis));
        if (n + 1) % 100 == 0 {
            eprintln!("[{id}] {}/{} utterances", n + 1, cases.len());
        }
    }
    let rate = |edits: usize| 100.0 * edits as f64 / reference_chars as f64;
    let wall = started.elapsed().as_secs_f64();
    let audio_secs = audio_time.as_secs_f64();
    eprintln!("\n=== {id} ===");
    eprintln!(
        "{} utterances, {} speakers, {:.1} min audio in {:.1} min (RTF {:.2})",
        cases.len(),
        per_speaker.len(),
        audio_secs / 60.0,
        wall / 60.0,
        wall / audio_secs.max(1.0)
    );
    eprintln!(
        "CER {:.2}%  (sub {:.2} del {:.2} ins {:.2} as % of reference chars; {} empty, {} failed)",
        rate(totals.total()),
        rate(totals.substitutions),
        rate(totals.deletions),
        rate(totals.insertions),
        empty,
        failures
    );
    let speakers = per_speaker
        .iter()
        .map(|(speaker, (edits, chars))| {
            #[expect(clippy::cast_precision_loss, reason = "per-speaker statistics")]
            let cer = 100.0 * *edits as f64 / *chars as f64;
            format!("{speaker} {cer:.1}")
        })
        .collect::<Vec<_>>()
        .join("  ");
    eprintln!("per speaker CER %: {speakers}");
    worst.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.0));
    for (edits, case, hypothesis) in worst.iter().take(3) {
        eprintln!(
            "worst ({edits} edits) {}: {hypothesis} | want: {}",
            case.id, case.reference
        );
    }
}

#[test]
#[ignore = "needs the AISHELL-1 corpus and models; see the module docs"]
fn aishell1_accuracy() {
    let Some(cases) = cases() else {
        return;
    };
    assert!(!cases.is_empty(), "corpus present but no cases built");
    // Small published models first, then the large tier; every entry
    // whose directory is set runs, the rest print a skip note.
    let lineup: [(&str, AsrFamily); 8] = [
        ("sense-voice", AsrFamily::SenseVoice),
        ("paraformer-zh", AsrFamily::Paraformer),
        ("transducer-zh", AsrFamily::OfflineTransducer),
        ("streaming-bilingual", AsrFamily::StreamingTransducer),
        ("funasr-nano", AsrFamily::FunAsrNano),
        ("firered-aed", AsrFamily::FireRedAed),
        ("firered-ctc", AsrFamily::FireRedCtc),
        ("qwen3-asr", AsrFamily::Qwen3Asr),
    ];
    for (id, family) in lineup {
        let Some(engine) = engine(id, family) else {
            continue;
        };
        evaluate(id, &engine, &cases);
    }
}
