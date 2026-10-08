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
//! cargo test --workspace --all-features --test sherpa_eval -- --ignored --nocapture
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
//! A decode or recognition failure scores as an empty hypothesis so the
//! run goes on; once every model has reported, the test fails if any
//! utterance failed, or if the corpus lacks test speakers. It never
//! fails on a rate.
//! Streaming backends are fed each utterance as one buffered push,
//! not at real-time pace, so their endpointing differs from
//! production: read their CER as a fast-push control, not a deployed
//! figure. Real-time pacing is future work (docs/eval.md).
#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "test helpers fail the calling test"
)]

use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
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

/// The number of speakers in AISHELL-1's test split.
const TEST_SPEAKERS: usize = 20;

/// One utterance to score.
struct Case {
    id: String,
    speaker: String,
    wav: PathBuf,
    reference: String,
    /// The length of `reference` after normalization.
    reference_len: usize,
}

/// The entries of `dir`, sorted.
fn sorted_entries(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}; rerun `cargo xtask fetch-evals`", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    paths.sort();
    paths
}

/// The evaluation subset: the first transcribed wavs of every test
/// speaker on disk, in utterance order. Driven by the wavs, not the
/// transcript (which lists train and dev too), so a speaker missing on
/// disk fails here instead of shrinking the run.
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
    let references: HashMap<&str, &str> = transcript
        .lines()
        .filter_map(|line| line.split_once(' '))
        .collect();
    let speakers = sorted_entries(&root.join("wav/test"));
    assert_eq!(
        speakers.len(),
        TEST_SPEAKERS,
        "{}: not every test speaker is on disk; delete it and rerun `cargo xtask fetch-evals`",
        root.display()
    );
    let mut cases = Vec::new();
    let mut untranscribed = 0_usize;
    for dir in speakers {
        let speaker = dir.file_name().unwrap().to_string_lossy().into_owned();
        let wavs: Vec<_> = sorted_entries(&dir)
            .into_iter()
            .filter(|path| path.extension().is_some_and(|ext| ext == "wav"))
            .collect();
        assert!(!wavs.is_empty(), "{}: no wavs", dir.display());
        let mut taken = 0;
        for wav in wavs {
            if taken == per_speaker {
                break;
            }
            let id = wav.file_stem().unwrap().to_string_lossy().into_owned();
            let Some(reference) = references.get(id.as_str()) else {
                untranscribed += 1;
                continue;
            };
            taken += 1;
            cases.push(Case {
                id,
                speaker: speaker.clone(),
                wav,
                reference: (*reference).to_string(),
                reference_len: normalize(reference).len(),
            });
        }
    }
    if untranscribed > 0 {
        eprintln!("skipped {untranscribed} wavs with no transcript line");
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
/// Scores `engine` on `cases`, prints the report, and returns the
/// number of utterances that failed to decode or transcribe.
fn evaluate(id: &str, engine: &AsrEngine, cases: &[Case]) -> usize {
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
        reference_chars += case.reference_len;
        totals.substitutions += scored.substitutions;
        totals.deletions += scored.deletions;
        totals.insertions += scored.insertions;
        let speaker_score = per_speaker.entry(&case.speaker).or_default();
        speaker_score.0 += scored.total();
        speaker_score.1 += case.reference_len;
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
    failures
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
    let mut failed = Vec::new();
    for (id, family) in lineup {
        let Some(engine) = engine(id, family) else {
            continue;
        };
        let failures = evaluate(id, &engine, &cases);
        if failures > 0 {
            failed.push(format!("{id}: {failures}/{}", cases.len()));
        }
    }
    assert!(
        failed.is_empty(),
        "utterances failed and were scored as empty: {failed:?}"
    );
}
