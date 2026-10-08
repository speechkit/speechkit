//! Tests against real models. They are ignored by default; run them with
//! `cargo test -p speechkit --features sherpa -- --ignored` after
//! `cargo xtask fetch-fixtures`, which sets `SPEECHKIT_MODEL_*`.
//!
//! - Goldens: each `fixtures/golden/asr/<platform>/<backend>/*.json`
//!   records the reference output for one sample on that platform. The text
//!   of the transcript and of each segment must be within `MAX_CER` of the
//!   golden, segment boundaries within 20 ms, and for streaming models at
//!   least `MIN_PARTIALS_IN_ORDER` of the golden partials must appear in
//!   order. Exact matches only hold on the CPU that recorded the goldens:
//!   ONNX Runtime picks its kernels by CPU (AVX2 or AVX-512), which flips
//!   near-ties on other runners of the same platform, so any mismatch is
//!   printed but only a larger one fails. Only 16 kHz clips have goldens,
//!   because resampling is tested on its own. Streaming goldens have no
//!   segment times, so only the text is compared. The goldens are fixed:
//!   for a new platform or a deliberate change in output, edit the JSON
//!   files by hand from a reviewed run of these tests.
//! - The ASR contract on real backends.
#![cfg(feature = "sherpa")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test helpers fail the calling test"
)]

use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use serde::Deserialize;
use speechkit::sherpa::{AsrConfig, AsrFamily, PunctuationConfig};
use speechkit::{
    AudioBuffer,
    asr::{AsrBackend, AsrEngine, AsrEvent, AsrOptions},
    audio::{self, DecodeLimits},
};
use speechkit_testkit::{
    asr::Collected,
    contract::asr::{check_activity, run_asr_contract},
    gates::{large_model_tier, model_dir},
    metrics::cer,
    secs,
};

fn streaming(id: &str) -> Option<AsrEngine> {
    let backend = AsrConfig::streaming(model_dir(id)?).load().unwrap();
    Some(AsrEngine::new(backend))
}

fn sense_voice() -> Option<AsrEngine> {
    let config = AsrConfig::offline(model_dir("sense-voice")?, model_dir("silero-vad")?);
    let backend = config.load().unwrap();
    assert_eq!(backend.family(), AsrFamily::SenseVoice);
    Some(AsrEngine::new(backend))
}

#[derive(Deserialize)]
struct Golden {
    /// The model whose `test_wavs` holds the sample, e.g. `streaming-en`.
    model: String,
    /// The sample, relative to the model directory.
    audio: String,
    text: String,
    segments: Vec<GoldenSegment>,
    #[serde(default)]
    partials: Vec<String>,
}

#[derive(Deserialize)]
struct GoldenSegment {
    text: String,
    /// Null for streaming models, whose goldens have no times.
    start_ms: Option<u64>,
    end_ms: Option<u64>,
}

/// `<os>-<arch>`. Goldens are recorded per platform: sherpa-onnx's native
/// math differs slightly between them, enough to change a streaming
/// model's text.
fn platform() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

fn goldens(backend: &str) -> Vec<(PathBuf, Golden)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/golden/asr")
        .join(platform())
        .join(backend);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skipped: no goldens in {}", dir.display());
        return Vec::new();
    };
    let mut goldens: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .map(|path| {
            let golden = serde_json::from_str(&std::fs::read_to_string(&path).unwrap())
                .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            (path, golden)
        })
        .collect();
    goldens.sort_by(|a, b| a.0.cmp(&b.0));
    goldens
}

/// The character error rate allowed against a golden text. A flipped
/// near-tie changes one or two characters of a clip; a broken backend
/// changes far more.
const MAX_CER: f64 = 0.1;

/// The share of a streaming golden's partial results that must appear, in
/// order, among the partials produced. A near-tie can move a word to the
/// next partial, which changes one entry of the sequence.
const MIN_PARTIALS_IN_ORDER: f64 = 0.9;

/// Checks `got` against the golden `want`: any difference is printed, and
/// a character error rate above `MAX_CER` fails.
fn check_text(name: &str, what: &str, got: &str, want: &str) {
    if got == want {
        return;
    }
    let rate = cer(want, got);
    eprintln!(
        "{name}: {what} differs from the golden (CER {rate:.3})\n  got:  {got}\n  want: {want}"
    );
    assert!(
        rate <= MAX_CER,
        "{name}: {what} CER {rate:.3} exceeds {MAX_CER}\n  got:  {got}\n  want: {want}"
    );
}

/// The length of the longest common subsequence of `a` and `b`.
fn common_in_order(a: &[String], b: &[String]) -> usize {
    let mut previous = vec![0; b.len() + 1];
    for x in a {
        let mut current = vec![0; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            current[j + 1] = if x == y {
                previous[j] + 1
            } else {
                previous[j + 1].max(current[j])
            };
        }
        previous = current;
    }
    previous[b.len()]
}

fn check_goldens(backend: &str, engine: impl Fn(&str) -> Option<AsrEngine>, streaming: bool) {
    for (path, golden) in goldens(backend) {
        let Some(engine) = engine(&golden.model) else {
            continue;
        };
        let Some(dir) = model_dir(&golden.model) else {
            continue;
        };
        let audio = audio::read(dir.join(&golden.audio), DecodeLimits::default()).unwrap();
        let session = engine
            .start(
                audio.sample_rate,
                AsrOptions::default(),
                Duration::from_secs(10),
            )
            .unwrap();
        for chunk in audio.samples.chunks(1_600) {
            session.push(chunk.to_vec(), secs(60)).unwrap();
        }
        let result = session.finish(secs(120));
        let outcome = result.as_ref().unwrap();
        let name = path.display().to_string();
        check_text(&name, "text", &outcome.text(), &golden.text);
        assert_eq!(outcome.segments.len(), golden.segments.len(), "{name}");
        let tolerance = Duration::from_millis(20);
        for (got, want) in outcome.segments.iter().zip(&golden.segments) {
            check_text(&name, "segment", &got.text, &want.text);
            if let Some(start) = want.start_ms {
                assert!(
                    got.start.abs_diff(Duration::from_millis(start)) <= tolerance,
                    "{name}"
                );
            }
            if let Some(end) = want.end_ms {
                assert!(
                    got.end.abs_diff(Duration::from_millis(end)) <= tolerance,
                    "{name}"
                );
            }
        }
        if streaming {
            let partials = streaming_partials(&golden.model, &audio);
            if partials != golden.partials {
                let common = common_in_order(&golden.partials, &partials);
                #[expect(clippy::cast_precision_loss, reason = "a few dozen partials")]
                let share = common as f64 / golden.partials.len().max(1) as f64;
                eprintln!(
                    "{name}: {common} of {} golden partials appear in order ({} produced)",
                    golden.partials.len(),
                    partials.len()
                );
                assert!(
                    share >= MIN_PARTIALS_IN_ORDER,
                    "{name}: only {common} of {} golden partials appear in order\n  got:  {partials:?}\n  want: {:?}",
                    golden.partials.len(),
                    golden.partials
                );
            }
        }
    }
}

/// The partial results a streaming model reports for `audio`, fed in the
/// 1 600-frame chunks `check_goldens` pushes. They are read from the
/// backend's stream, since an observer may skip a partial result that a
/// newer one replaced before it was read. Goldens are 16 kHz clips, the
/// model's own rate, so nothing is resampled.
fn streaming_partials(id: &str, audio: &AudioBuffer) -> Vec<String> {
    let backend = AsrConfig::streaming(model_dir(id).unwrap()).load().unwrap();
    assert_eq!(audio.sample_rate, backend.capabilities().sample_rate);
    let sent = Collected::default();
    let mut stream = backend.open(&AsrOptions::default(), sent.events()).unwrap();
    for chunk in audio.samples.chunks(1_600) {
        stream.accept(chunk).unwrap();
    }
    stream.finish().unwrap();
    sent.take()
        .into_iter()
        .filter_map(|event| match event {
            AsrEvent::Partial(partial) => Some(partial.text),
            _ => None,
        })
        .collect()
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn golden_streaming_zipformer() {
    check_goldens("sherpa-streaming", streaming, true);
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn golden_sense_voice() {
    check_goldens("sherpa-sense-voice", |_| sense_voice(), false);
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn contract_streaming_zipformer() {
    if streaming("streaming-en").is_some() {
        run_asr_contract(|| streaming("streaming-en").unwrap());
    }
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn contract_sense_voice_with_silero() {
    if sense_voice().is_some() {
        run_asr_contract(|| sense_voice().unwrap());
    }
}

/// Feeds `audio` to a stream of `backend` in 100 ms blocks, checks the
/// activity it reports, and returns how far each start lay behind the
/// input fed before the block that reported it.
fn activity(backend: &dyn AsrBackend, audio: &AudioBuffer) -> (Vec<AsrEvent>, Vec<Duration>) {
    assert_eq!(audio.sample_rate, backend.capabilities().sample_rate);
    assert!(backend.capabilities().reports_activity);
    let sent = Collected::default();
    let mut stream = backend.open(&AsrOptions::default(), sent.events()).unwrap();
    let (mut events, mut lags) = (Vec::new(), Vec::new());
    let mut fed = Duration::ZERO;
    for chunk in audio.samples.chunks(1_600) {
        stream.accept(chunk).unwrap();
        for event in sent.take() {
            if let AsrEvent::SpeechStarted { at } = event {
                lags.push(fed.saturating_sub(at));
            }
            events.push(event);
        }
        fed += audio.sample_rate.duration_of(chunk.len() as u64);
    }
    stream.finish().unwrap();
    events.extend(sent.take());
    check_activity(&events);
    (events, lags)
}

/// A streaming model reports every start within `LATENCY` (1 s) of the
/// input, so its activity promise holds; the lags are printed, to set the
/// constant.
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn streaming_activity_lags_the_input_by_at_most_the_latency() {
    for id in ["streaming-en", "streaming-bilingual", "streaming-zh"] {
        let Some(dir) = model_dir(id) else {
            continue;
        };
        let backend = AsrConfig::streaming(&dir).load().unwrap();
        for wav in ["test_wavs/0.wav", "test_wavs/1.wav"] {
            let audio = audio::read(dir.join(wav), DecodeLimits::default()).unwrap();
            let (events, lags) = activity(&backend, &audio);
            eprintln!("{id} {wav}: start lags {lags:?}");
            assert!(!lags.is_empty(), "no speech in {id} {wav}: {events:#?}");
            assert!(lags.iter().all(|lag| *lag <= Duration::from_secs(1)));
        }
    }
}

/// A real model ends a session at a pause at the same point however fast
/// the audio is pushed (A-07).
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn a_pause_ends_a_session_at_the_same_point_at_any_push_speed() {
    let cases = [
        ("streaming-en", "test_wavs/0.wav", streaming("streaming-en")),
        ("sense-voice", "test_wavs/en.wav", sense_voice()),
    ];
    for (id, wav, engine) in cases {
        let (Some(engine), Some(dir)) = (engine, model_dir(id)) else {
            continue;
        };
        // Three seconds of silence after the speech, so a pause ends it: the
        // 1.5 s pause must be confirmed, and a streaming model confirms
        // activity up to 1 s behind the input.
        let mut audio = audio::read(dir.join(wav), DecodeLimits::default()).unwrap();
        audio.samples.extend(vec![0.0; 48_000]);
        let options = AsrOptions::default().with_end_after_silence(Duration::from_millis(1_500));
        let results: Vec<_> = [800, 1_600, 16_000]
            .into_iter()
            .map(|chunk| {
                let session = engine
                    .start(audio.sample_rate, options.clone(), secs(10))
                    .unwrap();
                for piece in audio.samples.chunks(chunk) {
                    if session.push(piece, secs(60)).is_err() {
                        break;
                    }
                }
                session.finish(secs(120)).unwrap()
            })
            .collect();
        let total = audio.duration();
        eprintln!(
            "{}: cut at {:?} of {total:?}",
            engine.name(),
            results.iter().map(|r| r.duration).collect::<Vec<_>>()
        );
        assert!(results[0].duration < total, "the pause ended it");
        assert!(!results[0].segments.is_empty());
        for result in &results {
            assert_eq!(result.duration, results[0].duration);
            assert_eq!(result.text(), results[0].text());
        }
    }
}

/// SenseVoice behind silero reports each utterance's speech from the VAD.
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn offline_activity_comes_from_the_vad() {
    let (Some(dir), Some(vad)) = (model_dir("sense-voice"), model_dir("silero-vad")) else {
        return;
    };
    let backend = AsrConfig::offline(&dir, vad).load().unwrap();
    let audio = audio::read(dir.join("test_wavs/en.wav"), DecodeLimits::default()).unwrap();
    let (events, lags) = activity(&backend, &audio);
    eprintln!("sense-voice en.wav: start lags {lags:?}");
    let ends = events
        .iter()
        .filter(|event| matches!(event, AsrEvent::SpeechEnded { .. }))
        .count();
    assert!(ends >= 1 && ends == lags.len(), "{events:#?}");
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn punctuation_on_streaming_output() {
    let (Some(engine), Some(dir)) = (streaming("streaming-bilingual"), model_dir("punct-zh-en"))
    else {
        return;
    };
    let punct = PunctuationConfig::new(&dir).load().unwrap();
    let engine = engine.with_post_processor(punct);
    let wav = model_dir("streaming-bilingual")
        .unwrap()
        .join("test_wavs/0.wav");
    let audio = audio::read(wav, DecodeLimits::default()).unwrap();
    let outcome = engine
        .transcribe(&audio, AsrOptions::default(), secs(120))
        .unwrap();
    let text = outcome.text();
    assert!(text.contains(['，', '。', '？']), "{text}");
}

fn offline(id: &str, family: AsrFamily, hotwords: &[&str]) -> Option<AsrEngine> {
    let mut config =
        AsrConfig::offline(model_dir(id)?, model_dir("silero-vad")?).with_family(family);
    if !hotwords.is_empty() {
        config = config.with_hotwords(hotwords.iter().copied());
    }
    Some(AsrEngine::new(config.load().unwrap()))
}

fn transcribe_test_wav(engine: &AsrEngine, id: &str, wav: &str) -> String {
    let audio = audio::read(model_dir(id).unwrap().join(wav), DecodeLimits::default()).unwrap();
    engine
        .transcribe(&audio, AsrOptions::default(), secs(300))
        .unwrap()
        .text()
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn golden_offline_transducer() {
    check_goldens(
        "sherpa-offline-transducer",
        |id| offline(id, AsrFamily::OfflineTransducer, &[]),
        false,
    );
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn contract_offline_transducer() {
    if offline("transducer-zh", AsrFamily::OfflineTransducer, &[]).is_some() {
        run_asr_contract(|| offline("transducer-zh", AsrFamily::OfflineTransducer, &[]).unwrap());
    }
}

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn golden_paraformer() {
    check_goldens(
        "sherpa-paraformer",
        |id| offline(id, AsrFamily::Paraformer, &[]),
        false,
    );
}

/// A bias phrase changes the output of a transducer.
///
/// `test_wavs/5.wav` decodes as "周望君…". Biasing toward the homophone
/// "周望军" must replace it. The control engine is biased toward an
/// unrelated phrase: configuring any bias switches to
/// `modified_beam_search`, which alone changes some outputs, so comparing
/// against greedy decoding would not show the phrase had any effect.
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn bias_phrase_changes_the_output() {
    let engine = |phrase: &str| offline("transducer-zh", AsrFamily::OfflineTransducer, &[phrase]);
    let Some(control) = engine("文化") else {
        return;
    };
    let biased = engine("周望军").unwrap();
    let before = transcribe_test_wav(&control, "transducer-zh", "test_wavs/5.wav");
    let after = transcribe_test_wav(&biased, "transducer-zh", "test_wavs/5.wav");
    assert!(
        !before.contains("周望军"),
        "control already has it: {before}"
    );
    assert!(after.contains("周望军"), "bias had no effect: {after}");
}

/// Large models: Qwen3-ASR, FunASR-Nano, and FireRed. They need
/// `SPEECHKIT_LARGE_MODEL_TEST=1` and a `SPEECHKIT_MODEL_*` directory.
fn large(id: &str, family: AsrFamily) {
    if !large_model_tier() {
        return;
    }
    let Some(engine) = offline(id, family, &[]) else {
        return;
    };
    let dir = model_dir(id).unwrap();
    let wav = std::fs::read_dir(dir.join("test_wavs"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|ext| ext == "wav"))
        .unwrap();
    let audio = audio::read(&wav, DecodeLimits::default()).unwrap();
    let text = engine
        .transcribe(&audio, AsrOptions::default(), secs(600))
        .unwrap()
        .text();
    assert!(!text.trim().is_empty(), "{}", wav.display());
}

#[test]
#[ignore = "large model; set SPEECHKIT_LARGE_MODEL_TEST=1"]
fn large_qwen3_asr() {
    large("qwen3-asr", AsrFamily::Qwen3Asr);
}

#[test]
#[ignore = "large model; set SPEECHKIT_LARGE_MODEL_TEST=1"]
fn large_funasr_nano() {
    large("funasr-nano", AsrFamily::FunAsrNano);
}

#[test]
#[ignore = "large model; set SPEECHKIT_LARGE_MODEL_TEST=1"]
fn large_firered_aed() {
    large("firered-aed", AsrFamily::FireRedAed);
}

#[test]
#[ignore = "large model; set SPEECHKIT_LARGE_MODEL_TEST=1"]
fn large_firered_ctc() {
    large("firered-ctc", AsrFamily::FireRedCtc);
}

/// `inspect` names the kind of each published model.
#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn inspect_tells_published_models_apart() {
    use speechkit::sherpa::{PunctuationFamily, TtsFamily, inspect};
    let check = |id: &str, test: &dyn Fn(&speechkit::sherpa::ModelInfo) -> bool| {
        if let Some(dir) = model_dir(id) {
            let found = inspect(&dir).unwrap();
            assert!(test(&found), "{id}: {found:?}");
        }
    };
    check("streaming-en", &|m| {
        m.asr.contains(&AsrFamily::StreamingTransducer) && !m.keyword_spotter
    });
    check("sense-voice", &|m| m.asr == [AsrFamily::SenseVoice]);
    check("punct-en", &|m| {
        m.punctuation == Some(PunctuationFamily::CnnBiLstm) && m.asr.is_empty()
    });
    check("tts-piper-en", &|m| m.tts == Some(TtsFamily::Vits));
    for id in ["kws-en", "kws-zh", "kws-zh-en"] {
        check(id, &|m| m.keyword_spotter && m.asr.is_empty());
    }
}
