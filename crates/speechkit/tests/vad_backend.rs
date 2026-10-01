//! `VadBackend` passes the ASR contract, times and numbers its segments, and
//! reports speech activity from its VAD.
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{sync::atomic::Ordering, time::Duration};

use speechkit::{
    AudioBuffer, SampleRate,
    asr::{AsrBackend, AsrEngine, AsrEvent},
    vad::{EnergyVad, EnergyVadConfig, VadBackend},
};
use speechkit_testkit::{
    asr::Collected,
    contract::asr::{check_activity, options, run_asr_contract, tone},
    secs,
    vad::{CountingRecognizer, WindowVad},
};

fn engine(window: usize, min_samples: usize) -> AsrEngine {
    let backend = VadBackend::new(CountingRecognizer::new(min_samples), WindowVad::new(window))
        .expect("matching rates");
    AsrEngine::new(backend)
}

#[test]
fn passes_the_asr_contract() {
    run_asr_contract(|| engine(8_000, 0));
}

#[test]
fn segments_are_timed_numbered_and_empty_ones_skipped() {
    let recognizer = CountingRecognizer::new(1_000);
    let calls = recognizer.calls();
    let backend = VadBackend::new(recognizer, WindowVad::new(8_000)).unwrap();
    assert!(!backend.capabilities().reports_partials);
    let engine = AsrEngine::new(backend);
    let audio = AudioBuffer::new(SampleRate::HZ_16000, tone(16_500));
    let outcome = engine.transcribe(&audio, options(), secs(30)).unwrap();
    let segments = &outcome.segments;
    // Two full windows; the 500-sample tail is too short and is skipped.
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].utterance.0, 0);
    assert_eq!(segments[1].utterance.0, 1);
    assert_eq!(segments[0].start, Duration::ZERO);
    assert_eq!(segments[0].end, Duration::from_millis(500));
    assert_eq!(segments[1].start, Duration::from_millis(500));
    assert_eq!(segments[1].text, "n8000");
}

#[test]
fn rate_mismatch_is_rejected() {
    let vad = EnergyVad::new(EnergyVadConfig::default().with_sample_rate(SampleRate::HZ_8000));
    let error = VadBackend::new(CountingRecognizer::new(0), vad).unwrap_err();
    assert!(error.to_string().contains("8000 Hz"), "{error}");
}

#[test]
fn energy_vad_backend_finds_utterances() {
    let backend = VadBackend::new(
        CountingRecognizer::new(0),
        EnergyVad::new(EnergyVadConfig::default()),
    )
    .unwrap();
    let engine = AsrEngine::new(backend);
    let mut samples = tone(16_000);
    samples.extend(vec![0.0; 16_000]);
    samples.extend(tone(16_000));
    let audio = AudioBuffer::new(SampleRate::HZ_16000, samples);
    let outcome = engine.transcribe(&audio, options(), secs(30)).unwrap();
    assert_eq!(outcome.segments.len(), 2);
}

/// What a stream of `backend` sends for `audio`, fed in 100 ms blocks.
fn events_of(backend: &dyn AsrBackend, audio: &[f32]) -> Vec<AsrEvent> {
    let sent = Collected::default();
    let mut stream = backend.open(&options(), sent.events()).unwrap();
    for chunk in audio.chunks(1_600) {
        stream.accept(chunk).unwrap();
    }
    stream.finish().unwrap();
    sent.take()
}

fn energy(max_utterance: Duration) -> VadBackend<CountingRecognizer, EnergyVad> {
    VadBackend::new(
        CountingRecognizer::new(0),
        EnergyVad::new(EnergyVadConfig::default()),
    )
    .unwrap()
    .with_max_utterance(max_utterance)
}

fn silence(frames: usize) -> Vec<f32> {
    vec![0.0; frames]
}

fn starts(events: &[AsrEvent]) -> Vec<Duration> {
    events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::SpeechStarted { at } => Some(*at),
            _ => None,
        })
        .collect()
}

fn ends(events: &[AsrEvent]) -> Vec<(Duration, u64)> {
    events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::SpeechEnded { at, utterance } => Some((*at, utterance.0)),
            _ => None,
        })
        .collect()
}

fn segments(events: &[AsrEvent]) -> Vec<(u64, String)> {
    events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::Segment(segment) => Some((segment.utterance.0, segment.text.clone())),
            _ => None,
        })
        .collect()
}

fn near(got: Duration, want_ms: u64) -> bool {
    got.abs_diff(Duration::from_millis(want_ms)) <= Duration::from_millis(30)
}

#[test]
fn speech_starts_and_ends_with_each_utterance() {
    let backend = energy(Duration::from_secs(20));
    assert!(backend.capabilities().reports_activity);
    let mut audio = tone(16_000);
    audio.extend(silence(16_000));
    audio.extend(tone(16_000));
    audio.extend(silence(8_000));
    let events = events_of(&backend, &audio);
    check_activity(&events);
    let starts = starts(&events);
    assert_eq!(starts.len(), 2, "{events:#?}");
    assert!(near(starts[0], 0) && near(starts[1], 2_000), "{starts:?}");
    let ends = ends(&events);
    assert_eq!(ends.len(), 2, "{events:#?}");
    assert!(near(ends[0].0, 1_000) && near(ends[1].0, 3_000), "{ends:?}");
    assert_eq!((ends[0].1, ends[1].1), (0, 1));
    assert_eq!(segments(&events).len(), 2);
    // At the end of input, activity is known to its end.
    assert_eq!(
        events.last(),
        Some(&AsrEvent::ActivityKnown {
            through: Duration::from_millis(3_500)
        })
    );
}

#[test]
fn a_long_utterance_is_cut_while_speech_goes_on() {
    let backend = energy(Duration::from_secs(1));
    let mut audio = tone(40_000);
    audio.extend(silence(8_000));
    let events = events_of(&backend, &audio);
    check_activity(&events);
    assert_eq!(starts(&events).len(), 1, "{events:#?}");
    let ends = ends(&events);
    assert_eq!(ends.len(), 1, "{events:#?}");
    assert_eq!(ends[0].1, 2, "the end names the last utterance");
    assert!(near(ends[0].0, 2_500), "{ends:?}");
    let ids: Vec<u64> = segments(&events).iter().map(|(id, _)| *id).collect();
    assert_eq!(ids, [0, 1, 2]);
}

#[test]
fn a_dropped_blip_ends_with_an_empty_segment() {
    let backend = energy(Duration::from_secs(20));
    let mut audio = silence(8_000);
    audio.extend(tone(320));
    audio.extend(silence(16_000));
    let events = events_of(&backend, &audio);
    check_activity(&events);
    assert_eq!(starts(&events).len(), 1, "{events:#?}");
    assert_eq!(ends(&events).len(), 1, "{events:#?}");
    assert_eq!(segments(&events), [(0, String::new())]);
    // Readers and the transcript never see the empty segment.
    let engine = AsrEngine::new(backend);
    let buffer = AudioBuffer::new(SampleRate::HZ_16000, audio);
    let transcript = engine.transcribe(&buffer, options(), secs(30)).unwrap();
    assert!(transcript.segments.is_empty());
}

#[test]
fn activity_trails_the_input_by_the_start_delay_between_speech() {
    let backend = energy(Duration::from_secs(20));
    let sent = Collected::default();
    let mut stream = backend.open(&options(), sent.events()).unwrap();
    let block = silence(1_600);
    let mut known = Duration::ZERO;
    for tenth in 1..=10_u64 {
        stream.accept(&block).unwrap();
        for event in sent.take() {
            if let AsrEvent::ActivityKnown { through } = event {
                known = through;
            }
        }
        let fed = Duration::from_millis(100 * tenth);
        // Whole 30 ms frames, less one frame of start delay.
        assert!(known <= fed, "{known:?} {fed:?}");
        assert!(
            known + Duration::from_millis(60) >= fed,
            "{known:?} {fed:?}"
        );
    }
    // While speech goes on, activity is known only to its start.
    stream.accept(&tone(8_000)).unwrap();
    let events = sent.take();
    let start = starts(&events)[0];
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AsrEvent::ActivityKnown { through } if *through > start))
    );
}
