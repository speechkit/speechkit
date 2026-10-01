//! The ASR contract suite: rules `A-01` to `A-19`.
//!
//! Each `aXX_*` function checks one rule against engines built by `make`,
//! and works with any backend: it only pushes synthetic audio and checks
//! behavior every backend must show. Rules that need a misbehaving backend
//! (a blocked native call, a failure, a panic, events while idle) are
//! checked by the functions in [`faults`], which use
//! [`FakeAsr`](crate::asr::FakeAsr). `A-12` needs a process-wide log
//! capture, so [`a12_no_transcript_logging`] takes the captured logs from
//! its caller and is not part of [`run_asr_contract`].
//!
//! [`run_asr_contract`] runs every generic check; [`faults::run_all`] runs
//! every fault check.

use std::{
    f32::consts::TAU,
    time::{Duration, Instant},
};

use speechkit::{
    AudioBuffer, RecvError, SampleRate, SpeechError,
    asr::{
        AsrEngine, AsrEvent, AsrLimits, AsrOptions, AsrResult, AsrSession, AsrUpdate, AsrUpdates,
        PostProcessor, PushErrorKind, Segment,
    },
};

use super::SETTLE;
use crate::{eventually, secs};

/// The input rate every check uses.
pub const RATE: SampleRate = SampleRate::HZ_16000;

/// `frames` frames of a quiet 440 Hz tone at [`RATE`].
pub fn tone(frames: usize) -> Vec<f32> {
    let step = TAU * 440.0 / RATE.hz() as f32;
    (0..frames)
        .map(|i| 0.1 * (step * (i % 16_000) as f32).sin())
        .collect()
}

/// Default session options. Every check starts its sessions at [`RATE`].
pub fn options() -> AsrOptions {
    AsrOptions::default()
}

fn start(engine: &AsrEngine) -> AsrSession {
    engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start a session")
}

/// Pushes `tenths` tenths of a second of tone.
pub fn push_tenths(session: &AsrSession, tenths: usize) {
    for _ in 0..tenths {
        session.push(tone(1_600), secs(30)).expect("push audio");
    }
}

fn wait_idle(engine: &AsrEngine) {
    assert!(
        eventually(SETTLE, || engine.active_sessions() == 0),
        "sessions did not release their slots"
    );
}

fn error_of(result: &AsrResult) -> Option<&SpeechError> {
    result.as_ref().err().map(|failure| &failure.error)
}

fn fail(what: &str) -> ! {
    panic!("{what}")
}

/// Whether two results say the same: equal transcripts, or failures with
/// the same error and what they confirmed.
pub fn same(a: &AsrResult, b: &AsrResult) -> bool {
    match (a, b) {
        (Ok(a), Ok(b)) => a == b,
        (Err(a), Err(b)) => {
            a.error.to_string() == b.error.to_string() && a.confirmed == b.confirmed
        }
        _ => false,
    }
}

/// Reads updates until `Closed`: the segments among them, and every update.
pub fn drain(updates: &mut AsrUpdates) -> (Vec<Segment>, Vec<AsrUpdate>) {
    let mut segments = Vec::new();
    let mut all = Vec::new();
    loop {
        let update = updates.recv(secs(30)).expect("an update");
        if let AsrUpdate::Segment(segment) = &update {
            segments.push(segment.clone());
        }
        let closed = matches!(update, AsrUpdate::Closed(_));
        all.push(update);
        if closed {
            assert_eq!(updates.try_recv().err(), Some(RecvError::Closed));
            return (segments, all);
        }
    }
}

/// A-01: a refused chunk comes back unchanged, whatever the reason, and a
/// bad sample is reported by index while the session stays open.
pub fn a01_refused_chunk_returned(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    let limit = RATE.frames_in(engine.limits().input_queue);
    let too_large = tone(usize::try_from(limit).expect("chunk fits in memory") + 1);
    let refused = session
        .try_push(too_large.clone())
        .expect_err("longer than the queue");
    assert_eq!(refused.kind, PushErrorKind::TooLarge);
    assert_eq!(refused.into_chunk(), too_large);
    for (bad, index) in [(f32::NAN, 3), (f32::INFINITY, 0), (1.5, 159)] {
        let mut chunk = tone(160);
        chunk[index] = bad;
        let refused = session.try_push(chunk.clone()).expect_err("a bad sample");
        assert_eq!(refused.kind, PushErrorKind::Invalid { index });
        assert_eq!(refused.chunk.len(), chunk.len());
    }
    session.try_push(tone(1_600)).expect("still accepting");
    session.close_input();
    let late = tone(160);
    let refused = session.try_push(late.clone()).expect_err("closed");
    assert_eq!(refused.kind, PushErrorKind::Closed);
    assert_eq!(refused.into_chunk(), late);
    let transcript = session.finish(secs(30)).expect("the session succeeds");
    assert_eq!(transcript.duration, Duration::from_millis(100));
    wait_idle(&engine);
}

/// A-02: `push` waits for room, so more audio than the queue holds goes
/// through.
pub fn a02_backpressure(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    let queue = RATE.frames_in(engine.limits().input_queue);
    let tenths = usize::try_from(queue / 1_600).expect("small") * 2 + 5;
    push_tenths(&session, tenths);
    let transcript = session.finish(secs(60)).expect("the session succeeds");
    assert_eq!(
        transcript.duration,
        Duration::from_millis(tenths as u64 * 100)
    );
    wait_idle(&engine);
}

/// A-03: `finish` returns the same result every time, as do `wait` and
/// `result()` once the session has ended; a `recv` or `wait` that times out
/// changes nothing.
pub fn a03_result_never_changes(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    push_tenths(&session, 5);
    assert!(session.wait(Duration::from_millis(10)).is_none());
    let mut updates = session.updates();
    while let Ok(update) = updates.recv(Duration::from_millis(10)) {
        assert!(!matches!(update, AsrUpdate::Closed(_)), "still running");
    }
    push_tenths(&session, 1);
    let first = session.finish(secs(30));
    let second = session.finish(secs(30));
    assert!(same(&first, &second));
    assert!(same(&first, session.result().expect("a result")));
    assert!(same(&first, &session.wait(secs(1)).expect("ended")));
    let transcript = first.as_ref().expect("the session succeeds");
    assert_eq!(transcript.duration, Duration::from_millis(600));

    let cancelled = start(&engine);
    push_tenths(&cancelled, 5);
    cancelled.cancel();
    let first = cancelled.finish(secs(30));
    wait_idle(&engine);
    let later = cancelled.result().expect("a result");
    assert!(same(&first, later));
    assert!(matches!(error_of(later), Some(SpeechError::Cancelled)));
}

/// A-04: a failure keeps what was confirmed, including the audio
/// transcribed.
pub fn a04_failure_keeps_confirmed(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    push_tenths(&session, 3);
    // Wait until the session has taken the audio, then cancel.
    assert!(eventually(SETTLE, || session.queued() == Duration::ZERO));
    session.cancel();
    let failure = session.finish(secs(30)).expect_err("cancelled");
    assert!(failure.confirmed.duration <= Duration::from_millis(300));
    wait_idle(&engine);
}

/// A-05: every reader gets every segment, in order, whenever it started
/// reading; a reader that never reads, or is dropped, changes nothing.
pub fn a05_readers(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    let mut early = session.updates();
    let idle = session.updates();
    drop(session.updates());
    push_tenths(&session, 20);
    let result = session.finish(secs(60));
    let transcript = result.as_ref().expect("the session succeeds");
    let (segments, updates) = drain(&mut early);
    assert_eq!(segments, transcript.segments);
    assert!(matches!(updates.last(), Some(AsrUpdate::Closed(Ok(_)))));
    let mut late = session.updates();
    let (segments, _) = drain(&mut late);
    assert_eq!(segments, transcript.segments);
    drop(idle);
    let ids: Vec<_> = transcript.segments.iter().map(|s| s.utterance).collect();
    assert!(ids.windows(2).all(|w| w[0] <= w[1]), "{ids:?}");

    let plain = start(&engine);
    push_tenths(&plain, 20);
    let without = plain.finish(secs(60));
    assert_eq!(
        without.as_ref().expect("the session succeeds").segments,
        transcript.segments
    );
    wait_idle(&engine);
}

/// A-06: no audio stays queued once a session ends. The history, wake-word
/// reservations, backlogs, and recording of a microphone are tested with
/// it.
pub fn a06_no_retention(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let finished = start(&engine);
    push_tenths(&finished, 5);
    let _ = finished.finish(secs(30));
    assert_eq!(finished.queued(), Duration::ZERO);
    let cancelled = start(&engine);
    push_tenths(&cancelled, 5);
    cancelled.cancel();
    assert_eq!(cancelled.queued(), Duration::ZERO);
    wait_idle(&engine);
}

/// A-09: a session that ends on its own has freed its slot by the time
/// `transcribe` returns, so back-to-back sessions never wait.
///
/// The other slots are held open, so each round needs the slot the
/// previous round used. The race this guards against is a few
/// instructions wide, so the check repeats it.
pub fn a09_slots_and_panics(make: &dyn Fn() -> AsrEngine) {
    const ROUNDS: usize = 100;
    let engine = make();
    let limit = engine.max_sessions();
    let held: Vec<_> = (1..limit).map(|_| start(&engine)).collect();
    let audio = AudioBuffer::new(RATE, tone(1_600));
    for round in 0..ROUNDS {
        engine
            .transcribe(&audio, options(), secs(30))
            .unwrap_or_else(|failure| fail(&format!("round {round}: {}", failure.error)));
        assert_eq!(
            engine.active_sessions(),
            limit - 1,
            "round {round}: the slot was still held after transcribe returned"
        );
    }
    drop(held);
    wait_idle(&engine);
}

/// A-12: transcripts and audio never appear in logs above TRACE. `logs`
/// returns everything logged at DEBUG and above since the process started.
pub fn a12_no_transcript_logging(make: &dyn Fn() -> AsrEngine, logs: &dyn Fn() -> String) {
    let engine = make();
    let session = start(&engine);
    push_tenths(&session, 20);
    let transcript = session.finish(secs(60)).expect("the session succeeds");
    wait_idle(&engine);
    let logged = logs();
    assert!(
        logged.contains("session"),
        "the capture saw nothing: {logged}"
    );
    for segment in &transcript.segments {
        assert!(!logged.contains(segment.text.trim()), "{logged}");
    }
}

/// A-13: `cancel()` and dropping the session both cancel it.
pub fn a13_cancel_and_drop(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    session.try_push(tone(160)).expect("push");
    session.cancel();
    assert!(matches!(
        error_of(&session.finish(secs(30))),
        Some(SpeechError::Cancelled)
    ));
    let refused = session.try_push(tone(160)).expect_err("cancelled");
    assert_eq!(refused.kind, PushErrorKind::Closed);
    let dropped = start(&engine);
    push_tenths(&dropped, 2);
    drop(dropped);
    wait_idle(&engine);
}

/// A-14: a deadline that passes returns promptly.
pub fn a14_deadline_prompt(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    push_tenths(&session, 10);
    let begun = Instant::now();
    let result = session.finish(Instant::now());
    assert!(begun.elapsed() < Duration::from_millis(500));
    match error_of(&result) {
        None | Some(SpeechError::DeadlineExceeded) => {}
        Some(other) => fail(&format!("unexpected error {other}")),
    }
    drop(session);
    wait_idle(&engine);
}

/// A-15: the default limits.
pub fn a15_defaults(_: &dyn Fn() -> AsrEngine) {
    let limits = AsrLimits::default();
    assert_eq!(limits.input_queue, Duration::from_secs(2));
    assert_eq!(limits.max_history_bytes, 8 * 1024 * 1024);
    assert_eq!(
        AsrEngine::new(crate::asr::FakeAsr::hello_world()).max_sessions(),
        8
    );
}

/// Leaves segment text as it is.
struct Unchanged;

impl PostProcessor for Unchanged {
    fn process(&self, text: &str) -> Result<String, SpeechError> {
        Ok(text.to_owned())
    }
}

/// A-16: clones share the session limit, and so do engines made from them
/// with `with_limits` or `with_post_processor`.
pub fn a16_clones_share_slots(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let clone = engine.clone();
    let configured = clone
        .clone()
        .with_limits(*engine.limits())
        .with_post_processor(Unchanged);
    let limit = engine.max_sessions();
    let sessions: Vec<_> = (0..limit)
        .map(|i| {
            if i % 2 == 0 {
                start(&engine)
            } else {
                start(&clone)
            }
        })
        .collect();
    for handle in [&engine, &clone, &configured] {
        assert_eq!(handle.active_sessions(), limit);
        assert!(matches!(
            handle.start(RATE, options(), Duration::from_millis(50)),
            Err(SpeechError::Capacity)
        ));
    }
    drop(sessions);
    wait_idle(&engine);
    drop(start(&clone));
    wait_idle(&clone);
}

/// A-17: an empty chunk is accepted and does nothing.
pub fn a17_empty_chunk_noop(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = start(&engine);
    session.try_push(Vec::new()).expect("empty push");
    session
        .push(Vec::new(), Instant::now())
        .expect("empty push");
    let transcript = session.finish(secs(30)).expect("the session succeeds");
    assert_eq!(transcript.duration, Duration::ZERO);
    wait_idle(&engine);
}

/// A-18: hints with no phrases are ignored.
pub fn a18_empty_hints_ignored(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let session = engine
        .start(
            RATE,
            options().with_hints(Vec::<String>::new()),
            Duration::from_secs(10),
        )
        .expect("empty hints are accepted");
    drop(session);
    if !engine.capabilities().accepts_hints {
        assert!(matches!(
            engine.start(RATE, options().with_hints(["x"]), Duration::from_secs(10)),
            Err(SpeechError::Unsupported(_))
        ));
    }
    wait_idle(&engine);
}

/// A-19: an expired deadline fails `start` and `transcribe` without taking
/// a slot.
pub fn a19_expired_deadline_no_slot(make: &dyn Fn() -> AsrEngine) {
    let engine = make();
    let audio = AudioBuffer::new(RATE, tone(16_000));
    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .unwrap_or_else(Instant::now);
    let failure = engine
        .transcribe(&audio, options(), expired)
        .expect_err("expired");
    assert!(matches!(failure.error, SpeechError::DeadlineExceeded));
    assert_eq!(failure.confirmed.duration, Duration::ZERO);
    assert!(matches!(
        engine.start(RATE, options(), expired),
        Err(SpeechError::DeadlineExceeded)
    ));
    assert_eq!(engine.active_sessions(), 0);
    let transcript = engine
        .transcribe(&audio, options(), secs(60))
        .expect("transcribe succeeds");
    assert_eq!(transcript.duration, Duration::from_secs(1));
    wait_idle(&engine);
}

/// A generic check: it builds engines with the factory it is given.
pub type Check = fn(&dyn Fn() -> AsrEngine);

/// Every generic check, in order.
pub const GENERIC: &[(&str, Check)] = &[
    ("a01_refused_chunk_returned", a01_refused_chunk_returned),
    ("a02_backpressure", a02_backpressure),
    ("a03_result_never_changes", a03_result_never_changes),
    ("a04_failure_keeps_confirmed", a04_failure_keeps_confirmed),
    ("a05_readers", a05_readers),
    ("a06_no_retention", a06_no_retention),
    ("a09_slots_and_panics", a09_slots_and_panics),
    ("a13_cancel_and_drop", a13_cancel_and_drop),
    ("a14_deadline_prompt", a14_deadline_prompt),
    ("a15_defaults", a15_defaults),
    ("a16_clones_share_slots", a16_clones_share_slots),
    ("a17_empty_chunk_noop", a17_empty_chunk_noop),
    ("a18_empty_hints_ignored", a18_empty_hints_ignored),
    ("a19_expired_deadline_no_slot", a19_expired_deadline_no_slot),
];

/// Checks the speech activity in `events`, everything one backend stream
/// sent, against "Writing a backend":
///
/// - `SpeechStarted` and `SpeechEnded` alternate, starting with a start;
/// - `ActivityKnown` never goes back, and no start or end sent after it
///   lies before it;
/// - segment utterance IDs increase, and each `SpeechEnded` names an
///   utterance of its speech that is committed by the end.
pub fn check_activity(events: &[AsrEvent]) {
    let mut speaking = false;
    let mut known = Duration::ZERO;
    let mut last_segment: Option<u64> = None;
    // The last segment committed before the speech in progress started.
    let mut before_speech: Option<u64> = None;
    let mut named = Vec::new();
    for (index, event) in events.iter().enumerate() {
        let context = || format!("event {index} of {events:#?}");
        match event {
            AsrEvent::SpeechStarted { at } => {
                assert!(!speaking, "a second start: {}", context());
                assert!(*at >= known, "a start behind what was known: {}", context());
                speaking = true;
                before_speech = last_segment;
            }
            AsrEvent::SpeechEnded { at, utterance } => {
                assert!(speaking, "an end without a start: {}", context());
                assert!(*at >= known, "an end behind what was known: {}", context());
                assert!(
                    before_speech.is_none_or(|last| utterance.0 > last),
                    "an end naming an earlier speech's utterance: {}",
                    context()
                );
                speaking = false;
                named.push(utterance.0);
            }
            AsrEvent::ActivityKnown { through } => {
                assert!(*through >= known, "activity went back: {}", context());
                known = *through;
            }
            AsrEvent::Segment(segment) => {
                assert!(
                    last_segment.is_none_or(|last| segment.utterance.0 > last),
                    "segment IDs must increase: {}",
                    context()
                );
                last_segment = Some(segment.utterance.0);
            }
            AsrEvent::Partial(partial) => {
                assert!(
                    last_segment.is_none_or(|last| partial.utterance.0 > last),
                    "a partial after its segment: {}",
                    context()
                );
            }
        }
    }
    let committed: Vec<u64> = events
        .iter()
        .filter_map(|event| match event {
            AsrEvent::Segment(segment) => Some(segment.utterance.0),
            _ => None,
        })
        .collect();
    for utterance in named {
        assert!(
            committed.contains(&utterance),
            "SpeechEnded names utterance {utterance}, which is never committed: {events:#?}"
        );
    }
}

/// Runs every generic check against engines built by `make`.
pub fn run_asr_contract(make: impl Fn() -> AsrEngine) {
    for (name, check) in GENERIC {
        eprintln!("asr contract: {name}");
        check(&make);
    }
}

pub mod faults;
