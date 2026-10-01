//! Contract checks that need a misbehaving backend. They build their own
//! engines on [`FakeAsr`].

use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use speechkit::{
    AudioBuffer, SpeechError,
    asr::{AsrEngine, AsrLimits, AsrOptions, AsrResult, AsrUpdate, PushErrorKind},
    vad::{EnergyVad, EnergyVadConfig, VadBackend},
};

use super::{RATE, drain, options, same, tone};
use crate::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    contract::SETTLE,
    eventually, secs,
    vad::CountingRecognizer,
};

fn engine(fake: FakeAsr, session: AsrLimits) -> AsrEngine {
    AsrEngine::new(fake).with_limits(session)
}

fn small_queue() -> AsrLimits {
    AsrLimits::default().with_input_queue(Duration::from_millis(200))
}

fn blocked(gate: &Gate) -> FakeAsr {
    FakeAsr::new(Script::new().then(
        Trigger::AfterSamples(1),
        Step::BlockUntilReleased(gate.clone()),
    ))
}

/// A-01, A-02, A-17: a full queue refuses at once and hands the chunk back,
/// and `push` waits until there is room or the deadline passes.
pub fn a02_backpressure_full() {
    let gate = Gate::new();
    let engine = engine(blocked(&gate), small_queue());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.try_push(tone(1_600)).expect("first chunk");
    assert!(gate.wait_entered(1, SETTLE));
    // The session holds the first chunk; two more fill the 200 ms queue.
    session.try_push(tone(1_600)).expect("fits");
    session.try_push(tone(1_600)).expect("fits");
    let chunk = tone(1_600);
    let refused = session.try_push(chunk.clone()).expect_err("full");
    assert_eq!(refused.kind, PushErrorKind::Full);
    assert_eq!(refused.into_chunk(), chunk);
    // A-17: an empty chunk is fine even now.
    session.try_push(Vec::new()).expect("empty push");
    let begun = Instant::now();
    let refused = session
        .push(chunk.clone(), Duration::from_millis(50))
        .expect_err("still full");
    assert_eq!(refused.kind, PushErrorKind::Full);
    assert_eq!(refused.chunk, chunk);
    assert!(begun.elapsed() >= Duration::from_millis(50));
    let releaser = {
        let gate = gate.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            gate.release();
        })
    };
    session.push(chunk, secs(10)).expect("room frees up");
    releaser.join().expect("releaser");
    let transcript = session.finish(secs(10)).expect("success");
    assert_eq!(transcript.duration, Duration::from_millis(400));
}

/// A-02: `push` returns when the session ends while it waits.
pub fn a02_push_wait_wakes_on_close() {
    let gate = Gate::new();
    let engine = engine(blocked(&gate), small_queue());
    let session = Arc::new(
        engine
            .start(RATE, options(), Duration::from_secs(10))
            .expect("start"),
    );
    session.try_push(tone(1_600)).expect("first chunk");
    assert!(gate.wait_entered(1, SETTLE));
    for _ in 0..2 {
        session.try_push(tone(1_600)).expect("fill");
    }
    let canceller = {
        let session = session.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            session.cancel();
        })
    };
    let refused = session.push(tone(1_600), secs(10)).expect_err("ended");
    assert_eq!(refused.kind, PushErrorKind::Closed);
    assert_eq!(refused.chunk.len(), 1_600);
    canceller.join().expect("canceller");
    gate.release();
}

/// A-04: a backend failure keeps the confirmed text and duration.
pub fn a04_failure_keeps_confirmed() {
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1_600), Step::Segment(0, "kept"))
            .then(
                Trigger::AfterSamples(3_200),
                Step::Fail(SpeechError::backend("fake", true, "boom")),
            ),
    );
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    for _ in 0..5 {
        if let Err(refused) = session.push(tone(1_600), secs(10)) {
            assert_eq!(refused.kind, PushErrorKind::Closed);
        }
    }
    let failure = session.finish(secs(10)).expect_err("the backend fails");
    assert!(failure.error.retryable());
    assert_eq!(failure.confirmed.text(), "kept");
    assert!(failure.confirmed.duration >= Duration::from_millis(200));
}

/// A-05: partials merge by utterance, segments come in order, and a late
/// reader catches up without repeats.
pub fn a05_readers() {
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1_600), Step::Partial(0, "he"))
            .then(Trigger::AfterSamples(1_600), Step::Partial(0, "hell"))
            .then(Trigger::AfterSamples(3_200), Step::Segment(0, "hello"))
            .then(Trigger::AfterSamples(4_800), Step::Partial(1, "wor"))
            .then(Trigger::OnFinish, Step::Segment(1, "world")),
    );
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    let mut updates = session.updates();
    for _ in 0..4 {
        session.push(tone(1_600), secs(10)).expect("push");
    }
    let mut late = session.updates();
    let transcript = session.finish(secs(10)).expect("success");
    assert_eq!(transcript.text(), "hello world");
    let (segments, all) = drain(&mut updates);
    assert_eq!(segments, transcript.segments);
    let partials: Vec<_> = all
        .iter()
        .filter_map(|u| match u {
            AsrUpdate::Partial(p) => Some(p.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(!partials.contains(&"he") || partials.contains(&"hell"));
    let (segments, all) = drain(&mut late);
    assert_eq!(segments, transcript.segments);
    let segment_count = all
        .iter()
        .filter(|u| matches!(u, AsrUpdate::Segment(_)))
        .count();
    assert_eq!(segment_count, 2, "none repeated");
    assert!(transcript.segments[0].end <= transcript.segments[1].end);
    assert!(transcript.segments[0].utterance < transcript.segments[1].utterance);
}

/// A-05: a reader that never reads doesn't slow the session, and a reader
/// that joins after many segments still gets each once, in order.
pub fn a05_slow_reader() {
    let mut script = Script::new();
    for (i, text) in ["a", "b", "c", "d", "e", "f"].into_iter().enumerate() {
        script = script.then(
            Trigger::AfterSamples(1_600 * (i as u64 + 1)),
            Step::Segment(i as u64, text),
        );
    }
    let engine = engine(FakeAsr::new(script), AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    let mut updates = session.updates();
    for _ in 0..6 {
        session.push(tone(1_600), secs(10)).expect("push");
    }
    let transcript = session.finish(secs(10)).expect("success");
    assert_eq!(transcript.text(), "a b c d e f");
    let (segments, all) = drain(&mut updates);
    assert_eq!(segments, transcript.segments);
    assert_eq!(all.len(), 7, "six segments and `Closed`, none repeated");
}

/// A-09: a deadline returns promptly, and the slot stays held until the
/// blocked backend call returns.
pub fn a09_slot_held_while_blocked() {
    let gate = Gate::new();
    let fake = blocked(&gate);
    let stats = fake.stats();
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.try_push(tone(1_600)).expect("push");
    assert!(gate.wait_entered(1, SETTLE));
    let begun = Instant::now();
    let result = session.finish(Duration::from_millis(50));
    assert!(begun.elapsed() < Duration::from_secs(2));
    assert!(matches!(
        result.as_ref(),
        Err(failure) if matches!(failure.error, SpeechError::DeadlineExceeded)
    ));
    drop(session);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        engine.active_sessions(),
        1,
        "the slot is held while blocked"
    );
    assert_eq!(stats.alive(), 1);
    gate.release();
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    assert!(eventually(SETTLE, || stats.alive() == 0));
    assert_eq!(stats.cancelled(), 1);
}

/// A-09: a backend panic becomes a non-retryable backend error that keeps
/// what was confirmed, and the engine keeps working.
pub fn a09_backend_panic_isolated() {
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1), Step::Segment(0, "before"))
            .then(Trigger::AfterSamples(1_601), Step::Panic),
    );
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.push(tone(1_600), secs(10)).expect("push");
    let _ = session.push(tone(1_600), secs(10));
    let failure = session.finish(secs(10)).expect_err("the backend panicked");
    assert!(matches!(
        &failure.error,
        SpeechError::Backend {
            retryable: false,
            ..
        }
    ));
    assert_eq!(failure.confirmed.text(), "before");
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    let again = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("the engine still works");
    drop(again);
}

/// A-09, A-16: a limit of one is shared by clones, and freed with the
/// stream.
pub fn a09_single_slot() {
    let engine = AsrEngine::new(FakeAsr::hello_world()).with_max_sessions(1);
    let clone = engine.clone();
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    assert!(matches!(
        clone.start(RATE, options(), Duration::from_millis(50)),
        Err(SpeechError::Capacity)
    ));
    drop(session);
    assert!(eventually(SETTLE, || clone.active_sessions() == 0));
    drop(
        clone
            .start(RATE, options(), Duration::from_secs(10))
            .expect("the slot is free again"),
    );
}

/// A-10: an event a stream sends while no audio is pushed reaches readers
/// at once, a failure it reports ends the session at once, and `finish`
/// returns only after the final events.
pub fn a10_events_while_idle() {
    let fake = FakeAsr::new(Script::new().then(
        Trigger::OnOpen,
        Step::SegmentLater(Duration::from_millis(50), 0, "unprompted"),
    ));
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    let mut updates = session.updates();
    match updates.recv(secs(5)) {
        Ok(AsrUpdate::Segment(segment)) => assert_eq!(segment.text, "unprompted"),
        other => panic_on(&format!("expected the segment, got {other:?}")),
    }

    let fake = FakeAsr::new(Script::new().then(
        Trigger::OnOpen,
        Step::FailLater(
            Duration::from_millis(50),
            SpeechError::backend("fake", true, "dropped"),
        ),
    ));
    let engine = self::engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    let ended = session.wait(secs(5)).expect("the failure ended it");
    assert!(ended.expect_err("failed").error.retryable());
    assert_eq!(engine.active_sessions(), 0, "the slot went first");

    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::OnFinish, Step::Sleep(Duration::from_millis(50)))
            .then(Trigger::OnFinish, Step::Segment(0, "final")),
    );
    let engine = self::engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.push(tone(1_600), secs(10)).expect("push");
    let transcript = session.finish(secs(10)).expect("success");
    assert_eq!(transcript.text(), "final");
}

fn panic_on(what: &str) -> ! {
    panic!("{what}")
}

/// A-03: events that arrive after the session ended are discarded.
pub fn a03_late_events_discarded() {
    let gate = Gate::new();
    let fake = FakeAsr::new(
        Script::new()
            .then(
                Trigger::AfterSamples(1),
                Step::BlockUntilReleased(gate.clone()),
            )
            .then(Trigger::AfterSamples(1), Step::Segment(0, "late")),
    );
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.try_push(tone(1_600)).expect("push");
    assert!(gate.wait_entered(1, SETTLE));
    session.cancel();
    let first = session.finish(secs(10));
    gate.release();
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    let later = session.result().expect("a result");
    assert!(same(&first, later));
    let failure = later.as_ref().expect_err("cancelled");
    assert!(matches!(failure.error, SpeechError::Cancelled));
    assert!(failure.confirmed.segments.is_empty());
}

/// A-11: the history is bounded; reaching the bound fails with `Capacity`
/// and keeps what fit.
pub fn a11_bounded() {
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1), Step::Segment(0, "12345"))
            .then(Trigger::AfterSamples(1_601), Step::Segment(1, "67890")),
    );
    let limits = AsrLimits::default().with_max_history_bytes(100);
    let engine = engine(fake, limits);
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    session.push(tone(1_600), secs(10)).expect("push");
    let _ = session.push(tone(1_600), secs(10));
    let failure = session.finish(secs(10)).expect_err("over the limit");
    assert!(matches!(failure.error, SpeechError::Capacity));
    assert_eq!(failure.confirmed.text(), "12345");
}

/// A-14: a deadline that passes while every slot is busy is `Capacity`;
/// one that passes while the backend opens is `DeadlineExceeded`, promptly,
/// and the open keeps its slot until it returns.
pub fn a14_deadline_while_opening() {
    let gate = Gate::new();
    let fake =
        FakeAsr::new(Script::new().then(Trigger::OnOpen, Step::BlockUntilReleased(gate.clone())));
    let engine = AsrEngine::new(fake).with_max_sessions(1);
    let begun = Instant::now();
    let opening = engine.start(RATE, options(), Duration::from_millis(100));
    assert!(matches!(opening, Err(SpeechError::DeadlineExceeded)));
    assert!(begun.elapsed() < Duration::from_secs(2));
    assert_eq!(engine.active_sessions(), 1, "the open holds its slot");
    let waiting = engine.start(RATE, options(), Duration::from_millis(100));
    assert!(matches!(waiting, Err(SpeechError::Capacity)));
    gate.release();
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
}

/// A-14: `start` waits for a slot that frees up within its deadline.
pub fn a14_start_waits_for_a_slot() {
    let engine = AsrEngine::new(FakeAsr::hello_world()).with_max_sessions(1);
    let held = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    let waiting = {
        let engine = engine.clone();
        std::thread::spawn(move || {
            engine
                .start(RATE, options(), Duration::from_secs(10))
                .map(drop)
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    drop(held);
    assert!(waiting.join().expect("no panic").is_ok());
}

/// A-19: an expired deadline never opens a backend stream.
pub fn a19_expired_deadline_no_slot() {
    let fake = FakeAsr::hello_world();
    let stats = fake.stats();
    let engine = engine(fake, AsrLimits::default());
    let audio = AudioBuffer::new(RATE, tone(1_600));
    let failure = engine
        .transcribe(&audio, options(), Instant::now())
        .expect_err("expired");
    assert!(matches!(failure.error, SpeechError::DeadlineExceeded));
    assert_eq!(stats.opened(), 0);
}

/// The session feeds the backend in blocks of 100 ms, and a partial block
/// only at the end of the input (A-15).
pub fn a15_feed_blocks() {
    let fake = FakeAsr::hello_world();
    let stats = fake.stats();
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options(), Duration::from_secs(10))
        .expect("start");
    for _ in 0..25 {
        session.push(tone(160), secs(10)).expect("push");
    }
    let _ = session.finish(secs(10));
    // 4 000 samples: two blocks of 1 600, then the rest.
    assert_eq!(stats.accepts(), 3);
}

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}

/// The sample at `value` ms, at 16 kHz.
fn sample(value: u64) -> u64 {
    value * 16
}

/// What a session on a fake running `script` did with `options`, fed
/// `total_ms` of audio in pushes of `chunk` frames: its result, every
/// update one reader saw, and how many samples the backend was fed.
fn endpointed(
    script: Script,
    options: AsrOptions,
    total_ms: u64,
    chunk: usize,
) -> (AsrResult, Vec<AsrUpdate>, u64) {
    let fake = FakeAsr::new(script).reporting_activity();
    let stats = fake.stats();
    let engine = engine(fake, AsrLimits::default());
    let session = engine
        .start(RATE, options, Duration::from_secs(10))
        .expect("start");
    let mut updates = session.updates();
    let total = usize::try_from(sample(total_ms)).expect("fits");
    let audio = tone(total);
    for piece in audio.chunks(chunk) {
        if session.push(piece, secs(10)).is_err() {
            break;
        }
    }
    let result = session.finish(secs(10));
    let (_, all) = drain(&mut updates);
    (result, all, stats.samples.load(Ordering::SeqCst))
}

fn turns(updates: &[AsrUpdate]) -> Vec<(String, Duration, Duration)> {
    updates
        .iter()
        .filter_map(|update| match update {
            AsrUpdate::TurnEnded(turn) => Some((turn.text(), turn.start, turn.end)),
            _ => None,
        })
        .collect()
}

/// A-07: endpointing stops a session at a cutoff C no earlier than the
/// endpoint T, only once the backend confirmed activity past T; nothing
/// after C is transcribed, and the result reports C. With a local model,
/// the same audio stops at the same C whatever the push speed.
pub fn a07_endpointing() {
    // A pause reported after the input passed T = 1.7 s: C is where the
    // input was when activity was confirmed past T.
    let script = Script::new()
        .then(Trigger::OnOpen, Step::started(Duration::ZERO))
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::ended(ms(1_000), 0),
        )
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::Segment(0, "one"),
        )
        .then(Trigger::AfterSamples(sample(2_500)), Step::known(ms(1_700)))
        .then(
            Trigger::AfterSamples(sample(3_000)),
            Step::Segment(1, "late"),
        );
    let options = AsrOptions::default().with_end_after_silence(ms(700));
    let (result, _, fed) = endpointed(script, options, 4_000, 1_600);
    let transcript = result.expect("a success");
    assert_eq!(transcript.duration, ms(2_500));
    assert_eq!(transcript.text(), "one", "nothing after C");
    assert_eq!(fed, sample(2_500), "audio past C is dropped unheard");

    // Speech that ends at 1.0 s and restarts at 1.6 s, reported only once
    // the input passed 1.7 s but before activity was known that far:
    // neither the turn nor the session ends there.
    let script = Script::new()
        .then(Trigger::OnOpen, Step::started(Duration::ZERO))
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::ended(ms(1_000), 0),
        )
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::Segment(0, "one"),
        )
        .then(Trigger::AfterSamples(sample(1_500)), Step::known(ms(1_500)))
        .then(
            Trigger::AfterSamples(sample(1_800)),
            Step::started(ms(1_600)),
        )
        .then(Trigger::AfterSamples(sample(2_000)), Step::known(ms(1_600)))
        .then(
            Trigger::AfterSamples(sample(3_000)),
            Step::ended(ms(3_000), 1),
        )
        .then(
            Trigger::AfterSamples(sample(3_000)),
            Step::Segment(1, "two"),
        )
        .then(Trigger::AfterSamples(sample(3_800)), Step::known(ms(3_700)));
    let options = AsrOptions::default()
        .with_end_after_silence(ms(700))
        .with_turn_end(ms(700));
    let (result, updates, _) = endpointed(script, options, 5_000, 1_600);
    assert_eq!(result.expect("a success").duration, ms(3_800));
    assert_eq!(
        turns(&updates),
        [("one two".to_owned(), Duration::ZERO, ms(3_000))]
    );

    // No speech: the session ends at the timeout with nothing, at C.
    let script = Script::new().then(Trigger::AfterSamples(sample(2_000)), Step::known(ms(2_000)));
    let options = AsrOptions::default().with_no_speech_timeout(ms(2_000));
    let (result, _, _) = endpointed(script, options, 4_000, 1_600);
    let transcript = result.expect("a success");
    assert!(transcript.segments.is_empty());
    assert_eq!(transcript.duration, ms(2_000));

    // The maximum length cuts exactly, mid-block.
    let options = AsrOptions::default().with_max_length(ms(1_250));
    let (result, _, fed) = endpointed(Script::new(), options, 3_000, 1_600);
    assert_eq!(result.expect("a success").duration, ms(1_250));
    assert_eq!(fed, sample(1_250));

    // A local model stops at the same C however fast the audio comes.
    let mut audio = tone(16_000);
    audio.extend(vec![0.0; 32_000]);
    let cutoffs: Vec<Duration> = [160, 1_600, 32_000]
        .into_iter()
        .map(|chunk| {
            let backend = VadBackend::new(
                CountingRecognizer::new(0),
                EnergyVad::new(EnergyVadConfig::default()),
            )
            .expect("matching rates");
            let engine = AsrEngine::new(backend);
            let options = AsrOptions::default().with_end_after_silence(ms(500));
            let session = engine
                .start(RATE, options, Duration::from_secs(10))
                .expect("start");
            for piece in audio.chunks(chunk) {
                if session.push(piece, secs(10)).is_err() {
                    break;
                }
            }
            session.finish(secs(10)).expect("a success").duration
        })
        .collect();
    assert!(cutoffs[0] < ms(3_000), "{cutoffs:?}");
    assert!(cutoffs.iter().all(|c| *c == cutoffs[0]), "{cutoffs:?}");

    // Without activity, ending at a pause is refused.
    let engine = AsrEngine::new(FakeAsr::new(Script::new()));
    let refused = engine.start(
        RATE,
        AsrOptions::default().with_end_after_silence(ms(500)),
        Duration::from_secs(10),
    );
    assert!(matches!(refused, Err(SpeechError::Unsupported(_))));
}

/// A-08: a turn ends only once the backend confirmed its silence; a
/// shorter pause or a cut never ends it, and `TurnEnded` carries every
/// segment of its turn and none of another, even when its text arrives
/// after newer speech started.
pub fn a08_turns() {
    // SpeechEnded(A), 700 ms of silence, SpeechStarted(B), Partial(B),
    // then A's late segment.
    let script = Script::new()
        .then(Trigger::OnOpen, Step::started(Duration::ZERO))
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::ended(ms(1_000), 0),
        )
        .then(Trigger::AfterSamples(sample(1_800)), Step::known(ms(1_700)))
        .then(
            Trigger::AfterSamples(sample(1_800)),
            Step::started(ms(1_800)),
        )
        .then(Trigger::AfterSamples(sample(2_000)), Step::Partial(1, "b"))
        .then(Trigger::AfterSamples(sample(2_200)), Step::Segment(0, "a"))
        .then(
            Trigger::AfterSamples(sample(2_500)),
            Step::ended(ms(2_500), 1),
        )
        .then(Trigger::AfterSamples(sample(2_500)), Step::Segment(1, "b"))
        .then(Trigger::AfterSamples(sample(3_300)), Step::known(ms(3_200)));
    let (_, updates, _) = endpointed(script, with_turns(), 4_000, 1_600);
    assert_eq!(
        turns(&updates),
        [
            ("a".to_owned(), Duration::ZERO, ms(1_000)),
            ("b".to_owned(), ms(1_800), ms(2_500)),
        ]
    );
    // A's turn arrived after B's speech started: a controller can tell.
    let newer = updates
        .iter()
        .position(|u| matches!(u, AsrUpdate::SpeechStarted { at } if *at == ms(1_800)));
    let first_turn = updates
        .iter()
        .position(|u| matches!(u, AsrUpdate::TurnEnded(_)));
    assert!(newer < first_turn, "{updates:#?}");

    a08_pauses_cuts_and_coughs();

    // Without activity, turn ends are refused.
    let engine = AsrEngine::new(FakeAsr::new(Script::new()));
    let refused = engine.start(
        RATE,
        AsrOptions::default().with_turn_end(ms(700)),
        Duration::from_secs(10),
    );
    assert!(matches!(refused, Err(SpeechError::Unsupported(_))));
}

fn with_turns() -> AsrOptions {
    AsrOptions::default().with_turn_end(ms(700))
}

/// A-08: a pause shorter than the turn end and an utterance cut at its
/// maximum don't end a turn; a turn with no words, such as a cough, still
/// ends.
fn a08_pauses_cuts_and_coughs() {
    // A pause mid-sentence, shorter than the turn end, and an utterance
    // cut at its maximum: neither ends the turn. A cough still does.
    let script = Script::new()
        .then(Trigger::OnOpen, Step::started(Duration::ZERO))
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::ended(ms(1_000), 0),
        )
        .then(
            Trigger::AfterSamples(sample(1_000)),
            Step::Segment(0, "half"),
        )
        .then(Trigger::AfterSamples(sample(1_400)), Step::known(ms(1_300)))
        .then(
            Trigger::AfterSamples(sample(1_400)),
            Step::started(ms(1_300)),
        )
        .then(
            Trigger::AfterSamples(sample(2_000)),
            Step::Segment(1, "cut"),
        )
        .then(Trigger::AfterSamples(sample(2_000)), Step::known(ms(1_300)))
        .then(
            Trigger::AfterSamples(sample(3_000)),
            Step::ended(ms(3_000), 2),
        )
        .then(
            Trigger::AfterSamples(sample(3_000)),
            Step::Segment(2, "end"),
        )
        .then(Trigger::AfterSamples(sample(3_800)), Step::known(ms(3_700)))
        .then(
            Trigger::AfterSamples(sample(4_000)),
            Step::started(ms(3_900)),
        )
        .then(
            Trigger::AfterSamples(sample(4_100)),
            Step::ended(ms(4_000), 3),
        )
        .then(Trigger::AfterSamples(sample(4_100)), Step::Segment(3, ""))
        .then(Trigger::AfterSamples(sample(4_800)), Step::known(ms(4_700)));
    let (_, updates, _) = endpointed(script, with_turns(), 5_000, 1_600);
    assert_eq!(
        turns(&updates),
        [
            ("half cut end".to_owned(), Duration::ZERO, ms(3_000)),
            (String::new(), ms(3_900), ms(4_000)),
        ]
    );
}

/// Every fault check.
pub const ALL: &[(&str, fn())] = &[
    ("a02_backpressure_full", a02_backpressure_full),
    ("a02_push_wait_wakes_on_close", a02_push_wait_wakes_on_close),
    ("a03_late_events_discarded", a03_late_events_discarded),
    ("a04_failure_keeps_confirmed", a04_failure_keeps_confirmed),
    ("a05_readers", a05_readers),
    ("a05_slow_reader", a05_slow_reader),
    ("a07_endpointing", a07_endpointing),
    ("a08_turns", a08_turns),
    ("a09_slot_held_while_blocked", a09_slot_held_while_blocked),
    ("a09_backend_panic_isolated", a09_backend_panic_isolated),
    ("a09_single_slot", a09_single_slot),
    ("a10_events_while_idle", a10_events_while_idle),
    ("a11_bounded", a11_bounded),
    ("a14_deadline_while_opening", a14_deadline_while_opening),
    ("a14_start_waits_for_a_slot", a14_start_waits_for_a_slot),
    ("a15_feed_blocks", a15_feed_blocks),
    ("a19_expired_deadline_no_slot", a19_expired_deadline_no_slot),
];

/// Runs every fault check.
pub fn run_all() {
    for (name, check) in ALL {
        eprintln!("asr fault contract: {name}");
        check();
    }
}
