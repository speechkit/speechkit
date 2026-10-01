//! TTS contract checks that need [`FakeTts`].

use std::time::{Duration, Instant};

use speechkit::{
    SpeechError,
    tts::{TtsEngine, TtsLimits, TtsOptions},
};

use super::{TEXT, read_all};
use crate::{
    Gate,
    contract::SETTLE,
    eventually, secs,
    tts::{FakeTts, SAMPLES_PER_CHAR, TtsStep, TtsTrigger, level, samples_for},
};

fn engine(fake: FakeTts, session: TtsLimits) -> TtsEngine {
    TtsEngine::new(fake).with_limits(session)
}

/// T-01: audio is in chunk order, each chunk's audio at its own level.
pub fn t01_audio_follows_text_order() {
    let fake = FakeTts::plain();
    let stats = fake.stats();
    let engine = engine(fake, TtsLimits::default());
    let (session, mut output) = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .expect("start");
    session.push_text(TEXT).expect("push");
    session.close_text();
    let audio = read_all(&mut output).audio;
    let chunks = stats.chunks();
    let mut at = 0;
    for (index, chunk) in chunks.iter().enumerate() {
        let len = samples_for(chunk);
        assert!(
            audio[at..at + len]
                .iter()
                .all(|&s| (s - level(index)).abs() < 1e-6),
            "chunk {index}"
        );
        at += len;
    }
    assert_eq!(at, audio.len());
}

/// T-01: a full queue pauses synthesis; reading resumes it; nothing is lost.
pub fn t01_full_queue_pauses_synthesis() {
    let fake = FakeTts::plain();
    let stats = fake.stats();
    let limits = TtsLimits::default().with_output_queue(Duration::from_millis(20));
    let engine = engine(fake, limits);
    let text = TEXT.repeat(6);
    let (session, mut output) = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .expect("start");
    session.push_text(&text).expect("push");
    session.close_text();
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(output.queue_capacity(), 320);
    assert!(output.peak_queued_samples() <= 320);
    assert!(stats.chunks().len() < 12, "synthesis should be paused");
    assert!(session.result().is_none());
    let audio = read_all(&mut output).audio;
    let expected: usize = stats.chunks().iter().map(|c| samples_for(c)).sum();
    assert_eq!(audio.len(), expected);
    assert!(session.finish(secs(10)).is_ok());
}

/// T-01, T-02: audio made before anyone reads waits in the queue, and
/// `finish` waits for the reader to take it.
pub fn t02_finish_waits_for_the_reader() {
    let engine = engine(FakeTts::plain(), TtsLimits::default());
    let (session, mut output) = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .expect("start");
    session.push_text("Short.").expect("push");
    session.close_text();
    let unread = session.finish(Duration::from_millis(200));
    let failure = unread.expect_err("nobody read the audio");
    assert!(matches!(failure.error, SpeechError::DeadlineExceeded));
    assert!(failure.duration > Duration::ZERO, "{failure:?}");
    let read = read_all(&mut output);
    assert!(!read.audio.is_empty(), "queued audio is still delivered");
    assert!(read.result.expect("closed").is_err());
}

/// T-07: cancel returns promptly; the slot waits for the blocked call.
pub fn t07_cancel_waits_for_native_call() {
    let gate = Gate::new();
    let fake = FakeTts::new(vec![(
        TtsTrigger::BeforeChunk(0),
        TtsStep::BlockUntilReleased(gate.clone()),
    )]);
    let stats = fake.stats();
    let engine = engine(fake, TtsLimits::default());
    let (session, output) = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .expect("start");
    session.push_text(TEXT).expect("push");
    assert!(gate.wait_entered(1, SETTLE));
    let begun = Instant::now();
    session.cancel();
    assert!(session.finish(secs(10)).is_err());
    assert!(begun.elapsed() < Duration::from_secs(2));
    drop((session, output));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(engine.active_sessions(), 1);
    gate.release();
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    assert_eq!(stats.cancelled(), 1);
}

/// T-07: `start` waits for a slot within its deadline. A deadline that
/// passes while every slot is busy is `Capacity`; one that passes while the
/// backend opens is `DeadlineExceeded`, and the open keeps its slot until
/// it returns.
pub fn t07_start_waits_for_a_slot() {
    let engine = TtsEngine::new(FakeTts::plain()).with_max_sessions(1);
    let held = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .expect("start");
    let busy = engine.start(TtsOptions::default(), Duration::from_millis(100));
    assert!(matches!(busy, Err(SpeechError::Capacity)), "{busy:?}");
    let waiting = {
        let engine = engine.clone();
        std::thread::spawn(move || {
            engine
                .start(TtsOptions::default(), Duration::from_secs(10))
                .map(drop)
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    drop(held);
    assert!(waiting.join().expect("no panic").is_ok());
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));

    let gate = Gate::new();
    let slow = FakeTts::new(vec![(
        TtsTrigger::OnOpen,
        TtsStep::BlockUntilReleased(gate.clone()),
    )]);
    let engine = TtsEngine::new(slow).with_max_sessions(1);
    let opening = engine.start(TtsOptions::default(), Duration::from_millis(100));
    assert!(
        matches!(opening, Err(SpeechError::DeadlineExceeded)),
        "{opening:?}"
    );
    assert_eq!(engine.active_sessions(), 1);
    gate.release();
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    let passed = engine.start(TtsOptions::default(), Instant::now());
    assert!(matches!(passed, Err(SpeechError::DeadlineExceeded)));
    assert_eq!(engine.active_sessions(), 0);
}

/// T-04: empty text never opens a stream.
pub fn t04_empty_text_never_reaches_the_backend() {
    let fake = FakeTts::plain();
    let stats = fake.stats();
    let engine = engine(fake, TtsLimits::default());
    assert!(
        engine
            .synthesize("", TtsOptions::default(), secs(10))
            .is_err()
    );
    assert_eq!(stats.opened(), 0);
}

/// T-09: a failure after the first chunk reports that chunk's progress,
/// and the output still delivers the audio made before it.
pub fn t09_failure_after_first_chunk() {
    let fake = FakeTts::new(vec![(
        TtsTrigger::BeforeChunk(1),
        TtsStep::Fail(SpeechError::backend("fake-tts", true, "boom")),
    )]);
    let stats = fake.stats();
    let engine = engine(fake, TtsLimits::default());
    let failure = engine
        .synthesize(TEXT, TtsOptions::default(), secs(10))
        .expect_err("fails");
    let first = &stats.chunks()[0];
    assert!(failure.error.retryable());
    assert_eq!(
        failure.duration,
        speechkit::SampleRate::HZ_16000.duration_of(samples_for(first) as u64)
    );
    assert_eq!(failure.text_done, first.len());
}

/// A panicking backend fails the session, and the engine keeps working.
pub fn panic_is_isolated() {
    let fake = FakeTts::new(vec![(TtsTrigger::BeforeChunk(0), TtsStep::Panic)]);
    let engine = engine(fake, TtsLimits::default());
    let failure = engine
        .synthesize(TEXT, TtsOptions::default(), secs(10))
        .expect_err("panics");
    assert!(matches!(
        failure.error,
        SpeechError::Backend {
            retryable: false,
            ..
        }
    ));
    assert!(eventually(SETTLE, || engine.active_sessions() == 0));
    assert_eq!(SAMPLES_PER_CHAR, 160);
}

/// Every fault check.
pub const ALL: &[(&str, fn())] = &[
    ("t01_audio_follows_text_order", t01_audio_follows_text_order),
    (
        "t01_full_queue_pauses_synthesis",
        t01_full_queue_pauses_synthesis,
    ),
    (
        "t02_finish_waits_for_the_reader",
        t02_finish_waits_for_the_reader,
    ),
    (
        "t07_cancel_waits_for_native_call",
        t07_cancel_waits_for_native_call,
    ),
    ("t07_start_waits_for_a_slot", t07_start_waits_for_a_slot),
    (
        "t04_empty_text_never_reaches_the_backend",
        t04_empty_text_never_reaches_the_backend,
    ),
    (
        "t09_failure_after_first_chunk",
        t09_failure_after_first_chunk,
    ),
    ("panic_is_isolated", panic_is_isolated),
];
