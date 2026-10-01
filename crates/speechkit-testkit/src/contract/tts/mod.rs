//! The TTS contract suite: rules `T-01` to `T-11`.
//!
//! Each `tXX_*` function checks one rule against engines built by `make`
//! and works with any backend. Rules that need a misbehaving backend are
//! checked by [`faults`] with [`FakeTts`](crate::tts::FakeTts). `T-05`
//! needs a process-wide log capture, so [`t05_no_text_logging`] takes the
//! captured logs from its caller and is not part of [`run_tts_contract`].

use std::time::Duration;

use speechkit::{
    RecvError, SampleRate, SpeechError,
    tts::{Mark, TtsEngine, TtsLimits, TtsOptions, TtsOutput, TtsResult, TtsSession, TtsUpdate},
};

use super::SETTLE;
use crate::{eventually, secs};

/// Text every check synthesizes.
pub const TEXT: &str = "One sentence here. Another one follows! 第三句。";

fn wait_idle(engine: &TtsEngine) {
    assert!(
        eventually(SETTLE, || engine.active_sessions() == 0),
        "slots were not released"
    );
}

/// Everything an output delivered, in order.
#[derive(Debug, Default)]
pub struct Read {
    /// The audio, concatenated.
    pub audio: Vec<f32>,
    /// The marks, each with the audio samples read before it.
    pub marks: Vec<(Mark, usize)>,
    /// The result `Closed` carried.
    pub result: Option<TtsResult>,
}

/// Reads an output to its end, checking that `Closed` comes last (T-03).
pub fn read_all(output: &mut TtsOutput) -> Read {
    let mut read = Read::default();
    loop {
        match output.recv(secs(60)) {
            Ok(TtsUpdate::Audio(samples)) => read.audio.extend_from_slice(&samples),
            Ok(TtsUpdate::Mark(mark)) => read.marks.push((mark, read.audio.len())),
            Ok(TtsUpdate::Closed(result)) => {
                read.result = Some(result);
                break;
            }
            Ok(other) => panic_on(&format!("an unknown update {other:?}")),
            Err(error) => panic_on(&format!("the output ended without Closed: {error}")),
        }
    }
    assert_eq!(output.try_recv().err(), Some(RecvError::Closed));
    read
}

fn panic_on(what: &str) -> ! {
    panic!("{what}")
}

fn start(engine: &TtsEngine, opts: TtsOptions) -> (TtsSession, TtsOutput) {
    engine.start(opts, Duration::from_secs(10)).expect("start")
}

fn run(engine: &TtsEngine, opts: TtsOptions, text: &str) -> (TtsSession, Read) {
    let (session, mut output) = start(engine, opts);
    session.push_text(text).expect("push");
    session.close_text();
    let read = read_all(&mut output);
    (session, read)
}

/// The duration of `samples` at `rate`, as the engine reports it.
fn duration(rate: SampleRate, samples: usize) -> Duration {
    rate.duration_of(samples as u64)
}

/// T-01: mono audio at the session's rate, and nothing dropped while the
/// output queue stays within its bound.
pub fn t01_output(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let native = engine.capabilities().sample_rate;
    let (session, read) = run(&engine, TtsOptions::default(), TEXT);
    assert_eq!(session.sample_rate(), native);
    assert!(!read.audio.is_empty());
    assert!(
        read.audio
            .iter()
            .all(|s| s.is_finite() && (-1.0..=1.0).contains(s))
    );
    let summary = read.result.expect("closed").expect("success");
    assert_eq!(summary.duration, duration(native, read.audio.len()));
    let native_len = read.audio.len();
    // At least twice or at most half the native rate, so a missing
    // resample shows through the tolerance below.
    let target = if native.hz() > 24_000 {
        SampleRate::HZ_16000
    } else {
        SampleRate::HZ_48000
    };
    let (session, read) = run(
        &engine,
        TtsOptions::default().with_sample_rate(target),
        TEXT,
    );
    assert_eq!(session.sample_rate(), target);
    // The two runs are separate syntheses, and some models are stochastic
    // (VITS and Piper sample their durations), so lengths differ by several
    // percent: piper_en once came out 6% off. The rates differ by a factor
    // of two or more, so 15% still shows the output was resampled.
    let expected = native_len as f64 * f64::from(target.hz()) / f64::from(native.hz());
    #[expect(clippy::cast_precision_loss, reason = "sample counts are small")]
    let got = read.audio.len() as f64;
    assert!(
        (got - expected).abs() <= expected * 0.15 + 4.0,
        "{got} vs {expected}"
    );

    // Unread audio waits in the bounded queue; synthesis waits for room.
    let (session, mut output) = start(&engine, TtsOptions::default());
    session.push_text(&TEXT.repeat(8)).expect("push");
    session.close_text();
    std::thread::sleep(Duration::from_millis(300));
    assert!(output.peak_queued_samples() <= output.queue_capacity());
    let read = read_all(&mut output);
    let summary = read.result.expect("closed").expect("success");
    assert_eq!(summary.duration, duration(native, read.audio.len()));
    assert!(output.peak_queued_samples() <= output.queue_capacity());
    wait_idle(&engine);
}

/// T-02: `start` returns the output with the session. Dropping the session
/// after `close_text` changes nothing; dropping it before, or dropping the
/// output, cancels the synthesis.
pub fn t02_drop_rules(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, mut output) = start(&engine, TtsOptions::default());
    session.push_text(TEXT).expect("push");
    session.close_text();
    drop(session);
    let read = read_all(&mut output);
    assert!(read.result.expect("closed").is_ok());
    assert!(!read.audio.is_empty());

    let (session, mut output) = start(&engine, TtsOptions::default());
    session.push_text(TEXT).expect("push");
    drop(session);
    let read = read_all(&mut output);
    let failure = read.result.expect("closed").expect_err("cancelled");
    assert!(matches!(failure.error, SpeechError::Cancelled), "{failure}");

    let (session, output) = start(&engine, TtsOptions::default());
    session.push_text(TEXT).expect("push");
    drop(output);
    let failure = session.finish(secs(10)).expect_err("cancelled");
    assert!(matches!(failure.error, SpeechError::Cancelled), "{failure}");
    assert!(matches!(
        session.push_text("more"),
        Err(SpeechError::Closed)
    ));
    wait_idle(&engine);
}

/// T-03: a mark follows the last audio of its text; the marks are in text
/// order and match the summary; `Closed` is last.
pub fn t03_marks(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, read) = run(&engine, TtsOptions::default(), TEXT);
    let rate = session.sample_rate();
    let summary = read.result.expect("closed").expect("success");
    let marks: Vec<Mark> = read.marks.iter().map(|(mark, _)| mark.clone()).collect();
    assert_eq!(marks, summary.marks);
    assert!(!marks.is_empty());
    for (mark, before) in &read.marks {
        assert!(
            mark.audio.end <= duration(rate, *before),
            "{mark:?} came before its audio ({before} samples)"
        );
    }
    for pair in marks.windows(2) {
        assert!(pair[0].text.end <= pair[1].text.start);
        assert!(pair[0].audio.end <= pair[1].audio.start);
    }
    assert_eq!(marks.last().map(|m| m.text.end), Some(TEXT.len()));
    assert!(
        marks
            .last()
            .is_some_and(|m| m.audio.end <= summary.duration)
    );
    wait_idle(&engine);
}

/// T-04: bad options, an unknown voice, empty text, and text over the
/// limit are rejected before synthesis starts.
pub fn t04_validation(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let empty = engine
        .synthesize("  ", TtsOptions::default(), secs(10))
        .expect_err("empty text");
    assert!(matches!(empty.error, SpeechError::InvalidInput(_)));
    assert!(matches!(
        engine.start(
            TtsOptions::default().with_voice("no-such-voice"),
            Duration::from_secs(10)
        ),
        Err(SpeechError::InvalidInput(_))
    ));
    let speed = match &engine.capabilities().speed {
        Some(range) => range.end() * 2.0,
        None => 1.5,
    };
    assert!(
        engine
            .start(
                TtsOptions::default().with_speed(speed),
                Duration::from_secs(10)
            )
            .is_err()
    );
    let (session, output) = start(&engine, TtsOptions::default());
    let limit = engine.limits().max_text_chars;
    assert!(matches!(
        session.push_text(&"x".repeat(limit + 1)),
        Err(SpeechError::InvalidInput(_))
    ));
    session.close_text();
    drop(output);
    let result = session.finish(secs(10));
    assert!(result.is_err());
    wait_idle(&engine);
}

/// T-05: synthesized text never appears in logs above TRACE. `logs`
/// returns everything logged at DEBUG and above since the process started.
pub fn t05_no_text_logging(make: &dyn Fn() -> TtsEngine, logs: &dyn Fn() -> String) {
    let engine = make();
    let secret_text = "The quiet giraffe whispers passwords.";
    let (session, _) = run(&engine, TtsOptions::default(), secret_text);
    let _ = session.finish(secs(60));
    let (failing, output) = start(&engine, TtsOptions::default());
    failing
        .push_text("Another private sentence.")
        .expect("push");
    failing.cancel();
    drop((failing, output));
    wait_idle(&engine);
    let logged = logs();
    assert!(
        logged.contains("synthesis"),
        "the capture saw nothing: {logged}"
    );
    assert!(!logged.contains("giraffe"), "{logged}");
    assert!(!logged.contains("private sentence"), "{logged}");
}

/// T-06: `finish` returns the same result every time.
pub fn t06_finish_idempotent(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, _) = run(&engine, TtsOptions::default(), TEXT);
    let first = session.finish(secs(60));
    let second = session.finish(secs(60));
    assert!(first.is_ok());
    assert_eq!(first.as_ref().ok(), second.as_ref().ok());
    assert_eq!(
        first.as_ref().ok(),
        session.result().expect("a result").as_ref().ok()
    );
    wait_idle(&engine);
}

/// T-07: cancelling stops synthesis promptly and frees the slot.
pub fn t07_cancel_releases_after_backend(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, _output) = start(&engine, TtsOptions::default());
    session.push_text(&TEXT.repeat(4)).expect("push");
    session.cancel();
    let result = session.finish(secs(10));
    assert!(
        matches!(result.as_ref(), Err(failure) if matches!(failure.error, SpeechError::Cancelled))
    );
    assert!(matches!(
        session.push_text("more"),
        Err(SpeechError::Closed)
    ));
    let dropped = start(&engine, TtsOptions::default());
    dropped.0.push_text(TEXT).expect("push");
    drop(dropped);
    wait_idle(&engine);
}

/// T-08: text pushed piece by piece gives marks whose ranges cover it.
pub fn t08_incremental_text(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, mut output) = start(&engine, TtsOptions::default());
    let mut pushed = String::new();
    for piece in TEXT.split_inclusive(' ') {
        session.push_text(piece).expect("push");
        pushed.push_str(piece);
    }
    session.close_text();
    let read = read_all(&mut output);
    let summary = read.result.expect("closed").expect("success");
    assert!(summary.marks.len() >= 3, "{:?}", summary.marks);
    for mark in &summary.marks {
        assert!(!pushed[mark.text.clone()].trim().is_empty());
    }
    assert_eq!(summary.marks.last().map(|m| m.text.end), Some(TEXT.len()));
    wait_idle(&engine);
}

/// T-09: a failure keeps its progress and is never empty audio.
pub fn t09_failure_keeps_progress(make: &dyn Fn() -> TtsEngine) {
    let engine = make();
    let (session, mut output) = start(&engine, TtsOptions::default());
    session.push_text(TEXT).expect("push");
    // The text stays open, so the synthesis cannot succeed before the
    // cancel.
    let mut heard = 0;
    while let Ok(update) = output.recv(secs(60)) {
        if let TtsUpdate::Audio(samples) = update {
            heard = samples.len();
            break;
        }
    }
    session.cancel();
    let failure = session.finish(secs(10)).expect_err("cancelled");
    assert!(failure.duration >= duration(session.sample_rate(), heard));
    assert!(failure.text_done <= TEXT.len());
    wait_idle(&engine);
}

/// T-10: the default limits.
pub fn t10_defaults(_: &dyn Fn() -> TtsEngine) {
    let limits = TtsLimits::default();
    assert_eq!(limits.output_queue, Duration::from_secs(2));
    assert_eq!(limits.max_text_chars, 10_000);
    assert_eq!(limits.chunk_chars, 300);
    assert_eq!(
        TtsEngine::new(crate::tts::FakeTts::plain()).max_sessions(),
        8
    );
}

/// T-11: a synthesis that ends on its own has freed its slot by the time
/// `synthesize` returns, so back-to-back syntheses never wait.
///
/// The other slots are held open, so each round needs the slot the
/// previous round used. The race this guards against is a few
/// instructions wide, so the check repeats it; fewer rounds than A-09,
/// since real backends synthesize slowly.
pub fn t11_slot_free_after_finish(make: &dyn Fn() -> TtsEngine) {
    const ROUNDS: usize = 20;
    let engine = make();
    let limit = engine.max_sessions();
    let held: Vec<_> = (1..limit)
        .map(|_| start(&engine, TtsOptions::default()))
        .collect();
    for round in 0..ROUNDS {
        engine
            .synthesize("Hello.", TtsOptions::default(), secs(60))
            .unwrap_or_else(|failure| panic_on(&format!("round {round}: {}", failure.error)));
        assert_eq!(
            engine.active_sessions(),
            limit - 1,
            "round {round}: the slot was still held after synthesize returned"
        );
    }
    drop(held);
    wait_idle(&engine);
}

/// A generic check.
pub type Check = fn(&dyn Fn() -> TtsEngine);

/// Every generic check except T-05.
pub const GENERIC: &[(&str, Check)] = &[
    ("t01_output", t01_output),
    ("t02_drop_rules", t02_drop_rules),
    ("t03_marks", t03_marks),
    ("t04_validation", t04_validation),
    ("t06_finish_idempotent", t06_finish_idempotent),
    (
        "t07_cancel_releases_after_backend",
        t07_cancel_releases_after_backend,
    ),
    ("t08_incremental_text", t08_incremental_text),
    ("t09_failure_keeps_progress", t09_failure_keeps_progress),
    ("t10_defaults", t10_defaults),
    ("t11_slot_free_after_finish", t11_slot_free_after_finish),
];

/// Runs every generic check against engines built by `make`.
pub fn run_tts_contract(make: impl Fn() -> TtsEngine) {
    for (name, check) in GENERIC {
        eprintln!("tts contract: {name}");
        check(&make);
    }
}

pub mod faults;
