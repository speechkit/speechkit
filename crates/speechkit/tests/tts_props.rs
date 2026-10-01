//! Property tests for synthesis sessions: random
//! interleavings of text, reads, finish, cancel, and drop.

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use proptest::prelude::*;
use speechkit::{
    RecvError,
    tts::{TtsEngine, TtsLimits, TtsOptions, TtsOutput, TtsSession, TtsUpdate},
};
use speechkit_testkit::{
    eventually,
    tts::{FakeTts, level, samples_for},
};

#[derive(Debug, Clone)]
enum Op {
    Push(&'static str),
    Close,
    Read,
    Finish,
    Cancel,
    DropSession,
    DropOutput,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => prop::sample::select(vec!["Hello there. ", "你好。", "How are you? ", "a", "Fine, thanks! ", "  "]).prop_map(Op::Push),
        1 => Just(Op::Close),
        4 => Just(Op::Read),
        1 => Just(Op::Finish),
        1 => Just(Op::Cancel),
        1 => Just(Op::DropSession),
        1 => Just(Op::DropOutput),
    ]
}

fn short() -> Instant {
    Instant::now() + Duration::from_millis(5)
}

fn run(ops: Vec<Op>) {
    let fake = FakeTts::plain();
    let stats = fake.stats();
    let limits = TtsLimits::default().with_output_queue(Duration::from_millis(30));
    let engine = TtsEngine::new(fake).with_limits(limits);
    let Ok((session, output)) = engine.start(TtsOptions::default(), Duration::from_secs(10)) else {
        return;
    };
    let (mut session, mut output): (Option<TtsSession>, Option<TtsOutput>) =
        (Some(session), Some(output));
    let mut audio = Vec::new();
    let mut finished_early = false;
    let mut result = None;
    for op in ops {
        if let Some(output) = output.as_ref() {
            // Invariant: the output queue never exceeds its bound.
            assert!(output.peak_queued_samples() <= output.queue_capacity());
        }
        match op {
            Op::Push(text) => {
                if let Some(session) = &session {
                    let _ = session.push_text(text);
                }
            }
            Op::Close => session.iter().for_each(TtsSession::close_text),
            Op::Read => {
                if let Some(reader) = output.as_mut() {
                    match reader.recv(short()) {
                        Ok(TtsUpdate::Audio(piece)) => audio.extend(piece),
                        Ok(TtsUpdate::Closed(closed)) => result = Some(closed),
                        _ => {}
                    }
                }
            }
            Op::Finish => {
                if let Some(session) = &session {
                    let _ = session.finish(short());
                    finished_early = true;
                }
            }
            Op::Cancel => session.iter().for_each(TtsSession::cancel),
            Op::DropSession => session = None,
            Op::DropOutput => output = None,
        }
    }
    if let Some(session) = &session {
        session.close_text();
    }
    if let Some(mut reader) = output.take() {
        while result.is_none() {
            match reader.recv(Duration::from_secs(5)) {
                Ok(TtsUpdate::Audio(piece)) => audio.extend(piece),
                Ok(TtsUpdate::Closed(closed)) => result = Some(closed),
                Ok(_) => {}
                // Invariant: a synthesis always ends, and `Closed` comes
                // once, last.
                Err(error) => {
                    assert_ne!(error, RecvError::Timeout, "stalled");
                    break;
                }
            }
        }
        assert!(reader.peak_queued_samples() <= reader.queue_capacity());
        if let Some(Ok(summary)) = &result
            && !finished_early
        {
            // Invariant: all audio equals the sum of the fake's chunks, in order.
            let chunks = stats.chunks();
            let expected: usize = chunks.iter().map(|c| samples_for(c)).sum();
            assert_eq!(audio.len(), expected);
            assert_eq!(
                speechkit::SampleRate::HZ_16000.frames_in(summary.duration),
                expected as u64
            );
            let mut at = 0;
            for (index, chunk) in chunks.iter().enumerate() {
                let len = samples_for(chunk);
                assert!(
                    audio[at..at + len]
                        .iter()
                        .all(|&s| (s - level(index)).abs() < 1e-6)
                );
                at += len;
            }
        }
    }
    drop(session);
    // Invariant: every slot is released.
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(
        std::env::var("SPEECHKIT_PROPTEST_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(500)
    ))]

    #[test]
    fn random_interleavings_keep_invariants(ops in prop::collection::vec(op(), 0..30)) {
        let (done, finished) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            run(ops);
            let _ = done.send(());
        });
        match finished.recv_timeout(Duration::from_secs(10)) {
            Ok(()) => worker.join().unwrap(),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Err(panic) = worker.join() {
                    std::panic::resume_unwind(panic);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => panic!("case deadlocked"),
        }
    }
}
