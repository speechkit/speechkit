//! Property tests: random operation sequences against the real engine and
//! a scripted fake backend.
//!
//! Cases default to 1 000; set `SPEECHKIT_PROPTEST_CASES` to change that
//! (the nightly job runs 10 000).
#![expect(
    clippy::panic,
    reason = "the harness asserts outside #[test] functions"
)]

use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

use proptest::prelude::*;
use speechkit::{
    RecvError,
    asr::{AsrEngine, AsrLimits, AsrResult, AsrSession, AsrUpdate, AsrUpdates, Segment},
};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    contract::asr::{RATE, options, same, tone},
    eventually,
};

#[derive(Debug, Clone)]
enum Op {
    Push(usize),
    TryPush(usize),
    Read,
    Recv,
    Finish,
    Cancel,
    Drop,
    AdvanceScript,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (0_usize..2_000).prop_map(Op::Push),
        4 => (0_usize..2_000).prop_map(Op::TryPush),
        1 => Just(Op::Read),
        3 => Just(Op::Recv),
        1 => Just(Op::Finish),
        1 => Just(Op::Cancel),
        1 => Just(Op::Drop),
        1 => Just(Op::AdvanceScript),
    ]
}

const LIMIT: usize = 2;

struct Live {
    session: AsrSession,
    updates: Option<AsrUpdates>,
    /// The segments the reader delivered.
    segments: Vec<Segment>,
    first: Option<AsrResult>,
}

impl Live {
    /// Invariant 1: once terminal, the result never changes.
    fn check_terminal(&mut self) {
        if let Some(now) = self.session.result() {
            match &self.first {
                Some(first) => assert!(same(first, now)),
                None => self.first = Some(now.clone()),
            }
        }
    }

    fn recv(&mut self, deadline: Instant) {
        if let Some(updates) = &mut self.updates
            && let Ok(AsrUpdate::Segment(segment)) = updates.recv(deadline)
        {
            self.segments.push(segment);
        }
    }

    /// Invariant 2: a reader delivers every segment of the final transcript
    /// exactly once, in order, whenever it started reading.
    fn close(mut self, gate: &Gate) {
        gate.release();
        let result = self.session.finish(Duration::from_secs(5));
        self.check_terminal();
        if let Some(mut updates) = self.updates.take() {
            loop {
                match updates.recv(Duration::from_secs(5)) {
                    Ok(AsrUpdate::Segment(segment)) => self.segments.push(segment),
                    // `Closed` ends the updates; an earlier `Recv` may have
                    // read it already.
                    Ok(AsrUpdate::Closed(_)) | Err(RecvError::Closed) => break,
                    Ok(_) => {}
                    Err(error) => panic!("the reader never closed: {error}"),
                }
            }
            let segments = match result.as_ref() {
                Ok(outcome) => &outcome.segments,
                Err(failure) => &failure.confirmed.segments,
            };
            assert_eq!(&self.segments, segments);
        }
    }
}

fn run(ops: Vec<Op>) {
    let gate = Gate::new();
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(500), Step::Partial(0, "a"))
            .then(Trigger::AfterSamples(1_000), Step::Segment(0, "one"))
            .then(
                Trigger::AfterSamples(1_500),
                Step::BlockUntilReleased(gate.clone()),
            )
            .then(Trigger::AfterSamples(2_000), Step::Partial(1, "tw"))
            .then(Trigger::AfterSamples(3_000), Step::Segment(1, "two"))
            .then(Trigger::OnFinish, Step::Segment(2, "end")),
    );
    let session = AsrLimits::default().with_input_queue(Duration::from_millis(200));
    let engine = AsrEngine::new(fake)
        .with_max_sessions(LIMIT)
        .with_limits(session);
    let mut live: Option<Live> = None;
    let mut retired: Vec<Live> = Vec::new();
    let short = || Instant::now() + Duration::from_millis(5);
    for op in ops {
        if live.is_none() {
            // A short deadline: while retired sessions hold every slot,
            // `start` waits for one, and gives up with `Capacity`.
            live = engine
                .start(RATE, options(), short())
                .ok()
                .map(|session| Live {
                    session,
                    updates: None,
                    segments: Vec::new(),
                    first: None,
                });
        }
        // Invariant 3: never more sessions than the limit.
        assert!(engine.active_sessions() <= LIMIT);
        let Some(current) = live.as_mut() else {
            continue;
        };
        match op {
            Op::Push(n) => {
                let _ = current.session.push(tone(n), short());
            }
            Op::TryPush(n) => {
                let _ = current.session.try_push(tone(n));
            }
            Op::Read => {
                if current.updates.is_none() {
                    current.updates = Some(current.session.updates());
                }
            }
            Op::Recv => current.recv(short()),
            Op::Finish => {
                let _ = current.session.finish(short());
            }
            Op::Cancel => current.session.cancel(),
            Op::Drop => {
                if let Some(done) = live.take() {
                    retired.push(done);
                }
            }
            Op::AdvanceScript => gate.release(),
        }
        if let Some(current) = live.as_mut() {
            current.check_terminal();
        }
    }
    retired.extend(live);
    for done in retired {
        done.close(&gate);
    }
    gate.release();
    // Invariant 3: every worker exits and frees its slot.
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
}

fn cases() -> u32 {
    std::env::var("SPEECHKIT_PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases()))]

    #[test]
    fn random_operations_keep_invariants(ops in prop::collection::vec(op(), 1..40)) {
        // Invariant 4: no case panics or deadlocks; a watchdog bounds it.
        let (done, finished) = mpsc::channel();
        let worker = std::thread::spawn(move || {
            run(ops);
            let _ = done.send(());
        });
        match finished.recv_timeout(Duration::from_secs(5)) {
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
