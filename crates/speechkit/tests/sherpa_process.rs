//! `IsolatedAsr` runs sessions on a pool of worker processes: a worker
//! that aborts fails only its own session, sessions run in parallel, idle
//! workers are reused, and events reach the session while it is idle.
//!
//! The fake worker is this test binary, started again with an environment
//! variable that makes `worker_entry` serve the `Scripted` backend. What a
//! stream does comes from its session's hints.
#![cfg(feature = "sherpa")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test helpers fail the calling test"
)]

use std::{
    path::PathBuf,
    sync::{Arc, Barrier},
    thread::JoinHandle,
    time::Duration,
};

use speechkit::sherpa::process::{IsolatedAsr, WorkerCommand, serve};
use speechkit::{
    SampleRate, SpeechError,
    asr::{
        AsrBackend, AsrCapabilities, AsrEngine, AsrEvent, AsrEvents, AsrOptions, AsrStream,
        AsrUpdate, Partial, Segment, UtteranceId,
    },
};
use speechkit_testkit::{
    contract::asr::{RATE, run_asr_contract, tone},
    eventually, secs,
};

const MARKER: &str = "SPEECHKIT_FAKE_WORKER_MARKER";

/// A backend whose streams follow their session's hints:
///
/// - `abort`: abort the worker at 3 200 samples, unless the marker file
///   exists; it creates the marker first, so only one worker aborts;
/// - `late`: send a segment from a thread, 100 ms after opening;
/// - `fail`: report a failure from a thread, 100 ms after opening;
/// - `hang`: never return from `accept`;
/// - `exit`: exit the worker 100 ms after the stream is closed, while it
///   is idle in the pool.
///
/// Every stream commits a segment naming its worker's process ID at the
/// end.
struct Scripted {
    caps: AsrCapabilities,
    marker: PathBuf,
}

struct Stream {
    events: AsrEvents,
    marker: PathBuf,
    abort: bool,
    hang: bool,
    exit: bool,
    seen: usize,
    thread: Option<JoinHandle<()>>,
}

impl AsrBackend for Scripted {
    fn name(&self) -> &'static str {
        "scripted"
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(
        &self,
        options: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        let has = |hint: &str| options.hints.iter().any(|h| h == hint);
        let thread = if has("late") || has("fail") {
            let (events, fail) = (events.clone(), has("fail"));
            Some(std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                if fail {
                    events.fail(SpeechError::backend("scripted", true, "dropped"));
                } else {
                    events.send(AsrEvent::Segment(Segment {
                        utterance: UtteranceId(0),
                        text: "late".into(),
                        start: Duration::ZERO,
                        end: Duration::ZERO,
                    }));
                }
            }))
        } else {
            None
        };
        Ok(Box::new(Stream {
            events,
            marker: self.marker.clone(),
            abort: has("abort"),
            hang: has("hang"),
            exit: has("exit"),
            seen: 0,
            thread,
        }))
    }
}

impl AsrStream for Stream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.seen += samples.len();
        if self.hang {
            loop {
                std::thread::sleep(Duration::from_secs(1));
            }
        }
        if self.abort && self.seen >= 3_200 && !self.marker.exists() {
            std::fs::write(&self.marker, "crashed once").unwrap();
            std::process::abort();
        }
        self.events.send(AsrEvent::Partial(Partial {
            utterance: UtteranceId(1),
            text: format!("{} samples", self.seen),
        }));
        Ok(())
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        self.events.send(AsrEvent::Segment(Segment {
            utterance: UtteranceId(1),
            text: format!("pid{}", std::process::id()),
            start: Duration::ZERO,
            end: Duration::from_millis(100),
        }));
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        if self.exit {
            std::thread::spawn(|| {
                std::thread::sleep(Duration::from_millis(100));
                std::process::exit(0);
            });
        }
    }
}

/// The worker's entry point; a no-op unless the marker variable is set.
#[test]
fn worker_entry() {
    let Some(marker) = std::env::var_os(MARKER) else {
        return;
    };
    let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
    caps.reports_partials = true;
    caps.accepts_hints = true;
    let backend = Scripted {
        caps,
        marker: PathBuf::from(marker),
    };
    let _ = serve(&backend, std::io::stdin().lock(), std::io::stdout());
    // Leave before the test harness prints its summary on stdout.
    std::process::exit(0);
}

/// An engine on fake workers, and the directory holding the marker.
///
/// A new worker is a new process of this test binary, which can take a
/// while to start when every test runs at once, so starts get 60 s.
fn engine() -> (AsrEngine, tempfile::TempDir) {
    let (backend, dir) = backend();
    (AsrEngine::new(backend), dir)
}

fn backend() -> (IsolatedAsr, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let command = WorkerCommand::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("worker_entry")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MARKER, dir.path().join("marker"));
    let backend = IsolatedAsr::spawn_command(command).unwrap();
    assert!(backend.capabilities().reports_partials);
    assert_eq!(backend.name(), "scripted@worker");
    (backend, dir)
}

fn hints(hints: &[&str]) -> AsrOptions {
    AsrOptions::default().with_hints(hints.iter().copied())
}

/// Runs a session with `hints` on 0.4 s of audio, and returns its text.
fn run(engine: &AsrEngine, with: &[&str]) -> Result<String, SpeechError> {
    let session = engine.start(RATE, hints(with), secs(60))?;
    for _ in 0..4 {
        let _ = session.push(tone(1_600), secs(10));
    }
    Ok(session.finish(secs(10))?.text())
}

fn assert_retryable(error: &SpeechError) {
    match error {
        SpeechError::Backend { retryable, .. } => assert!(retryable, "{error}"),
        other => panic!("expected a retryable backend error, got {other}"),
    }
}

#[test]
fn worker_crash_fails_one_session_and_the_next_succeeds() {
    let (engine, _dir) = engine();
    let error = run(&engine, &["abort"]).expect_err("the worker crashed");
    assert_retryable(&error);
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
    assert!(run(&engine, &[]).unwrap().starts_with("pid"));
}

#[test]
fn a_crash_fails_only_the_session_on_that_worker() {
    let (engine, _dir) = engine();
    let barrier = Arc::new(Barrier::new(2));
    let session = |with: &'static [&'static str]| {
        let (engine, barrier) = (engine.clone(), barrier.clone());
        std::thread::spawn(move || {
            let session = engine.start(RATE, hints(with), secs(60)).unwrap();
            // Both sessions are open, on two workers, before either runs.
            barrier.wait();
            for _ in 0..4 {
                let _ = session.push(tone(1_600), secs(10));
            }
            session.finish(secs(10))
        })
    };
    let crashing = session(&["abort"]);
    let surviving = session(&[]);
    let failure = crashing.join().unwrap().expect_err("the worker crashed");
    assert_retryable(&failure.error);
    let text = surviving.join().unwrap().unwrap().text();
    assert!(text.starts_with("pid"), "{text}");
}

#[test]
fn idle_workers_are_reused_and_busy_ones_are_not() {
    let (engine, _dir) = engine();
    let first = run(&engine, &[]).unwrap();
    assert_eq!(run(&engine, &[]).unwrap(), first, "one worker, reused");
    let a = engine.start(RATE, hints(&[]), secs(60)).unwrap();
    let b = engine.start(RATE, hints(&[]), secs(60)).unwrap();
    let (a, b) = (a.finish(secs(10)).unwrap(), b.finish(secs(10)).unwrap());
    assert_ne!(a.text(), b.text(), "concurrent sessions on two workers");
}

#[test]
fn events_and_failures_arrive_while_the_session_is_idle() {
    let (engine, _dir) = engine();
    let session = engine.start(RATE, hints(&["late"]), secs(60)).unwrap();
    let mut updates = session.updates();
    match updates.recv(secs(10)).unwrap() {
        AsrUpdate::Segment(segment) => assert_eq!(segment.text, "late"),
        other => panic!("expected the late segment, got {other:?}"),
    }
    drop(session);
    let session = engine.start(RATE, hints(&["fail"]), secs(60)).unwrap();
    let result = session.wait(secs(10)).expect("the stream failed");
    assert_retryable(&result.unwrap_err().error);
}

#[test]
fn a_hung_worker_is_killed_after_its_call_timeout() {
    let (backend, _dir) = backend();
    let engine = AsrEngine::new(backend.with_call_timeout(Duration::from_millis(500)));
    let error = run(&engine, &["hang"]).expect_err("the worker hung");
    assert_retryable(&error);
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
    assert!(run(&engine, &[]).unwrap().starts_with("pid"));
}

#[test]
fn cancelling_a_session_on_a_hung_worker_frees_its_slot() {
    let (engine, _dir) = engine();
    let session = engine.start(RATE, hints(&["hang"]), secs(60)).unwrap();
    let _ = session.push(tone(1_600), secs(10));
    std::thread::sleep(Duration::from_millis(200));
    session.cancel();
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
    assert!(run(&engine, &[]).unwrap().starts_with("pid"));
}

#[test]
fn a_worker_that_exited_while_idle_is_not_used() {
    let (engine, _dir) = engine();
    let first = run(&engine, &["exit"]).unwrap();
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
    std::thread::sleep(Duration::from_millis(500));
    let second = run(&engine, &[]).unwrap();
    assert_ne!(first, second, "a new worker");
}

#[test]
fn passes_the_asr_contract() {
    run_asr_contract(|| engine().0);
}

#[test]
fn a_missing_program_is_a_retryable_error() {
    let error = IsolatedAsr::spawn_command(WorkerCommand::new("/nonexistent/speechkit-worker"))
        .unwrap_err();
    assert!(error.retryable(), "{error}");
}
