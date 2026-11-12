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
    time::{Duration, Instant},
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

/// A file whose existence makes a starting fake worker hang before its
/// handshake, as a native model load that never returns would.
const STALL: &str = "SPEECHKIT_FAKE_WORKER_STALL";

/// A file whose existence makes a running fake worker stop reading its
/// requests, as a frozen process would.
const DEAF: &str = "SPEECHKIT_FAKE_WORKER_DEAF";

/// Set to a number of seconds, makes the fake worker a wrapper, as a shell
/// script that runs the worker without `exec` is: it starts the worker as
/// its child, sharing its pipes, and waits for it. The child hangs before
/// its handshake for that many seconds, then exits.
const WRAP: &str = "SPEECHKIT_FAKE_WORKER_WRAP";
/// Set in the wrapped child: how long it hangs.
const LINGER: &str = "SPEECHKIT_FAKE_WORKER_LINGER";

/// Stdin, until the file at `flag` exists: then every read blocks.
struct Deaf<R> {
    inner: R,
    flag: Option<PathBuf>,
}

impl<R: std::io::Read> std::io::Read for Deaf<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.flag.as_ref().is_some_and(|flag| flag.exists()) {
            loop {
                std::thread::sleep(Duration::from_secs(60));
            }
        }
        self.inner.read(buf)
    }
}

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
    if let Some(seconds) = std::env::var_os(WRAP) {
        // Its stderr goes nowhere, so it does not hold the test's open.
        let _ = std::process::Command::new(std::env::current_exe().unwrap())
            .args(std::env::args_os().skip(1))
            .env_remove(WRAP)
            .env(LINGER, seconds)
            .stderr(std::process::Stdio::null())
            .status();
        std::process::exit(0);
    }
    if let Some(seconds) = std::env::var_os(LINGER) {
        let seconds: u64 = seconds.to_str().unwrap().parse().unwrap();
        std::thread::sleep(Duration::from_secs(seconds));
        std::process::exit(0);
    }
    if std::env::var_os(STALL).is_some_and(|stall| PathBuf::from(stall).exists()) {
        loop {
            std::thread::sleep(Duration::from_secs(60));
        }
    }
    let mut caps = AsrCapabilities::new(SampleRate::HZ_16000);
    caps.reports_partials = true;
    caps.accepts_hints = true;
    let backend = Scripted {
        caps,
        marker: PathBuf::from(marker),
    };
    let input = Deaf {
        inner: std::io::stdin().lock(),
        flag: std::env::var_os(DEAF).map(PathBuf::from),
    };
    let _ = serve(&backend, input, std::io::stdout());
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

/// The command that starts a fake worker keeping its files in `dir`.
fn command(dir: &tempfile::TempDir) -> WorkerCommand {
    WorkerCommand::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("worker_entry")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(MARKER, dir.path().join("marker"))
        .env(STALL, dir.path().join("stall"))
        .env(DEAF, dir.path().join("deaf"))
}

fn backend() -> (IsolatedAsr, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let backend = IsolatedAsr::spawn_command(command(&dir)).unwrap();
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

/// A request to a worker that stopped reading fails within the call
/// timeout, even when it is larger than the pipe can hold, and the worker
/// is killed: the write does not block the session forever.
#[test]
fn a_request_to_a_worker_that_stopped_reading_times_out() {
    let (backend, dir) = backend();
    let engine = AsrEngine::new(backend.with_call_timeout(Duration::from_millis(500)));
    // Start the worker, and leave it idle in the pool.
    assert!(run(&engine, &[]).unwrap().starts_with("pid"));
    std::fs::write(dir.path().join("deaf"), "").unwrap();
    // Far more than a pipe holds: the worker's read stops partway.
    let phrase = "x".repeat(4_000);
    let (done, finished) = std::sync::mpsc::channel();
    let started = Instant::now();
    std::thread::spawn(move || {
        let many = vec![phrase.as_str(); 100];
        let _ = done.send(run(&engine, &many).map(|_| engine));
    });
    let outcome = finished
        .recv_timeout(Duration::from_secs(20))
        .expect("the session blocked on the request");
    let Err(error) = outcome else {
        panic!("the session succeeded on a worker that stopped reading");
    };
    assert_retryable(&error);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
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

/// A worker that never finishes starting, as a model load that hangs in
/// native code would, is killed at its start timeout: the caller gets a
/// retryable error instead of waiting forever.
#[test]
fn a_worker_that_hangs_while_starting_is_killed() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("stall"), "").unwrap();
    let begun = Instant::now();
    let error =
        IsolatedAsr::spawn_command(command(&dir).with_start_timeout(Duration::from_millis(500)))
            .expect_err("the worker never started");
    assert!(
        begun.elapsed() < Duration::from_secs(10),
        "{:?}",
        begun.elapsed()
    );
    assert_retryable(&error);
    let detail = std::error::Error::source(&error).unwrap().to_string();
    assert!(detail.contains("did not start within"), "{detail}");
}

/// A worker command that wraps the worker, as a shell script without
/// `exec` does, leaves the worker running when it is killed, holding the
/// pipes. A start that times out still returns at its timeout, rather than
/// when that orphan lets go of them.
#[test]
fn a_wrapped_worker_that_hangs_while_starting_is_given_up_on() {
    let dir = tempfile::tempdir().unwrap();
    let begun = Instant::now();
    let error = IsolatedAsr::spawn_command(
        command(&dir)
            .env(WRAP, "15")
            .with_start_timeout(Duration::from_millis(500)),
    )
    .expect_err("the worker never started");
    assert!(
        begun.elapsed() < Duration::from_secs(10),
        "the start waited {:?} for the wrapped worker's pipes",
        begun.elapsed()
    );
    assert_retryable(&error);
}

/// A session that needs a new worker which hangs while starting fails,
/// and frees its slot, at the start timeout.
#[test]
fn a_session_on_a_worker_that_hangs_while_starting_fails() {
    let dir = tempfile::tempdir().unwrap();
    let backend =
        IsolatedAsr::spawn_command(command(&dir).with_start_timeout(Duration::from_millis(500)))
            .unwrap();
    let engine = AsrEngine::new(backend);
    // The first session takes the idle worker; the second needs a new one.
    let first = engine.start(RATE, hints(&[]), secs(60)).unwrap();
    std::fs::write(dir.path().join("stall"), "").unwrap();
    // Without a start timeout this would wait out the whole deadline, then
    // fail with DeadlineExceeded, and leave the session's thread blocked.
    let begun = Instant::now();
    let error = engine
        .start(RATE, hints(&[]), secs(20))
        .expect_err("the second worker never started");
    assert!(
        begun.elapsed() < Duration::from_secs(10),
        "{:?}",
        begun.elapsed()
    );
    assert_retryable(&error);
    let counted = counters(&engine);
    assert_eq!(counted["worker_start_failures"], 1, "{counted:?}");
    assert_eq!(counted["workers_killed"], 1, "{counted:?}");
    drop(first);
    assert!(eventually(Duration::from_secs(10), || engine
        .active_sessions()
        == 0));
}

/// The engine's counters, by name.
fn counters(engine: &AsrEngine) -> std::collections::HashMap<&'static str, u64> {
    engine.counters().into_iter().collect()
}

/// The counters count worker starts, crashes (including a worker that
/// exited while idle), and kills, so a crash loop or a hanging model shows
/// on the server's `/health`.
#[test]
fn counters_count_starts_crashes_and_kills() {
    let (backend, _dir) = backend();
    let engine = AsrEngine::new(backend.with_call_timeout(Duration::from_millis(500)));
    let idle = || eventually(Duration::from_secs(10), || engine.active_sessions() == 0);
    assert_eq!(counters(&engine)["workers_started"], 1);
    assert_retryable(&run(&engine, &["abort"]).expect_err("the worker crashed"));
    assert!(idle());
    assert_retryable(&run(&engine, &["hang"]).expect_err("the worker hung"));
    assert!(idle());
    run(&engine, &["exit"]).unwrap();
    assert!(idle());
    std::thread::sleep(Duration::from_millis(500));
    run(&engine, &[]).unwrap();
    let counted = counters(&engine);
    // The first, and one after each worker that went away.
    assert_eq!(counted["workers_started"], 4, "{counted:?}");
    assert_eq!(counted["workers_crashed"], 2, "{counted:?}");
    assert_eq!(counted["workers_killed"], 1, "{counted:?}");
    assert_eq!(counted["worker_start_failures"], 0, "{counted:?}");
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
