//! The parent side: an `AsrBackend` backed by a pool of worker processes.

use std::{
    ffi::OsString,
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, RecvTimeoutError},
    },
    time::{Duration, Instant},
};

use crate::{
    SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvents, AsrOptions, AsrStream},
};

use super::{
    lock,
    protocol::{Request, Response, WireError, read_frame, skip_to_handshake, write_frame},
};

const NAME: &str = "speechkit-worker";

/// How long one request may take by default before the worker is taken
/// as hung and killed.
const CALL_TIMEOUT: Duration = Duration::from_secs(60);
/// How long a worker may take by default to load its model and send its
/// handshake before it is taken as hung and killed.
const START_TIMEOUT: Duration = Duration::from_secs(120);
/// How long a worker may still take to answer once its session has
/// stopped, such as after a cancel.
const STOPPED_GRACE: Duration = Duration::from_secs(2);
/// How often a waiting call checks its session.
const POLL: Duration = Duration::from_millis(100);

fn crashed(detail: impl std::fmt::Display) -> SpeechError {
    SpeechError::backend(NAME, true, format!("the worker process failed: {detail}"))
}

/// How to start a worker process.
///
/// A worker that hangs or breaks is killed, which kills that process only.
/// A wrapper, such as a shell script, should `exec` the worker rather than
/// run it as a child: killing the wrapper would leave the worker running.
/// The caller does not wait for it either way.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Vec<(OsString, OsString)>,
    start_timeout: Duration,
}

impl WorkerCommand {
    /// Runs `program`.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
            start_timeout: START_TIMEOUT,
        }
    }

    /// Sets how long a worker may take to start: to load its model and
    /// send its handshake. A worker that takes longer is taken as hung and
    /// killed, and the start fails with a retryable
    /// [`SpeechError::Backend`]. It applies to every worker the pool
    /// starts, the first one included. Default: 120 s.
    #[must_use]
    pub fn with_start_timeout(mut self, timeout: Duration) -> Self {
        self.start_timeout = timeout;
        self
    }

    /// Adds an argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Sets an environment variable.
    #[must_use]
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }
}

/// The answer to a request, or the worker's end.
enum Reply {
    Done,
    Error(WireError),
    /// The worker exited, or its pipes broke.
    Gone(String),
    /// The worker is running but did not take or answer the request in
    /// time.
    Hung(String),
}

/// What a pool counts about its workers, for
/// [`counters`](AsrBackend::counters).
#[derive(Debug, Default)]
struct Counts {
    started: AtomicU64,
    start_failures: AtomicU64,
    /// Workers that exited on their own, in use or idle.
    crashed: AtomicU64,
    /// Workers killed for not starting or answering in time.
    killed: AtomicU64,
}

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

/// Where the reader thread sends the events of the stream in progress.
type Link = Arc<Mutex<Option<AsrEvents>>>;

/// A running worker, serving one stream at a time.
struct Worker {
    process: Child,
    /// Encoded requests for the writer thread. A blocking write to a worker
    /// that stopped reading would hold the caller past any timeout, so the
    /// caller waits for [`written`](Self::written) instead.
    requests: Option<mpsc::Sender<Vec<u8>>>,
    /// The outcome of each write the writer thread made.
    written: mpsc::Receiver<std::io::Result<()>>,
    replies: mpsc::Receiver<Reply>,
    link: Link,
    /// The worker is gone, or a call to it failed.
    broken: bool,
    counts: Arc<Counts>,
}

impl Worker {
    /// Starts a worker, counting whether it started.
    fn start(
        command: &WorkerCommand,
        counts: &Arc<Counts>,
    ) -> Result<(Self, String, AsrCapabilities), SpeechError> {
        let started = Self::launch(command, counts);
        bump(if started.is_ok() {
            &counts.started
        } else {
            &counts.start_failures
        });
        started
    }

    fn launch(
        command: &WorkerCommand,
        counts: &Arc<Counts>,
    ) -> Result<(Self, String, AsrCapabilities), SpeechError> {
        let program = &command.program;
        let mut process = Command::new(program)
            .args(&command.args)
            .envs(command.envs.iter().map(|(k, v)| (k, v)))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| crashed(format!("cannot start {}: {e}", program.display())))?;
        let (Some(input), Some(output)) = (process.stdin.take(), process.stdout.take()) else {
            let _ = process.kill();
            let _ = process.wait();
            return Err(crashed("no pipes to the worker"));
        };
        // The reader thread reads the handshake too, so waiting for it can
        // time out: the worker loads its model before it answers, and a
        // load that hangs must not block the caller forever.
        let link: Link = Arc::default();
        let (send, replies) = mpsc::channel();
        let (send_hello, hello) = mpsc::channel();
        // Neither pipe thread is waited for: a worker killed while a child
        // of its own holds its pipes, such as a wrapper script that does not
        // `exec` the worker, would hold the caller until that child exits.
        // The threads end once the pipes close.
        std::thread::Builder::new()
            .name("speechkit-worker-reader".into())
            .spawn({
                let link = link.clone();
                move || {
                    let mut output = BufReader::new(output);
                    let first = skip_to_handshake(&mut output)
                        .and_then(|()| read_frame::<Response>(&mut output));
                    let started = matches!(first, Ok(Some(Response::Hello { .. })));
                    if send_hello.send(first).is_ok() && started {
                        read(output, &link, &send);
                    }
                }
            })
            .map_err(|error| {
                kill(&mut process);
                crashed(error)
            })?;
        let timeout = command.start_timeout;
        let hello = match hello.recv_timeout(timeout) {
            Ok(hello) => hello,
            Err(RecvTimeoutError::Timeout) => {
                bump(&counts.killed);
                let status = kill(&mut process);
                return Err(crashed(format!(
                    "it did not start within {timeout:?} ({status})"
                )));
            }
            Err(RecvTimeoutError::Disconnected) => {
                let status = kill(&mut process);
                return Err(crashed(format!("no hello from the worker ({status})")));
            }
        };
        let (name, caps) = match hello {
            Ok(Some(Response::Hello { name, caps })) => match caps.try_into() {
                Ok(caps) => (name, caps),
                Err(error) => {
                    kill(&mut process);
                    return Err(error);
                }
            },
            Ok(Some(Response::Error(error))) => {
                kill(&mut process);
                return Err(error.into());
            }
            Ok(_) => {
                kill(&mut process);
                return Err(crashed("no hello from the worker"));
            }
            Err(error) => {
                let status = kill(&mut process);
                return Err(crashed(format!("{error} ({status})")));
            }
        };
        let (requests, written) = match spawn_writer(input) {
            Ok(writer) => writer,
            Err(error) => {
                kill(&mut process);
                return Err(crashed(error));
            }
        };
        let worker = Self {
            process,
            requests: Some(requests),
            written,
            replies,
            link,
            broken: false,
            counts: counts.clone(),
        };
        Ok((worker, name, caps))
    }

    /// Sends `request` and waits for its answer. Every event it produced
    /// has been forwarded by then. A worker that does not take the request
    /// and answer within `timeout` is killed, so a hang fails its session
    /// like a crash.
    fn call(&mut self, request: &Request, timeout: Duration) -> Result<(), SpeechError> {
        if self.broken {
            return Err(crashed("it stopped earlier"));
        }
        let started = Instant::now();
        let reply = match self.send(request, timeout) {
            Ok(()) => self.reply(timeout.saturating_sub(started.elapsed())),
            Err(gone) => gone,
        };
        match reply {
            Reply::Done => Ok(()),
            Reply::Error(error) => Err(error.into()),
            Reply::Gone(detail) => {
                bump(&self.counts.crashed);
                self.broken = true;
                Err(crashed(format!("{detail} ({})", kill(&mut self.process))))
            }
            Reply::Hung(detail) => {
                bump(&self.counts.killed);
                self.broken = true;
                Err(crashed(format!("{detail} ({})", kill(&mut self.process))))
            }
        }
    }

    /// Hands `request` to the writer thread and waits up to `timeout` for
    /// the worker to take it.
    fn send(&self, request: &Request, timeout: Duration) -> Result<(), Reply> {
        let mut frame = Vec::new();
        write_frame(&mut frame, request).map_err(|error| Reply::Gone(error.to_string()))?;
        let sent = self
            .requests
            .as_ref()
            .is_some_and(|requests| requests.send(frame).is_ok());
        if !sent {
            return Err(Reply::Gone("it stopped taking requests".into()));
        }
        match self.written.recv_timeout(timeout) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => Err(Reply::Gone(error.to_string())),
            Err(RecvTimeoutError::Timeout) => Err(Reply::Hung(format!(
                "it did not take the request within {timeout:?}"
            ))),
            Err(RecvTimeoutError::Disconnected) => {
                Err(Reply::Gone("it stopped taking requests".into()))
            }
        }
    }

    /// Waits for the answer to the request just sent: at most `timeout`,
    /// and at most [`STOPPED_GRACE`] once the attached session has stopped,
    /// since nobody waits for its work then. A worker past either limit
    /// is `Gone`.
    fn reply(&self, timeout: Duration) -> Reply {
        let started = Instant::now();
        let mut stopped: Option<Instant> = None;
        loop {
            match self.replies.recv_timeout(POLL) {
                Ok(reply) => return reply,
                Err(RecvTimeoutError::Disconnected) => {
                    return Reply::Gone("it stopped answering".into());
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            let now = Instant::now();
            if now.duration_since(started) >= timeout {
                return Reply::Hung(format!("it did not answer within {timeout:?}"));
            }
            let stopping = lock(&self.link)
                .as_ref()
                .is_some_and(AsrEvents::is_stopping);
            if stopping && now.duration_since(*stopped.get_or_insert(now)) >= STOPPED_GRACE {
                return Reply::Hung("it did not answer after its session stopped".into());
            }
        }
    }

    /// Whether the process is still running.
    fn is_alive(&mut self) -> bool {
        matches!(self.process.try_wait(), Ok(None))
    }

    /// Sends the stream's events to `events`, or nowhere.
    fn attach(&self, events: Option<AsrEvents>) {
        *lock(&self.link) = events;
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.attach(None);
        // Killing the worker ends a write it was not taking, and its output,
        // and so its pipe threads, unless a child of its own holds the
        // pipes; they are not waited for (see `launch`).
        kill(&mut self.process);
        drop(self.requests.take());
    }
}

/// The writer thread's ends: where encoded requests go, and how each
/// write went.
type WriterEnds = (mpsc::Sender<Vec<u8>>, mpsc::Receiver<std::io::Result<()>>);

/// The writer thread, which writes each frame it is sent to `input` and
/// reports how the write went, until a write fails or the sender is gone.
fn spawn_writer(input: ChildStdin) -> std::io::Result<WriterEnds> {
    let (requests, frames) = mpsc::channel::<Vec<u8>>();
    let (wrote, written) = mpsc::channel();
    std::thread::Builder::new()
        .name("speechkit-worker-writer".into())
        .spawn(move || {
            let mut input = BufWriter::new(input);
            for frame in frames {
                let result = input.write_all(&frame).and_then(|()| input.flush());
                let failed = result.is_err();
                if wrote.send(result).is_err() || failed {
                    return;
                }
            }
        })?;
    Ok((requests, written))
}

/// Stops `process` and describes how it ended.
fn kill(process: &mut Child) -> String {
    let _ = process.kill();
    match process.wait() {
        Ok(status) => status.to_string(),
        Err(error) => error.to_string(),
    }
}

/// The reader thread: forwards events and failures to the stream in
/// progress, and answers to the caller, until the worker's output ends.
fn read(mut output: BufReader<ChildStdout>, link: &Link, replies: &mpsc::Sender<Reply>) {
    let attached = || lock(link).clone();
    let gone = loop {
        let reply = match read_frame::<Response>(&mut output) {
            Ok(Some(Response::Event(event))) => {
                if let Some(events) = attached() {
                    events.send(event.into());
                }
                continue;
            }
            Ok(Some(Response::Failed(error))) => {
                if let Some(events) = attached() {
                    events.fail(error.into());
                }
                continue;
            }
            Ok(Some(Response::Done)) => Reply::Done,
            Ok(Some(Response::Error(error))) => Reply::Error(error),
            Ok(Some(Response::Hello { .. })) => break "it sent a second hello".to_owned(),
            Ok(None) => break "it exited".to_owned(),
            Err(error) => break error.to_string(),
        };
        if replies.send(reply).is_err() {
            return;
        }
    };
    // A crash fails the stream in progress at once, even between calls.
    if let Some(events) = attached() {
        events.fail(crashed(&gone));
    }
    let _ = replies.send(Reply::Gone(gone));
}

/// The workers not serving a stream, and how to start more.
struct Pool {
    command: WorkerCommand,
    idle: Mutex<Vec<Worker>>,
    counts: Arc<Counts>,
}

impl Pool {
    /// An idle worker that is still running, or a new one. Workers that
    /// exited while idle are dropped, so they fail no session.
    fn take(&self, name: &str) -> Result<Worker, SpeechError> {
        loop {
            // Popped in its own statement, so the lock is not held while a
            // dead worker is dropped.
            let idle = lock(&self.idle).pop();
            let Some(mut worker) = idle else { break };
            if worker.is_alive() {
                return Ok(worker);
            }
            bump(&self.counts.crashed);
            tracing::debug!(backend = %name, "dropped an idle worker process that had exited");
        }
        let (worker, _, _) = Worker::start(&self.command, &self.counts)?;
        tracing::debug!(backend = %name, "started a worker process");
        Ok(worker)
    }

    /// Takes `worker` back, unless it broke.
    fn give_back(&self, worker: Worker) {
        if !worker.broken {
            worker.attach(None);
            lock(&self.idle).push(worker);
        }
    }
}

/// A recognizer in worker processes, so a native crash fails one session
/// rather than the application.
///
/// Each session takes an idle worker, or starts one, which must load the
/// model within its command's
/// [start timeout](WorkerCommand::with_start_timeout) or is killed, failing
/// the session. A worker serves one session at a time, and
/// goes back to the pool when the session's stream is dropped, so
/// sessions run in parallel, at the cost of one model load per concurrent
/// session. If a worker dies, or hangs past its
/// [call timeout](Self::with_call_timeout) and is killed, only its session
/// fails, with a retryable [`SpeechError::Backend`].
pub struct IsolatedAsr {
    pool: Arc<Pool>,
    name: String,
    caps: AsrCapabilities,
    call_timeout: Duration,
}

impl IsolatedAsr {
    /// Starts `speechkit-worker` at `worker`, which loads the recognizer
    /// `config` describes in its own process.
    ///
    /// # Errors
    ///
    /// The errors of
    /// [`AsrConfig::validate`](crate::sherpa::AsrConfig::validate),
    /// checked here before the worker starts, or those of
    /// [`spawn_command`](Self::spawn_command). A worker that cannot load
    /// `config` reports the error
    /// [`AsrConfig::load`](crate::sherpa::AsrConfig::load) gives.
    pub fn spawn(
        worker: impl Into<PathBuf>,
        config: &crate::sherpa::AsrConfig,
    ) -> Result<Self, SpeechError> {
        config.validate()?;
        let json =
            serde_json::to_string(&super::config::WireConfig::from(config)).map_err(|e| {
                SpeechError::InvalidInput(format!("cannot encode the worker config: {e}"))
            })?;
        Self::spawn_command(WorkerCommand::new(worker).arg("--config").arg(json))
    }

    /// Starts the worker `command`, such as one of your own that calls
    /// [`serve`](super::serve), and waits for its handshake. That first
    /// worker waits in the pool for the first session.
    ///
    /// # Errors
    ///
    /// A retryable backend error if the process cannot start, dies before
    /// its handshake, or does not send it within the command's
    /// [start timeout](WorkerCommand::with_start_timeout); or the error the
    /// worker reports.
    pub fn spawn_command(command: WorkerCommand) -> Result<Self, SpeechError> {
        let counts = Arc::default();
        let (worker, name, caps) = Worker::start(&command, &counts)?;
        Ok(Self {
            pool: Arc::new(Pool {
                command,
                idle: Mutex::new(vec![worker]),
                counts,
            }),
            name: format!("{name}@worker"),
            caps,
            call_timeout: CALL_TIMEOUT,
        })
    }

    /// Sets how long the worker may take to take and answer one request:
    /// to open a stream, take a chunk of audio, or finish. A worker that takes
    /// longer is taken as hung and killed, and its session fails with a
    /// retryable [`SpeechError::Backend`]. A worker whose session was
    /// cancelled gets 2 s. Default: 60 s.
    #[must_use]
    pub fn with_call_timeout(mut self, timeout: Duration) -> Self {
        self.call_timeout = timeout;
        self
    }
}

impl AsrBackend for IsolatedAsr {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    /// `workers_started`, `worker_start_failures`, `workers_crashed`
    /// (exited on their own, in use or idle), and `workers_killed` (did not
    /// start or answer in time), counting the first worker too.
    fn counters(&self) -> Vec<(&'static str, u64)> {
        let counts = &self.pool.counts;
        let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        vec![
            ("workers_started", read(&counts.started)),
            ("worker_start_failures", read(&counts.start_failures)),
            ("workers_crashed", read(&counts.crashed)),
            ("workers_killed", read(&counts.killed)),
        ]
    }

    fn open(
        &self,
        opts: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        let mut worker = self.pool.take(&self.name)?;
        worker.attach(Some(events));
        let opened = worker.call(
            &Request::Open {
                language: opts.language.clone(),
                hints: opts.hints.clone(),
            },
            self.call_timeout,
        );
        match opened {
            Ok(()) => Ok(Box::new(ProcessStream {
                worker: Some(worker),
                pool: self.pool.clone(),
                call_timeout: self.call_timeout,
            })),
            Err(error) => {
                self.pool.give_back(worker);
                Err(error)
            }
        }
    }
}

impl std::fmt::Debug for IsolatedAsr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IsolatedAsr")
            .field("name", &self.name)
            .field("command", &self.pool.command)
            .field("idle", &lock(&self.pool.idle).len())
            .field("call_timeout", &self.call_timeout)
            .finish_non_exhaustive()
    }
}

/// A session's stream on one worker.
struct ProcessStream {
    worker: Option<Worker>,
    pool: Arc<Pool>,
    call_timeout: Duration,
}

impl ProcessStream {
    fn call(&mut self, request: &Request) -> Result<(), SpeechError> {
        match &mut self.worker {
            Some(worker) => worker.call(request, self.call_timeout),
            None => Err(crashed("it is gone")),
        }
    }
}

impl AsrStream for ProcessStream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.call(&Request::Accept {
            samples: samples.to_vec(),
        })
    }

    fn finish(&mut self) -> Result<(), SpeechError> {
        self.call(&Request::Finish)
    }

    fn cancel(&mut self) {
        let _ = self.call(&Request::Cancel);
    }
}

impl Drop for ProcessStream {
    /// Closes the stream, which stops its work in the worker, and returns
    /// the worker to the pool. The session's slot is held until then.
    fn drop(&mut self) {
        if let Some(mut worker) = self.worker.take() {
            let _ = worker.call(&Request::Close, self.call_timeout);
            self.pool.give_back(worker);
        }
    }
}
