//! The parent side: an `AsrBackend` backed by a pool of worker processes.

use std::{
    ffi::OsString,
    io::{BufReader, BufWriter},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc},
    thread::JoinHandle,
};

use crate::{
    SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvents, AsrOptions, AsrStream},
};

use super::protocol::{Request, Response, WireError, read_frame, skip_to_handshake, write_frame};

const NAME: &str = "speechkit-worker";

fn crashed(detail: impl std::fmt::Display) -> SpeechError {
    SpeechError::backend(NAME, true, format!("the worker process failed: {detail}"))
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// How to start a worker process.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Vec<(OsString, OsString)>,
}

impl WorkerCommand {
    /// Runs `program`.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
        }
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
    Gone(String),
}

/// Where the reader thread sends the events of the stream in progress.
type Link = Arc<Mutex<Option<AsrEvents>>>;

/// A running worker, serving one stream at a time.
struct Worker {
    process: Child,
    input: BufWriter<ChildStdin>,
    replies: mpsc::Receiver<Reply>,
    link: Link,
    reader: Option<JoinHandle<()>>,
    /// The worker is gone, or a call to it failed.
    broken: bool,
}

impl Worker {
    fn start(command: &WorkerCommand) -> Result<(Self, String, AsrCapabilities), SpeechError> {
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
        let mut output = BufReader::new(output);
        let hello =
            skip_to_handshake(&mut output).and_then(|()| read_frame::<Response>(&mut output));
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
        let link: Link = Arc::default();
        let (send, replies) = mpsc::channel();
        let reader = std::thread::Builder::new()
            .name("speechkit-worker-reader".into())
            .spawn({
                let link = link.clone();
                move || read(output, &link, &send)
            })
            .map_err(|error| {
                kill(&mut process);
                crashed(error)
            })?;
        let worker = Self {
            process,
            input: BufWriter::new(input),
            replies,
            link,
            reader: Some(reader),
            broken: false,
        };
        Ok((worker, name, caps))
    }

    /// Sends `request` and waits for its answer. Every event it produced
    /// has been forwarded by then.
    fn call(&mut self, request: &Request) -> Result<(), SpeechError> {
        if self.broken {
            return Err(crashed("it stopped earlier"));
        }
        let reply = match write_frame(&mut self.input, request) {
            Ok(()) => self
                .replies
                .recv()
                .unwrap_or_else(|_| Reply::Gone("it stopped answering".into())),
            Err(error) => Reply::Gone(error.to_string()),
        };
        match reply {
            Reply::Done => Ok(()),
            Reply::Error(error) => Err(error.into()),
            Reply::Gone(detail) => {
                self.broken = true;
                Err(crashed(format!("{detail} ({})", kill(&mut self.process))))
            }
        }
    }

    /// Sends the stream's events to `events`, or nowhere.
    fn attach(&self, events: Option<AsrEvents>) {
        *lock(&self.link) = events;
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.attach(None);
        kill(&mut self.process);
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
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
}

impl Pool {
    /// An idle worker, or a new one.
    fn take(&self, name: &str) -> Result<Worker, SpeechError> {
        if let Some(worker) = lock(&self.idle).pop() {
            return Ok(worker);
        }
        let (worker, _, _) = Worker::start(&self.command)?;
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
/// Each session takes an idle worker, or starts one, which loads the model
/// within `start`'s deadline. A worker serves one session at a time, and
/// goes back to the pool when the session's stream is dropped, so
/// sessions run in parallel, at the cost of one model load per concurrent
/// session. If a worker dies, only its session fails, with a retryable
/// [`SpeechError::Backend`].
pub struct IsolatedAsr {
    pool: Arc<Pool>,
    name: String,
    caps: AsrCapabilities,
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
    /// A retryable backend error if the process cannot start or dies
    /// before its handshake, or the error the worker reports.
    pub fn spawn_command(command: WorkerCommand) -> Result<Self, SpeechError> {
        let (worker, name, caps) = Worker::start(&command)?;
        Ok(Self {
            pool: Arc::new(Pool {
                command,
                idle: Mutex::new(vec![worker]),
            }),
            name: format!("{name}@worker"),
            caps,
        })
    }
}

impl AsrBackend for IsolatedAsr {
    fn name(&self) -> &str {
        &self.name
    }

    fn capabilities(&self) -> &AsrCapabilities {
        &self.caps
    }

    fn open(
        &self,
        opts: &AsrOptions,
        events: AsrEvents,
    ) -> Result<Box<dyn AsrStream>, SpeechError> {
        let mut worker = self.pool.take(&self.name)?;
        worker.attach(Some(events));
        let opened = worker.call(&Request::Open {
            language: opts.language.clone(),
            hints: opts.hints.clone(),
        });
        match opened {
            Ok(()) => Ok(Box::new(ProcessStream {
                worker: Some(worker),
                pool: self.pool.clone(),
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
            .finish_non_exhaustive()
    }
}

/// A session's stream on one worker.
struct ProcessStream {
    worker: Option<Worker>,
    pool: Arc<Pool>,
}

impl ProcessStream {
    fn call(&mut self, request: &Request) -> Result<(), SpeechError> {
        match &mut self.worker {
            Some(worker) => worker.call(request),
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
            let _ = worker.call(&Request::Close);
            self.pool.give_back(worker);
        }
    }
}
