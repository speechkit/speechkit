//! The worker side of the protocol.

use std::{
    io::{self, Read, Write},
    process::ExitCode,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use crate::{
    Flow, SpeechError,
    asr::{AsrBackend, AsrEvents, AsrOptions, AsrStream},
    sherpa::AsrConfig,
};

use super::{
    config::WireConfig,
    protocol::{
        HANDSHAKE, Request, Response, WireCaps, WireError, options, read_frame, write_frame,
    },
};

/// Runs a worker: loads the recognizer that `--config JSON` on the command
/// line describes, and serves it on stdin and stdout until stdin closes.
/// `speechkit-worker` is this function, and
/// [`IsolatedAsr::spawn`](super::IsolatedAsr::spawn) writes the
/// JSON. A configuration the worker cannot load is reported to the parent
/// through the protocol.
pub fn worker_main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let config = match (args.next().as_deref(), args.next()) {
        (Some("--config"), Some(json)) => serde_json::from_str::<WireConfig>(&json)
            .map_err(|e| SpeechError::InvalidInput(format!("bad --config: {e}")))
            .and_then(AsrConfig::try_from),
        _ => Err(SpeechError::InvalidInput(
            "usage: speechkit-worker --config JSON".into(),
        )),
    };
    match config.and_then(|config| config.load()) {
        Ok(backend) => match serve(&backend, io::stdin().lock(), io::stdout()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        },
        Err(error) => {
            let mut output = io::stdout().lock();
            let _ = output.write_all(HANDSHAKE);
            let _ = write_frame(&mut output, &Response::Error(WireError::from(&error)));
            ExitCode::FAILURE
        }
    }
}

/// Serves `backend` over `input` and `output` until the input ends, one
/// stream at a time.
///
/// This is the protocol side of a worker: [`worker_main`] loads a
/// recognizer and calls it with stdin and stdout. A worker of your own,
/// started with [`IsolatedAsr::spawn_command`](super::IsolatedAsr::spawn_command),
/// can call it with any backend. Events go to `output` as the stream sends
/// them, from any thread, so `output` is shared with the stream.
///
/// # Errors
///
/// Any I/O error on the pipes.
pub fn serve(
    backend: &dyn AsrBackend,
    mut input: impl Read,
    output: impl Write + Send + 'static,
) -> io::Result<()> {
    let output: Output = Arc::new(Mutex::new(Box::new(output)));
    {
        let mut out = lock(&output);
        out.write_all(HANDSHAKE)?;
        write_frame(
            &mut *out,
            &Response::Hello {
                name: backend.name().to_owned(),
                caps: WireCaps::from(backend.capabilities()),
            },
        )?;
    }
    let mut served: Option<Served> = None;
    while let Some(request) = read_frame::<Request>(&mut input)? {
        let reply = match request {
            Request::Open { language, hints } => {
                if let Some(old) = served.take() {
                    old.close();
                }
                match Served::open(backend, &options(language, hints), &output) {
                    Ok(stream) => {
                        served = Some(stream);
                        Ok(())
                    }
                    Err(error) => Err(error),
                }
            }
            Request::Accept { samples } => match &mut served {
                Some(stream) => stream.stream.accept(&samples),
                None => Err(not_open()),
            },
            Request::Finish => match &mut served {
                Some(stream) => stream.stream.finish(),
                None => Err(not_open()),
            },
            Request::Cancel => {
                if let Some(stream) = &mut served {
                    stream.stream.cancel();
                }
                Ok(())
            }
            Request::Close => {
                if let Some(stream) = served.take() {
                    stream.close();
                }
                Ok(())
            }
        };
        let reply = match reply {
            Ok(()) => Response::Done,
            Err(error) => Response::Error(WireError::from(&error)),
        };
        write_frame(&mut *lock(&output), &reply)?;
    }
    Ok(())
}

/// The output, shared with the stream's events.
type Output = Arc<Mutex<Box<dyn Write + Send>>>;

/// The open stream.
struct Served {
    stream: Box<dyn AsrStream>,
    /// Set once the stream is dropped, so an events handle it leaked
    /// sends nothing more.
    closed: Arc<AtomicBool>,
}

impl Served {
    fn open(
        backend: &dyn AsrBackend,
        options: &AsrOptions,
        output: &Output,
    ) -> Result<Self, SpeechError> {
        let closed = Arc::new(AtomicBool::new(false));
        let (out, gone) = (output.clone(), closed.clone());
        let (fail_out, fail_gone) = (output.clone(), closed.clone());
        let forward = AsrEvents::forward(
            move |event| {
                if gone.load(Ordering::SeqCst) {
                    return Flow::Stop;
                }
                match write_frame(&mut *lock(&out), &Response::Event(event.into())) {
                    Ok(()) => Flow::Continue,
                    Err(_) => Flow::Stop,
                }
            },
            move |error| {
                if !fail_gone.load(Ordering::SeqCst) {
                    let failed = Response::Failed(WireError::from(&error));
                    let _ = write_frame(&mut *lock(&fail_out), &failed);
                }
            },
        );
        Ok(Self {
            stream: backend.open(options, forward)?,
            closed,
        })
    }

    /// Drops the stream, which stops its threads, then silences it.
    fn close(self) {
        let Self { stream, closed } = self;
        drop(stream);
        closed.store(true, Ordering::SeqCst);
    }
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn not_open() -> SpeechError {
    SpeechError::InvalidInput("no stream is open".into())
}
