//! `speechkit stream`: push a file in small chunks with `try_push`,
//! retrying the same chunk while the queue is full.

use std::{
    io::Write,
    time::{Duration, Instant},
};

use speechkit::asr::PushErrorKind;

use crate::{BackendFactory, CliError, StreamArgs, args::deadline_after, output, transcribe};

/// How long to wait before retrying a chunk the queue refused.
const RETRY: Duration = Duration::from_millis(5);

pub(crate) fn stream(
    args: &StreamArgs,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    interactive: bool,
) -> Result<(), CliError> {
    if args.chunk_ms == 0 {
        return Err(CliError::usage("--chunk-ms must be positive"));
    }
    let audio = transcribe::load(&args.file)?;
    let engine = backends.asr(&args.backend)?;
    let deadline = deadline_after(args.timeout);
    let chunk = usize::try_from(
        audio
            .sample_rate
            .frames_in(Duration::from_millis(args.chunk_ms)),
    )
    .unwrap_or(1_600)
    .max(1);
    let mut retries = 0_u64;
    let transcript = transcribe::with_progress(
        &engine,
        audio.sample_rate,
        transcribe::options(&args.backend),
        stderr,
        interactive,
        deadline,
        |session| {
            for piece in audio.samples.chunks(chunk) {
                let mut pending = piece.to_vec();
                loop {
                    let refused = match session.try_push(pending) {
                        Ok(()) => break,
                        Err(refused) => refused,
                    };
                    match refused.kind {
                        // The queue is full: back off and push the same audio again.
                        PushErrorKind::Full if Instant::now() < deadline => {
                            retries += 1;
                            pending = refused.into_chunk();
                            std::thread::sleep(RETRY);
                        }
                        PushErrorKind::Full => {
                            session.cancel();
                            return Err(CliError::input("timed out waiting for queue space"));
                        }
                        PushErrorKind::Closed => return Ok(()),
                        _ => {
                            session.cancel();
                            return Err(CliError::input(refused.to_string()));
                        }
                    }
                }
            }
            session.close_input();
            Ok(())
        },
    )?;
    tracing::info!(retries, "stream finished");
    if retries > 0 {
        let _ = writeln!(
            stderr,
            "the queue was full {retries} times; chunks were retried"
        );
    }
    output::print(&transcript, args.format, stdout).map_err(|e| CliError::input(e.to_string()))
}
