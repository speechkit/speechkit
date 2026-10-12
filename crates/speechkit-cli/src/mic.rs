//! `speechkit mic`: transcribe a microphone until Enter or Ctrl+C.

use std::{
    io::Write,
    sync::{Arc, Weak},
    time::Duration,
};

use speechkit::io::{Listening, Microphone};

use crate::{BackendFactory, CliError, MicArgs, output, progress, transcribe};

/// How long finishing the session after the stop may take.
const WAIT: Duration = Duration::from_secs(60);

pub(crate) fn mic(
    args: &MicArgs,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    interactive: bool,
) -> Result<(), CliError> {
    let microphone = match &args.device {
        Some(name) => Microphone::open(name)?,
        None => Microphone::open_default()?,
    };
    let engine = backends.asr(&args.backend)?;
    let listening = Arc::new(microphone.listen(&engine, transcribe::options(&args.backend))?);
    let updates = listening.updates();
    let _ = writeln!(stderr, "listening; press Enter or Ctrl+C to stop");
    // Detached, so a listening that ends on its own does not wait for
    // Enter, and holding only a weak handle, so it is still dropped here.
    stop_on_enter(Arc::downgrade(&listening));
    stop_on_signal(Arc::downgrade(&listening));
    progress::show_all(updates, stderr, interactive);
    let transcript = listening
        .finish(WAIT)
        .map_err(speechkit::SpeechError::from)?;
    Ok(output::print(&transcript, args.format, stdout)?)
}

/// Stops the listening when a line is read. At the end of the input, as
/// with `< /dev/null`, only a signal stops it.
fn stop_on_enter(listening: Weak<Listening>) {
    std::thread::spawn(move || {
        let mut line = String::new();
        if matches!(std::io::stdin().read_line(&mut line), Ok(read) if read > 0)
            && let Some(listening) = listening.upgrade()
        {
            listening.stop();
        }
    });
}

/// Stops the listening on Ctrl+C or SIGTERM, so the transcript is still
/// printed. A second signal exits at once.
fn stop_on_signal(listening: Weak<Listening>) {
    std::thread::spawn(move || {
        let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        runtime.block_on(speechkit::server::shutdown_signal());
        if let Some(listening) = listening.upgrade() {
            listening.stop();
        }
        runtime.block_on(speechkit::server::shutdown_signal());
        std::process::exit(130);
    });
}
