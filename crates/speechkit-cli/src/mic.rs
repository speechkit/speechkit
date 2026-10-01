//! `speechkit mic`: transcribe a microphone until Enter.

use std::{io::Write, time::Duration};

use speechkit::asr::AsrOptions;
use speechkit::io::Microphone;

use crate::{BackendFactory, CliError, MicArgs, output, progress};

/// How long finishing the session after Enter may take.
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
    let mut options = AsrOptions::default();
    options.language.clone_from(&args.backend.language);
    let listening = microphone.listen(&engine, options)?;
    let updates = listening.updates();
    let _ = writeln!(stderr, "listening; press Enter to stop");
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            listening.stop();
        });
        progress::show_all(updates, stderr, interactive);
    });
    let transcript = listening
        .finish(WAIT)
        .map_err(speechkit::SpeechError::from)?;
    output::print(&transcript, args.format, stdout).map_err(|e| CliError::input(e.to_string()))
}
