//! `speechkit transcribe`.

use std::{
    io::Write,
    time::{Duration, Instant},
};

use speechkit::{
    AudioBuffer, SampleRate,
    asr::{AsrEngine, AsrOptions, PushErrorKind},
    audio::{self, DecodeLimits},
};

use crate::{
    BackendArgs, BackendFactory, CliError, TranscribeArgs, args::deadline_after, output, progress,
};

pub(crate) fn load(path: &std::path::Path) -> Result<AudioBuffer, CliError> {
    Ok(audio::read(path, DecodeLimits::default())?)
}

pub(crate) fn options(backend: &BackendArgs) -> AsrOptions {
    let options = AsrOptions::default();
    match &backend.language {
        Some(language) => options.with_language(language),
        None => options,
    }
}

/// Runs `feed` against a session for audio at `rate` while its progress is
/// shown on stderr, then returns the transcript. A `feed` that fails must cancel the
/// session, or the progress display waits for the deadline; its error is
/// returned in place of the session's.
pub(crate) fn with_progress(
    engine: &AsrEngine,
    rate: SampleRate,
    options: AsrOptions,
    stderr: &mut dyn Write,
    interactive: bool,
    deadline: Instant,
    feed: impl FnOnce(&speechkit::asr::AsrSession) -> Result<(), CliError> + Send,
) -> Result<speechkit::asr::Transcript, CliError> {
    let session = engine.start(rate, options, deadline)?;
    let updates = session.updates();
    let pushed = std::thread::scope(|scope| {
        let session = &session;
        let pusher = scope.spawn(move || feed(session));
        progress::show_until_done(updates, stderr, interactive, session, deadline);
        pusher
            .join()
            .unwrap_or_else(|_| Err(CliError::input("the feeding thread panicked")))
    });
    let finished = session.finish(deadline);
    pushed?;
    Ok(finished.map_err(speechkit::SpeechError::from)?)
}

pub(crate) fn transcribe(
    args: &TranscribeArgs,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    interactive: bool,
) -> Result<(), CliError> {
    let audio = load(&args.file)?;
    let engine = backends.asr(&args.backend)?;
    let deadline = deadline_after(args.timeout);
    let transcript = with_progress(
        &engine,
        audio.sample_rate,
        options(&args.backend),
        stderr,
        interactive,
        deadline,
        |session| {
            let rate = audio.sample_rate;
            let push = Duration::from_millis(100).min(engine.limits().input_queue);
            let chunk = usize::try_from(rate.frames_in(push))
                .unwrap_or(1_600)
                .max(1);
            for piece in audio.samples.chunks(chunk) {
                match session.push(piece, deadline) {
                    Ok(()) => {}
                    // The deadline passed or the session ended: `finish`
                    // reports why.
                    Err(refused)
                        if matches!(refused.kind, PushErrorKind::Full | PushErrorKind::Closed) =>
                    {
                        break;
                    }
                    Err(refused) => {
                        session.cancel();
                        return Err(CliError::input(format!("cannot transcribe: {refused}")));
                    }
                }
            }
            session.close_input();
            Ok(())
        },
    )?;
    output::print(&transcript, args.format, stdout).map_err(|e| CliError::input(e.to_string()))
}
