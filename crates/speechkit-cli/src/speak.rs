//! `speechkit speak` and `speechkit voices`.

use std::io::Write;

use speechkit::io::Speaker;
use speechkit::{audio::encode_wav, tts::TtsOptions};

use crate::{BackendFactory, CliError, SpeakArgs, VoicesArgs, args::deadline_after};

fn text(args: &SpeakArgs) -> Result<String, CliError> {
    match (&args.text, &args.file) {
        (Some(text), None) => Ok(text.clone()),
        (None, Some(path)) => std::fs::read_to_string(path)
            .map_err(|e| CliError::input(format!("{}: {e}", path.display()))),
        _ => Err(CliError::usage("give the text or --file, not both")),
    }
}

pub(crate) fn speak(
    args: &SpeakArgs,
    backends: &dyn BackendFactory,
    stderr: &mut dyn Write,
) -> Result<(), CliError> {
    // clap's `requires` counts a flag's default as present, so check here.
    if args.device.is_some() && !args.play {
        return Err(CliError::usage("--device needs --play"));
    }
    let text = text(args)?;
    let engine = backends.tts(&args.backend)?;
    let mut options = TtsOptions::default().with_speed(args.speed);
    options.voice.clone_from(&args.voice);
    let deadline = deadline_after(args.timeout);
    if let Some(path) = &args.out {
        let audio = engine
            .synthesize(&text, options, deadline)
            .map_err(|failure| CliError::from(failure.error))?;
        let wav = encode_wav(&audio)?;
        std::fs::write(path, wav)
            .map_err(|e| CliError::input(format!("{}: {e}", path.display())))?;
        let _ = writeln!(
            stderr,
            "wrote {:.2} s at {} to {}",
            audio.duration().as_secs_f64(),
            audio.sample_rate,
            path.display()
        );
        return Ok(());
    }
    let speaker = match &args.device {
        Some(name) => Speaker::open(name)?,
        None => Speaker::open_default()?,
    };
    speaker.speak(&engine, &text, options)?.finish(deadline)?;
    Ok(())
}

pub(crate) fn voices(
    args: &VoicesArgs,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
) -> Result<(), CliError> {
    let engine = backends.tts(&args.backend)?;
    let write = |stdout: &mut dyn Write, line: String| {
        writeln!(stdout, "{line}").map_err(|e| CliError::input(e.to_string()))
    };
    for voice in engine.voices() {
        let languages = if voice.languages.is_empty() {
            "any".to_owned()
        } else {
            voice.languages.join(",")
        };
        if voice.name == voice.id {
            write(stdout, format!("{}\t{languages}", voice.id))?;
        } else {
            write(stdout, format!("{}\t{languages}\t{}", voice.id, voice.name))?;
        }
    }
    Ok(())
}
