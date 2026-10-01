//! The `speechkit` command as a library.
//!
//! [`run`] parses arguments and runs a command, writing the transcript (or
//! the voice or device list) to `stdout` and everything else to `stderr`.
//! The binary only passes the real [`Backends`] and the process streams;
//! tests pass a factory built on speechkit-testkit instead, so no fake
//! backend ships in the binary.
//!
//! This Rust API is for those tests and is not covered by semver: new
//! flags add fields to the public args structs in patch releases. The
//! command line itself is.

mod args;
mod backends;
mod devices;
mod mic;
mod output;
mod progress;
mod serve;
mod speak;
mod stream;
mod transcribe;
mod tts_backends;

use std::{ffi::OsString, io::Write, process::ExitCode};

use clap::Parser;
use speechkit::SpeechError;

pub use args::{
    BackendArgs, BackendKind, Cli, Command, MicArgs, OutputFormat, ServeArgs, SpeakArgs,
    StreamArgs, TranscribeArgs, TtsBackendArgs, TtsBackendKind, VoicesArgs,
};
pub use backends::{BackendFactory, Backends};
pub use serve::{ServeHandle, serve};

/// Exit code for a usage error.
pub const EXIT_USAGE: u8 = 2;
/// Exit code for invalid input: a bad file, model, or option.
pub const EXIT_INPUT: u8 = 3;
/// Exit code for a backend failure.
pub const EXIT_BACKEND: u8 = 4;

/// A command failure, with the exit code it maps to.
#[derive(Debug)]
pub struct CliError {
    /// The process exit code.
    pub code: u8,
    /// What to print.
    pub message: String,
}

impl CliError {
    pub(crate) fn usage(message: impl Into<String>) -> Self {
        Self {
            code: EXIT_USAGE,
            message: message.into(),
        }
    }

    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self {
            code: EXIT_INPUT,
            message: message.into(),
        }
    }
}

impl From<SpeechError> for CliError {
    fn from(error: SpeechError) -> Self {
        let code = match error {
            SpeechError::InvalidInput(_)
            | SpeechError::InvalidModel(_)
            | SpeechError::Unsupported(_) => EXIT_INPUT,
            _ => EXIT_BACKEND,
        };
        let message = match std::error::Error::source(&error) {
            Some(source) => format!("{error}: {source}"),
            None => error.to_string(),
        };
        Self { code, message }
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Runs the command line `args` (including the program name), for a
/// non-interactive stderr.
pub fn run<I, T>(
    args: I,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    run_with(args, backends, stdout, stderr, false)
}

/// Like [`run`]. On an interactive stderr, partial results rewrite a
/// single line.
pub fn run_with<I, T>(
    args: I,
    backends: &dyn BackendFactory,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    interactive: bool,
) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(error) => {
            // Help and version are "errors" to clap that belong on stdout.
            let stream: &mut dyn Write = if error.use_stderr() { stderr } else { stdout };
            let _ = write!(stream, "{}", error.render());
            return ExitCode::from(u8::try_from(error.exit_code()).unwrap_or(EXIT_USAGE));
        }
    };
    init_logging(cli.verbose, cli.quiet);
    let result = match cli.command {
        Command::Transcribe(args) => {
            transcribe::transcribe(&args, backends, stdout, stderr, interactive && !cli.quiet)
        }
        Command::Stream(args) => {
            stream::stream(&args, backends, stdout, stderr, interactive && !cli.quiet)
        }
        Command::Serve(args) => serve::serve_until_signal(&args, backends, stderr),
        Command::Mic(args) => mic::mic(&args, backends, stdout, stderr, interactive && !cli.quiet),
        Command::Speak(args) => speak::speak(&args, backends, stderr),
        Command::Voices(args) => speak::voices(&args, backends, stdout),
        Command::Devices => devices::devices(stdout, stderr),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            let _ = writeln!(stderr, "error: {error}");
            ExitCode::from(error.code)
        }
    }
}

fn init_logging(verbose: u8, quiet: bool) {
    let level = match (quiet, verbose) {
        (true, _) => tracing::Level::ERROR,
        (false, 0) => tracing::Level::WARN,
        (false, 1) => tracing::Level::INFO,
        (false, 2) => tracing::Level::DEBUG,
        (false, _) => tracing::Level::TRACE,
    };
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(level)
        .try_init();
}
