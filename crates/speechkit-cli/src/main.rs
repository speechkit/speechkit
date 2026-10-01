//! The `speechkit` command. All logic lives in the library, so tests run
//! it in-process.

use std::{io::IsTerminal, process::ExitCode};

fn main() -> ExitCode {
    let interactive = std::io::stderr().is_terminal();
    speechkit_cli::run_with(
        std::env::args_os(),
        &speechkit_cli::Backends::default(),
        &mut std::io::stdout(),
        &mut std::io::stderr(),
        interactive,
    )
}
