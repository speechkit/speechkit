//! Development tasks for the speechkit workspace.
//!
//! Run with `cargo xtask <command>`.

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::{Parser, Subcommand};

mod audio_fixtures;
mod models;
mod release;

type Result<T = ()> = std::result::Result<T, String>;

/// The workspace root.
fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

#[derive(Parser)]
#[command(
    name = "cargo xtask",
    about = "Development tasks for the speechkit workspace"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Download and verify the model archives listed in fixtures/manifest.json.
    FetchFixtures {
        /// Only these model IDs (comma-separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
    },
    /// Generate audio fixtures in many formats with ffmpeg.
    GenAudioFixtures {
        /// Output directory. Default: fixtures/audio.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Check that the workspace is ready to release.
    ReleaseCheck,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::FetchFixtures { only } => models::fetch_fixtures(&only),
        Command::GenAudioFixtures { out } => {
            audio_fixtures::generate(&out.unwrap_or_else(audio_fixtures::default_dir))
        }
        Command::ReleaseCheck => release::release_check(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}
