//! Command-line arguments.

use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

use clap::{Args, Parser, Subcommand, ValueEnum};

/// The deadline of a `--timeout`, `seconds` from now. The largest values
/// mean "no limit": where `Instant + Duration` would panic, it gives the
/// latest instant the platform can represent near the sum.
pub(crate) fn deadline_after(seconds: u64) -> Instant {
    let now = Instant::now();
    let mut timeout = Duration::from_secs(seconds);
    loop {
        if let Some(deadline) = now.checked_add(timeout) {
            return deadline;
        }
        timeout /= 2;
    }
}

/// Speech recognition and synthesis from the command line.
#[derive(Debug, Parser)]
#[command(name = "speechkit", version, about, propagate_version = true)]
pub struct Cli {
    /// More log output (repeat for more).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
    /// Only errors on stderr.
    #[arg(short, long, global = true)]
    pub quiet: bool,
    /// What to do.
    #[command(subcommand)]
    pub command: Command,
}

/// A subcommand.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Transcribe an audio file.
    Transcribe(TranscribeArgs),
    /// Stream an audio file in 100 ms chunks, showing how backpressure works.
    Stream(StreamArgs),
    /// Serve an OpenAI-compatible transcription (and speech) API.
    Serve(ServeArgs),
    /// Transcribe a microphone until Enter is pressed.
    Mic(MicArgs),
    /// Synthesize speech to a WAV file or the speakers.
    Speak(SpeakArgs),
    /// List the voices of a synthesis backend.
    Voices(VoicesArgs),
    /// List the audio input and output devices.
    Devices,
}

/// Parses `--device`: any name but an empty one, without spaces around it.
fn device_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        Err("the device name is empty".to_owned())
    } else {
        Ok(name.to_owned())
    }
}

/// A recognition backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendKind {
    /// A local streaming transducer (sherpa-onnx), such as streaming Zipformer.
    SherpaStreaming,
    /// A local offline model behind silero VAD (sherpa-onnx), such as SenseVoice.
    SherpaOffline,
    /// An OpenAI-compatible transcription service.
    OpenaiHttp,
    /// OpenAI Realtime transcription over WebSocket.
    OpenaiRealtime,
    /// DashScope real-time recognition over WebSocket.
    Dashscope,
}

/// Which backend to run, and how.
#[derive(Debug, Clone, Args)]
#[expect(
    clippy::doc_markdown,
    reason = "clap prints these comments as help text, where backticks would show"
)]
pub struct BackendArgs {
    /// The backend.
    #[arg(long, value_enum, default_value = "sherpa-streaming")]
    pub backend: BackendKind,
    /// The model directory (sherpa backends).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The model family, for flat layouts: sense-voice, paraformer, firered-ctc, ...
    #[arg(long)]
    pub family: Option<String>,
    /// The silero VAD model file (sherpa-offline; optional for openai-http).
    #[arg(long)]
    pub vad: Option<PathBuf>,
    /// A punctuation model directory.
    #[arg(long)]
    pub punct: Option<PathBuf>,
    /// Phrases to favor, comma-separated: decoding bias for transducers,
    /// prompt hints for qwen3-asr and funasr-nano.
    #[arg(long, value_delimiter = ',')]
    pub hotwords: Vec<String>,
    /// Inference threads (sherpa backends).
    #[arg(long, default_value_t = 2)]
    pub threads: usize,
    /// Where inference runs: cpu, cuda, or coreml.
    #[arg(long, default_value = "cpu")]
    pub provider: String,
    /// The spoken language, for backends that accept it.
    #[arg(long)]
    pub language: Option<String>,
    /// A JSON settings file (cloud backends). The key comes from SPEECHKIT_API_KEY.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

/// A synthesis backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TtsBackendKind {
    /// A local model (sherpa-onnx): VITS/Piper, Matcha, or Kokoro.
    Sherpa,
    /// An OpenAI-compatible speech service.
    Openai,
    /// DashScope speech synthesis over WebSocket.
    Dashscope,
}

/// Which synthesis backend to run, and how.
#[derive(Debug, Clone, Args)]
#[expect(
    clippy::doc_markdown,
    reason = "clap prints these comments as help text, where backticks would show"
)]
pub struct TtsBackendArgs {
    /// The backend.
    #[arg(long, value_enum, default_value = "sherpa")]
    pub backend: TtsBackendKind,
    /// The model directory (sherpa).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// A JSON settings file (cloud backends). The key comes from SPEECHKIT_API_KEY.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Inference threads (sherpa).
    #[arg(long, default_value_t = 2)]
    pub threads: usize,
    /// Where inference runs: cpu, cuda, or coreml.
    #[arg(long, default_value = "cpu")]
    pub provider: String,
}

/// `speechkit speak`.
#[derive(Debug, Clone, Args)]
#[command(group(clap::ArgGroup::new("input").required(true).args(["text", "file"])))]
#[command(group(clap::ArgGroup::new("output").required(true).args(["out", "play"])))]
pub struct SpeakArgs {
    /// The text to speak.
    pub text: Option<String>,
    /// Read the text from this file instead.
    #[arg(long)]
    pub file: Option<PathBuf>,
    /// The backend.
    #[command(flatten)]
    pub backend: TtsBackendArgs,
    /// The voice ID; see `speechkit voices`.
    #[arg(long)]
    pub voice: Option<String>,
    /// The speaking speed.
    #[arg(long, default_value_t = 1.0)]
    pub speed: f32,
    /// Write a WAV file.
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Play through an output device: the default, or the one --device names.
    #[arg(long)]
    pub play: bool,
    /// With --play, the output device: its name, or part of one; see
    /// `speechkit devices`.
    #[arg(long, value_parser = device_name)]
    pub device: Option<String>,
    /// Give up after this many seconds.
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,
}

/// `speechkit voices`.
#[derive(Debug, Clone, Args)]
pub struct VoicesArgs {
    /// The backend.
    #[command(flatten)]
    pub backend: TtsBackendArgs,
}

/// How to print the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Plain text.
    Text,
    /// JSON with the text and timed segments.
    Json,
    /// SubRip subtitles.
    Srt,
}

/// `speechkit transcribe`.
#[derive(Debug, Clone, Args)]
pub struct TranscribeArgs {
    /// The audio file. Any supported format; WAV, FLAC, MP3, M4A, Ogg Vorbis, MKV.
    pub file: PathBuf,
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,
    /// The output format.
    #[arg(long, value_enum, default_value = "text")]
    pub format: OutputFormat,
    /// Give up after this many seconds.
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,
}

/// `speechkit stream`.
#[derive(Debug, Clone, Args)]
pub struct StreamArgs {
    /// The audio file.
    pub file: PathBuf,
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,
    /// Chunk length in milliseconds.
    #[arg(long, default_value_t = 100)]
    pub chunk_ms: u64,
    /// The output format.
    #[arg(long, value_enum, default_value = "text")]
    pub format: OutputFormat,
    /// Give up after this many seconds.
    #[arg(long, default_value_t = 600)]
    pub timeout: u64,
}

/// `speechkit serve`.
#[derive(Debug, Clone, Args)]
pub struct ServeArgs {
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,
    /// Where to listen.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub bind: String,
    /// An environment variable holding the bearer token clients must send.
    #[arg(long)]
    pub auth_token_env: Option<String>,
    /// The largest upload, in MiB.
    #[arg(long, default_value_t = 25)]
    pub max_body_mib: usize,
    /// The limit for one transcription, in seconds.
    #[arg(long)]
    pub timeout: Option<u64>,
    /// The model name reported by /v1/models.
    #[arg(long)]
    pub model_id: Option<String>,
    /// Also serve /v1/audio/speech with this synthesis backend.
    #[arg(long, value_enum)]
    pub tts_backend: Option<TtsBackendKind>,
    /// The synthesis model directory (sherpa); implies --tts-backend sherpa.
    #[arg(long)]
    pub tts_model: Option<PathBuf>,
    /// A JSON settings file for a cloud synthesis backend.
    #[arg(long)]
    pub tts_config: Option<PathBuf>,
    /// The synthesis model name reported by /v1/models.
    #[arg(long)]
    pub tts_model_id: Option<String>,
}

impl ServeArgs {
    /// The synthesis backend to serve, if any. Threads and provider are
    /// shared with the recognition backend.
    pub fn tts(&self) -> Option<TtsBackendArgs> {
        let backend = self
            .tts_backend
            .or_else(|| self.tts_model.as_ref().map(|_| TtsBackendKind::Sherpa))?;
        Some(TtsBackendArgs {
            backend,
            model: self.tts_model.clone(),
            config: self.tts_config.clone(),
            threads: self.backend.threads,
            provider: self.backend.provider.clone(),
        })
    }
}

/// `speechkit mic`.
#[derive(Debug, Clone, Args)]
pub struct MicArgs {
    /// The backend.
    #[command(flatten)]
    pub backend: BackendArgs,
    /// The input device: its name, or part of one; see `speechkit devices`.
    /// The default input device if not given.
    #[arg(long, value_parser = device_name)]
    pub device: Option<String>,
    /// The output format.
    #[arg(long, value_enum, default_value = "text")]
    pub format: OutputFormat,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timeout_is_that_many_seconds_from_now() {
        let before = Instant::now();
        let deadline = deadline_after(2);
        assert!(deadline >= before + Duration::from_secs(2));
        assert!(deadline <= Instant::now() + Duration::from_secs(2));
    }

    #[test]
    fn the_largest_timeout_reaches_far_without_overflowing() {
        let far = deadline_after(u64::MAX);
        assert!(far > Instant::now() + Duration::from_secs(1_000_000));
        assert_eq!(
            deadline_after(0).saturating_duration_since(Instant::now()),
            Duration::ZERO
        );
    }
}
