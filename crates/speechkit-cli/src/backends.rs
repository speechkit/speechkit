//! Turning `--backend` and its options into an engine.

use std::{
    path::Path,
    sync::{Arc, OnceLock},
};

use serde::Deserialize;
use speechkit::cloud::{
    CloudRuntime, DashScopeAsr, DashScopeAsrConfig, OpenAiRealtime, OpenAiRealtimeConfig,
    OpenAiTranscription, OpenAiTranscriptionConfig,
};
use speechkit::sherpa::{Asr, AsrConfig, Inference, PunctuationConfig, SileroVadConfig};
use speechkit::{Secret, SpeechError, asr::AsrEngine, tts::TtsEngine, vad::VadBackend};

use crate::{BackendArgs, BackendKind, TtsBackendArgs};

/// Builds engines from command-line options.
pub trait BackendFactory {
    /// The speech recognition engine `args` describe.
    ///
    /// # Errors
    ///
    /// Any error building the backend.
    fn asr(&self, args: &BackendArgs) -> Result<AsrEngine, SpeechError>;

    /// The speech synthesis engine `args` describe.
    ///
    /// # Errors
    ///
    /// Any error building the backend. By default, `Unsupported`.
    fn tts(&self, args: &TtsBackendArgs) -> Result<TtsEngine, SpeechError> {
        let _ = args;
        Err(SpeechError::Unsupported(
            "this build has no speech synthesis backends".into(),
        ))
    }
}

/// Workers of the runtime the cloud backends share.
const CLOUD_THREADS: usize = 2;

/// The real backends: sherpa-onnx, OpenAI, and DashScope.
///
/// The cloud backends one `Backends` builds share a runtime, started with
/// the first of them, so a command that never uses the cloud starts none.
#[derive(Debug, Clone, Default)]
pub struct Backends {
    cloud: OnceLock<CloudRuntime>,
}

impl Backends {
    /// The runtime the cloud backends share.
    pub(crate) fn cloud(&self) -> Result<CloudRuntime, SpeechError> {
        if let Some(runtime) = self.cloud.get() {
            return Ok(runtime.clone());
        }
        let started = CloudRuntime::owned(CLOUD_THREADS)?;
        Ok(self.cloud.get_or_init(|| started).clone())
    }
}

fn engine(backend: impl speechkit::asr::AsrBackend) -> AsrEngine {
    AsrEngine::new(backend)
}

pub(crate) fn required<'a>(
    value: Option<&'a Path>,
    flag: &str,
    backend: &str,
) -> Result<&'a Path, SpeechError> {
    value.ok_or_else(|| SpeechError::InvalidInput(format!("--backend {backend} needs {flag}")))
}

/// `--config` for openai-http.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiFile {
    base_url: String,
    model: String,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    /// Replaced by `stream` in 0.3; kept only to say so.
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

impl OpenAiFile {
    fn check(&self) -> Result<(), SpeechError> {
        if self.response_format.is_some() {
            return Err(SpeechError::InvalidInput(
                "response_format was replaced by \"stream\": true for server-sent events; \
                 remove it (JSON is always used otherwise)"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// `--config` for openai-realtime and dashscope.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SocketFile {
    model: String,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    server_vad: Option<bool>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

pub(crate) fn config_file<T: for<'de> Deserialize<'de>>(
    config: Option<&Path>,
    backend: &str,
) -> Result<T, SpeechError> {
    let path = required(config, "--config CONFIG.json", backend)?;
    let text = std::fs::read_to_string(path)
        .map_err(|e| SpeechError::InvalidInput(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| SpeechError::InvalidInput(format!("{}: {e}", path.display())))
}

pub(crate) fn api_key() -> Option<Arc<Secret>> {
    std::env::var("SPEECHKIT_API_KEY")
        .ok()
        .filter(|key| !key.trim().is_empty())
        .map(|key| Arc::new(Secret::new(key)))
}

pub(crate) fn required_key() -> Result<Arc<Secret>, SpeechError> {
    api_key().ok_or_else(|| SpeechError::InvalidInput("set SPEECHKIT_API_KEY".into()))
}

fn realtime(backends: &Backends, args: &BackendArgs) -> Result<OpenAiRealtime, SpeechError> {
    let file: SocketFile = config_file(args.config.as_deref(), "openai-realtime")?;
    let mut config = OpenAiRealtimeConfig::new(file.model, required_key()?);
    if let Some(endpoint) = file.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(server_vad) = file.server_vad {
        config = config.with_server_vad(server_vad);
    }
    if let Some(seconds) = file.timeout_secs {
        config = config.with_timeout(std::time::Duration::from_secs(seconds));
    }
    OpenAiRealtime::new(config, backends.cloud()?)
}

fn dashscope(backends: &Backends, args: &BackendArgs) -> Result<DashScopeAsr, SpeechError> {
    let file: SocketFile = config_file(args.config.as_deref(), "dashscope")?;
    let mut config = DashScopeAsrConfig::new(file.model, required_key()?);
    if let Some(endpoint) = file.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(seconds) = file.timeout_secs {
        config = config.with_timeout(std::time::Duration::from_secs(seconds));
    }
    DashScopeAsr::new(config, backends.cloud()?)
}

fn openai(backends: &Backends, args: &BackendArgs) -> Result<OpenAiTranscription, SpeechError> {
    let file: OpenAiFile = config_file(args.config.as_deref(), "openai-http")?;
    file.check()?;
    let mut config = OpenAiTranscriptionConfig::new(file.model)
        .with_endpoint(file.base_url)
        .with_streaming(file.stream);
    if let Some(key) = api_key() {
        config = config.with_api_key(key);
    }
    if let Some(language) = file.language {
        config = config.with_language(language);
    }
    if let Some(prompt) = file.prompt {
        config = config.with_prompt(prompt);
    }
    if let Some(seconds) = file.timeout_secs {
        config = config.with_timeout(std::time::Duration::from_secs(seconds));
    }
    OpenAiTranscription::new(config, backends.cloud()?)
}

/// A sherpa-onnx recognizer from `config` and the shared flags.
fn sherpa(config: AsrConfig, args: &BackendArgs) -> Result<Asr, SpeechError> {
    let inference = Inference::default()
        .with_provider(args.provider.parse()?)
        .with_threads(args.threads);
    let mut config = config.with_inference(inference);
    if !args.hotwords.is_empty() {
        config = config.with_hotwords(args.hotwords.iter().map(String::as_str));
    }
    config.load()
}

impl BackendFactory for Backends {
    fn asr(&self, args: &BackendArgs) -> Result<AsrEngine, SpeechError> {
        let vad = |path: &Path| SileroVadConfig::new(path).load();
        let built = match args.backend {
            BackendKind::SherpaStreaming => {
                let dir = required(args.model.as_deref(), "--model DIR", "sherpa-streaming")?;
                engine(sherpa(AsrConfig::streaming(dir), args)?)
            }
            BackendKind::SherpaOffline => {
                let dir = required(args.model.as_deref(), "--model DIR", "sherpa-offline")?;
                let vad_file = required(args.vad.as_deref(), "--vad FILE", "sherpa-offline")?;
                let mut config = AsrConfig::offline(dir, vad_file);
                if let Some(family) = &args.family {
                    config = config.with_family(family.parse()?);
                }
                engine(sherpa(config, args)?)
            }
            BackendKind::OpenaiRealtime => engine(realtime(self, args)?),
            BackendKind::Dashscope => engine(dashscope(self, args)?),
            BackendKind::OpenaiHttp => match &args.vad {
                Some(file) => engine(VadBackend::new(openai(self, args)?, vad(file)?)?),
                None => engine(openai(self, args)?),
            },
        };
        match &args.punct {
            Some(dir) => Ok(built.with_post_processor(PunctuationConfig::new(dir).load()?)),
            None => Ok(built),
        }
    }

    fn tts(&self, args: &TtsBackendArgs) -> Result<TtsEngine, SpeechError> {
        crate::tts_backends::build(self, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cloud_runtime_starts_on_first_use() {
        let backends = Backends::default();
        assert!(backends.cloud.get().is_none());
        backends.cloud().unwrap();
        assert!(backends.cloud.get().is_some());
        backends.cloud().unwrap();
    }

    #[test]
    fn response_format_points_to_stream() {
        let file: OpenAiFile = serde_json::from_str(
            r#"{"base_url": "http://x/v1", "model": "m", "response_format": "json"}"#,
        )
        .unwrap();
        let error = file.check().unwrap_err();
        assert!(error.to_string().contains("\"stream\": true"), "{error}");
        let file: OpenAiFile =
            serde_json::from_str(r#"{"base_url": "http://x/v1", "model": "m", "stream": true}"#)
                .unwrap();
        assert!(file.check().is_ok());
        assert!(file.stream);
    }
}
