//! Turning the synthesis `--backend` and its options into an engine.

use std::time::Duration;

use serde::Deserialize;
use speechkit::cloud::{DashScopeTts, DashScopeTtsConfig, OpenAiSpeech, OpenAiSpeechConfig};
use speechkit::sherpa::{Inference, TtsConfig};
use speechkit::{
    SampleRate, SpeechError,
    tts::{TtsEngine, Voice},
};

use crate::{
    Backends, TtsBackendArgs, TtsBackendKind,
    backends::{api_key, config_file, required, required_key},
};

/// `--config` for openai.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenAiFile {
    base_url: String,
    model: String,
    #[serde(default)]
    voices: Option<Vec<String>>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

/// `--config` for dashscope.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DashScopeFile {
    model: String,
    voices: Vec<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    sample_rate: Option<u32>,
    #[serde(default)]
    timeout_secs: Option<u64>,
}

fn voices(ids: Vec<String>) -> Vec<Voice> {
    ids.into_iter().map(Voice::new).collect()
}

fn openai(backends: &Backends, args: &TtsBackendArgs) -> Result<OpenAiSpeech, SpeechError> {
    let file: OpenAiFile = config_file(args.config.as_deref(), "openai")?;
    let mut config = OpenAiSpeechConfig::new(file.model).with_endpoint(file.base_url);
    if let Some(key) = api_key() {
        config = config.with_api_key(key);
    }
    if let Some(ids) = file.voices {
        config = config.with_voices(voices(ids));
    }
    if let Some(hz) = file.sample_rate {
        config = config.with_sample_rate(SampleRate::new(hz)?);
    }
    if let Some(seconds) = file.timeout_secs {
        config = config.with_timeout(Duration::from_secs(seconds));
    }
    OpenAiSpeech::new(config, backends.cloud()?)
}

fn dashscope(backends: &Backends, args: &TtsBackendArgs) -> Result<DashScopeTts, SpeechError> {
    let file: DashScopeFile = config_file(args.config.as_deref(), "dashscope")?;
    let mut config = DashScopeTtsConfig::new(file.model, required_key()?, voices(file.voices));
    if let Some(endpoint) = file.endpoint {
        config = config.with_endpoint(endpoint);
    }
    if let Some(hz) = file.sample_rate {
        config = config.with_sample_rate(SampleRate::new(hz)?);
    }
    if let Some(seconds) = file.timeout_secs {
        config = config.with_timeout(Duration::from_secs(seconds));
    }
    DashScopeTts::new(config, backends.cloud()?)
}

fn engine(backend: impl speechkit::tts::TtsBackend) -> TtsEngine {
    TtsEngine::new(backend)
}

/// Builds the synthesis engine `args` describe.
pub(crate) fn build(backends: &Backends, args: &TtsBackendArgs) -> Result<TtsEngine, SpeechError> {
    Ok(match args.backend {
        TtsBackendKind::Sherpa => {
            let dir = required(args.model.as_deref(), "--model DIR", "sherpa")?;
            let inference = Inference::default()
                .with_provider(args.provider.parse()?)
                .with_threads(args.threads);
            engine(TtsConfig::new(dir).with_inference(inference).load()?)
        }
        TtsBackendKind::Openai => engine(openai(backends, args)?),
        TtsBackendKind::Dashscope => engine(dashscope(backends, args)?),
    })
}
