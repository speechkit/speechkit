//! What a `speechkit-worker` runs: a [`AsrConfig`] as JSON on its
//! command line.

use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{
    SpeechError,
    sherpa::{AsrConfig, Inference, SileroVadConfig},
};

/// A [`AsrConfig`] on the wire. It changes with the protocol
/// version, so it is not part of the API. Both conversions name every
/// field, so a new setting cannot compile without crossing the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WireConfig {
    model: PathBuf,
    family: Option<String>,
    vad: Option<WireVad>,
    provider: String,
    threads: usize,
    hotwords: Option<Vec<String>>,
    language: Option<String>,
    endpoint_silence: Duration,
    max_utterance: Duration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct WireVad {
    model: PathBuf,
    threshold: f32,
    min_silence: Duration,
    min_speech: Duration,
}

impl From<&AsrConfig> for WireConfig {
    fn from(config: &AsrConfig) -> Self {
        let AsrConfig {
            model,
            family,
            vad,
            inference,
            hotwords,
            language,
            endpoint_silence,
            max_utterance,
        } = config;
        let Inference { provider, threads } = inference;
        Self {
            model: model.clone(),
            family: family.map(|family| family.as_str().to_owned()),
            vad: vad.as_ref().map(
                |SileroVadConfig {
                     model,
                     threshold,
                     min_silence,
                     min_speech,
                 }| WireVad {
                    model: model.clone(),
                    threshold: *threshold,
                    min_silence: *min_silence,
                    min_speech: *min_speech,
                },
            ),
            provider: provider.as_str().to_owned(),
            threads: *threads,
            hotwords: hotwords.clone(),
            language: language.clone(),
            endpoint_silence: *endpoint_silence,
            max_utterance: *max_utterance,
        }
    }
}

impl TryFrom<WireConfig> for AsrConfig {
    type Error = SpeechError;

    fn try_from(wire: WireConfig) -> Result<Self, SpeechError> {
        Ok(Self {
            model: wire.model,
            family: wire.family.as_deref().map(str::parse).transpose()?,
            vad: wire.vad.map(|vad| SileroVadConfig {
                model: vad.model,
                threshold: vad.threshold,
                min_silence: vad.min_silence,
                min_speech: vad.min_speech,
            }),
            inference: Inference {
                provider: wire.provider.parse()?,
                threads: wire.threads,
            },
            hotwords: wire.hotwords,
            language: wire.language,
            endpoint_silence: wire.endpoint_silence,
            max_utterance: wire.max_utterance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sherpa::AsrFamily;

    #[test]
    fn configs_survive_the_wire() {
        let configs = [
            AsrConfig::streaming("m")
                .with_hotwords(["speechkit"])
                .with_endpoint_silence(Duration::from_millis(700))
                .with_max_utterance(Duration::from_secs(30)),
            AsrConfig::offline("m", "v.onnx")
                .with_family(AsrFamily::Paraformer)
                .with_vad(
                    SileroVadConfig::new("v.onnx")
                        .with_min_silence(Duration::from_micros(300_500))
                        .with_min_speech(Duration::from_nanos(250_000_001)),
                )
                .with_inference(Inference::default().with_threads(4))
                .with_language("zh"),
        ];
        for config in configs {
            let json = serde_json::to_string(&WireConfig::from(&config)).unwrap();
            let back: WireConfig = serde_json::from_str(&json).unwrap();
            assert_eq!(AsrConfig::try_from(back).unwrap(), config, "{json}");
        }
        assert!(serde_json::from_str::<WireConfig>(r#"{"model":"m","x":1}"#).is_err());
    }

    #[test]
    fn bad_settings_are_refused_before_the_worker_starts() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["encoder.onnx", "decoder.onnx", "joiner.onnx"] {
            std::fs::write(dir.path().join(name), "x").unwrap();
        }
        std::fs::write(dir.path().join("tokens.txt"), "x 1\n").unwrap();
        let config = AsrConfig::streaming(dir.path()).with_hotwords(["unknown"]);
        let error =
            super::super::IsolatedAsr::spawn("/nonexistent/speechkit-worker", &config).unwrap_err();
        assert!(error.to_string().contains("does not know"), "{error}");
    }
}
