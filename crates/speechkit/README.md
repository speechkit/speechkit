# speechkit

A Rust toolkit for speech recognition (ASR) and speech synthesis (TTS). The core is always built; backends, audio devices, and the server are behind features.

speechkit is not affiliated with Yandex SpeechKit.

| Feature | Contents |
|---|---|
| (always) | recognition and synthesis engines, sessions, WAV files (any depth, downmixed), the VAD adapter |
| `sherpa` | `speechkit::sherpa`: local sherpa-onnx recognition, synthesis, silero VAD, and punctuation |
| `sherpa-shared` | link sherpa-onnx dynamically instead of statically |
| `openai`, `dashscope` | `speechkit::cloud`: OpenAI and DashScope backends |
| `devices` | `speechkit::io`: microphones and speakers |
| `server` | `speechkit::server`: an OpenAI-compatible HTTP server (turns on `decode`) |
| `decode` | FLAC, MP3, AAC, Ogg, and Matroska files in `speechkit::audio` |
| `serde` | serde support for the public types |
| `full` | everything except `sherpa-shared` and `serde` |

No feature is on by default.

```rust
use std::time::Duration;
use speechkit::{AudioBuffer, SampleRate};
use speechkit::asr::{AsrEngine, AsrOptions};
use speechkit::vad::{EnergyVad, EnergyVadConfig, OfflineRecognizer, VadBackend};

/// A stand-in recognizer; use a real backend such as `speechkit::sherpa`.
struct Echo(speechkit::asr::AsrCapabilities);

impl OfflineRecognizer for Echo {
    fn name(&self) -> &str { "echo" }
    fn capabilities(&self) -> &speechkit::asr::AsrCapabilities { &self.0 }
    fn recognize(&self, samples: &[f32], _: &AsrOptions) -> Result<String, speechkit::SpeechError> {
        Ok(format!("{} samples", samples.len()))
    }
}

let recognizer = Echo(speechkit::asr::AsrCapabilities::new(SampleRate::HZ_16000));
let backend = VadBackend::new(recognizer, EnergyVad::new(EnergyVadConfig::default()))?;
let engine = AsrEngine::new(backend);

let tone: Vec<f32> = (0..16_000).map(|i| 0.3 * (i as f32 * 0.17).sin()).collect();
let audio = AudioBuffer::new(SampleRate::HZ_16000, tone);
let transcript = engine
    .transcribe(&audio, AsrOptions::default(), Duration::from_secs(10))
    .map_err(|failure| failure.error)?;
assert_eq!(transcript.segments.len(), 1);
# Ok::<(), speechkit::SpeechError>(())
```
