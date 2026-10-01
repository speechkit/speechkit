Local speech recognition and synthesis, built on [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) (feature `sherpa`): streaming Zipformer, SenseVoice and other offline models behind silero VAD, keyword spotting for wake words, punctuation, and TTS models.

One configuration, [`AsrConfig`], describes every recognition model: [`streaming`](AsrConfig::streaming) for a streaming transducer, or [`offline`](AsrConfig::offline) with a silero VAD model for the rest. It finds the family from the files where they tell it apart, and [`load`](AsrConfig::load) loads the result. Each other kind of model has a config of its own, [`TtsConfig`], [`KeywordSpotterConfig`], [`PunctuationConfig`], and [`SileroVadConfig`], with `validate` to check it without native code and `load` to load it. [`inspect`] tells a model picker what a directory holds.

```toml
[dependencies]
speechkit = { version = "0.4", features = ["sherpa"] }
```

## Linking

By default the build script of `sherpa-onnx-sys` downloads prebuilt static libraries for your platform into `target/sherpa-onnx-prebuilt` and links them into your binary. Nothing needs to be installed.

With the `sherpa-shared` feature, sherpa-onnx is linked dynamically. Use this for GPU builds: set `SHERPA_ONNX_LIB_DIR` to a directory holding a CUDA-enabled (Linux, Windows) or CoreML-enabled (macOS) build of the sherpa-onnx C API before building, and ship those libraries next to your binary.

On Linux, the `devices` feature also needs `libasound2-dev` and `pkg-config`.

In CI, a cache that restores `target/` without the downloaded libraries leaves an empty `target/sherpa-onnx-prebuilt` that the build script treats as complete. Remove that directory and run `cargo clean -p sherpa-onnx-sys` after restoring the cache; `.github/actions/native` in this repository does exactly that.

## Models

Every loader checks the directory's layout, file names, and non-empty files before it calls into sherpa-onnx, and reports mistakes as `SpeechError::InvalidModel`.

## Corrupt models abort the process

A model file with the right name but corrupt content is loaded by the native library, which **aborts the whole process** (SIGABRT) instead of returning an error. This holds for recognizers, the VAD, and punctuation models (see `tests/sherpa_corrupt_model.rs`). speechkit cannot catch an abort.

If you load models you do not control, such as user-supplied directories, run recognition in a separate process with [`process::IsolatedAsr`]. It runs each session on a `speechkit-worker` process from a pool, started on demand and reused, and talks to it over stdin and stdout. If a worker dies mid-session, only that session fails, with a retryable error, and the next session starts a new worker.

## Example

```rust,no_run
use std::time::Duration;
use speechkit::{SampleRate, asr::{AsrEngine, AsrOptions}};
use speechkit::sherpa::AsrConfig;

// Streaming Zipformer; for SenseVoice, use
// `AsrConfig::offline("models/sense-voice", "models/silero_vad.onnx")`.
let config = AsrConfig::streaming("models/sherpa-onnx-streaming-zipformer-en-2023-06-26");
let engine = AsrEngine::new(config.load()?);
let session = engine.start(SampleRate::HZ_16000, AsrOptions::default(), Duration::from_secs(10))?;
# Ok::<(), speechkit::SpeechError>(())
```
