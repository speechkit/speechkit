# speechkit

A Rust toolkit for speech recognition (ASR) and speech synthesis (TTS).

- **Local and cloud backends.** sherpa-onnx for local recognition (streaming Zipformer; SenseVoice, Paraformer, Qwen3-ASR, and more behind silero VAD; hotwords and punctuation) and synthesis (VITS/Piper, Matcha, Kokoro). In the cloud, OpenAI-compatible transcription, OpenAI Realtime, and OpenAI speech, plus DashScope recognition and synthesis.
- **A command and a server.** `speechkit transcribe`, `stream`, `mic`, `speak`, `voices`, and `devices`, and `speechkit serve`, an OpenAI-compatible HTTP API for transcription and speech.

speechkit is not affiliated with Yandex SpeechKit.

## Design

|              | One call                | Session (streaming)                                | Device                                                    |
| ------------ | ----------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| audio → text | `AsrEngine::transcribe` | `start` → `push` → `updates` → `finish`            | `Microphone::listen`, `Capture::listen`, `Capture::watch` |
| text → audio | `TtsEngine::synthesize` | `start` → `push_text` → read the output → `finish` | `Speaker::speak`, `speak_streaming`, `play`, `sink`       |

- **One lifecycle.** Every interface goes start (take a bounded resource) → push (input, with backpressure) → read (with a deadline) → finish or cancel → result. A failure still carries the progress made.
- **Engines and sessions.** A backend is a loaded model or a service client; an engine wraps it and limits how many sessions run at once; a session is one recognition or one synthesis. Applications hold engines and sessions, and the same engine type works with every backend.
- **Waiting is explicit.** A method that can wait takes a deadline, a `Duration` or an `Instant`; its `try_` twin never waits. A timeout only stops waiting, except on `finish`, where it ends the session. Nothing has a hidden timeout.
- **A microphone is one timeline.** A `Capture` counts capture time and keeps the last few seconds. Every reader names the position it starts at and reports capture times, so key-down to session, a session that connects late, and wake word to request all hand off with no gap. A heard wake word reserves the audio after it.
- **Speech activity is data.** Backends report where speech starts and ends, and every reader receives every event. Ending at a pause, turn ends, and barge-in are built on those events; nothing guesses where speech is.
- **Nothing is lost silently.** A refused chunk comes back, a reader never skips audio, a failure keeps what was confirmed, and a stopped playback reports the text the listener heard. Every stream ends with `Closed`, carrying the result.
- **Honest about native code.** Each session runs on its own thread and holds its slot until its backend has stopped. A backend panic fails its session, not the process, and a model can run in a worker process so a native crash fails only one session.
- **Policy stays with the application.** The library never retries, falls back, or splits a session. It reports exact positions, events, and whether an error is worth retrying.
- **Private by default.** A session keeps no audio after processing it, and audio, transcripts, and synthesized text are never logged above `trace`.

Each rule is a named test in `speechkit-testkit`, run against every backend and device. [docs/guide.md](docs/guide.md) walks through eleven applications, from a dictation input method to a voice agent the user can interrupt.

For FunASR Nano INT8 returning empty text or unrelated phrases on some x86 CPUs,
see the [model troubleshooting guide](docs/funasr-nano.md) and its offline
conversion script.

## Crates

| Crate | Contents |
|---|---|
| [`speechkit`](crates/speechkit) | The library: audio types, the ASR and TTS engines, the VAD adapter, and, behind features, sherpa-onnx and cloud backends, microphone and speaker, and an OpenAI-compatible server |
| [`speechkit-cli`](crates/speechkit-cli) | The `speechkit` command, and the `speechkit-worker` crash-isolation process |

## Status

The current release is 0.5.0. The API may still change in minor releases until 1.0; [CHANGELOG.md](CHANGELOG.md) says what changed and how to migrate, and [AGENTS.md](AGENTS.md) explains how to work in the repository.

## Development

```sh
cargo test --workspace --all-features
cargo xtask --help
```

Tests need no models, keys, or network. Tests against real models are ignored; `cargo xtask fetch-fixtures` downloads the models they use and prints the `SPEECHKIT_MODEL_*` variables to set.

## License

MIT. See [LICENSE](LICENSE).
