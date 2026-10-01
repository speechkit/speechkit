# Changelog

All notable changes to speechkit. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [0.5.0] - 2026-09-30

This release redesigns the public API around one contract per direction, before 1.0. Almost all of it is breaking; the migration table below maps each 0.4 item to its replacement. The rules are the contract checks in `speechkit-testkit`: `A-xx` for recognition, `T-xx` for synthesis, and the new `D-xx` for devices.

### Added

- **core:** `Deadline`, taken by every waiting method as `impl Into<Deadline>`: a `Duration` from when the call starts, or an `Instant` shared by several calls.
- **asr:** Speech activity (`SpeechStarted`, `SpeechEnded`) from every backend that can tell, and endpointing: `AsrOptions::with_turn_end` reports `TurnEnded(Turn)`, and `with_end_after_silence`, `with_no_speech_timeout`, and `with_max_length` end a session at a point in the audio (A-07, A-08). A session has any number of readers (`updates()`), which never slow it, and `wait` returns its result once it ended by itself.
- **tts:** A synthesis is a text side and an audio side: `TtsEngine::start` returns `(TtsSession, TtsOutput)`, and the output carries `Mark`s that map audio back to the text.
- **io:** One capture timeline. `Capture::listen` starts a `Listening` whose session opens in the background, from a position still held (`ListenOptions::starting_at`, up to `CaptureOptions::with_history`); `Capture::watch(&model)` runs a wake-word detector (`watch_with` takes `WatchOptions`), and each `Wake` reserves the audio after its keyword. A `Speaker` queues sounds (`speak`, `speak_streaming`, `play`, `sink`), and each `Playback` reports `played()` and `text_played()`, what the listener actually heard.
- **sherpa:** `sherpa::inspect` reports a model directory's kind without loading it. `IsolatedAsr` is a pool of worker processes, one session each, so a crash fails only its session. Streaming recognizers take `with_endpoint_silence` and `with_max_utterance`.
- **cloud:** `OpenAiRealtimeConfig::with_vad` runs the application's VAD for models that refuse server VAD, and `with_activity_lag` bounds how late the service reports speech. Realtime streams reconnect on their own before OpenAI's 60-minute session limit. DashScope streams send `heartbeat: true`.
- **audio:** `audio::read` and `decode` read any WAV (any integer depth up to 32 bits, or float, downmixed) without the `decode` feature.

### Changed

- [**breaking**] Every renamed or reshaped item is in the migration table below.
- **asr:** [**breaking**] `start` waits for a slot within its deadline instead of failing at once (the server still answers 503 at once). A full history fails the session with `Capacity`, and every refused push returns its chunk.
- **tts:** [**breaking**] `TtsSession::finish` waits until the reader has taken all the audio. Dropping the session before `close_text`, or dropping the output, cancels the synthesis (T-02).
- **io:** [**breaking**] The `capture` and `playback` features are one feature, `devices`.
- **cloud:** Streaming connections are read as messages arrive, so events reach a session while no audio is pushed (A-10), and a dropped connection fails it at once.
- **contract:** The rules are renumbered: A-01…A-19 for recognition, T-01…T-11 for synthesis, and D-01…D-06 for devices.

### Removed

- **asr:** `TranscriptJoin`, `text_with`, `Outcome`, and the `max_chunk`, `max_session`, and `observer_queue` limits.
- **sherpa:** `SherpaOffline`, `offline_model`, `Hotword` (`with_hotwords` takes strings), `AsrFamily::detect`, and the keyword spotter's `with_variant`, `with_unit`, `KeywordUnit`, and `Keyword::read_file`.
- **cloud:** The upload and response limit setters of the OpenAI transcription config, and `OpenAiSpeechConfig::with_max_input_chars`.
- **server:** `with_header_timeout`, `with_body_timeout`, and `with_max_speech_chars`; their defaults stay.
- **io:** `input_devices`, `output_devices`, `Player`, and `Capture::attach`.
- **core:** `Voice::speaks` and `SampleRate::{MIN_HZ, MAX_HZ}` are private.

### Migration

| 0.4 | 0.5.0 |
|---|---|
| deadlines as `Instant`; `start_until`, `listen_until` | `impl Into<Deadline>` on every waiting method; `start(.., deadline)` |
| `asr::RecvError`, `tts::Flow` | `speechkit::RecvError`, `speechkit::Flow` |
| `SessionOptions` | `AsrOptions` |
| `AsrSessionLimits` | `AsrLimits {input_queue, max_history_bytes}`; the backend is fed in fixed 100 ms blocks |
| `AsrCapabilities {preferred_sample_rate, partial_results, session_hints, language_override, native_punctuation}` | `{sample_rate, reports_partials, reports_activity, accepts_hints, accepts_language, punctuated}` |
| `AsrSession::push_until` | `push(chunk, deadline)` |
| `subscribe() -> Observer` | `updates() -> AsrUpdates` (`recv`, `try_recv`, `Iterator`) |
| `queued_frames()` | `queued() -> Duration` |
| `Update` | `AsrUpdate` |
| `Outcome` / `SessionFailure` / `SessionResult` | `Transcript` / `AsrFailure` / `AsrResult` |
| `RecognizerStream`, `BackendEvent` | `AsrStream`, `AsrEvent`; backends send through `AsrEvents` |
| `SegmentPostProcessor` | `PostProcessor` |
| `SynthOptions::with_output_rate` | `TtsOptions::with_sample_rate` |
| `TtsSessionLimits`, `TtsCapabilities` fields | `TtsLimits {output_queue, max_text_chars, chunk_chars}`, `{sample_rate, streams_audio, speed, max_chunk_chars}` |
| `SynthesisSession`, `next_audio`, `iter` | `(TtsSession, TtsOutput)` from `start`; `TtsUpdate {Audio, Mark, Closed}` |
| `ChunkTiming` | `Mark` |
| `SynthOutcome` / `SynthFailure` / `SynthResult` / `SynthStream` | `TtsSummary` / `TtsFailure` / `TtsResult` / `TtsStream` |
| `read_wav_pcm16`, `decode_wav_pcm16`, `read_audio`, `decode_audio` | `audio::read`, `audio::decode` |
| `encode_wav_pcm16`, `DECODE_EXTENSIONS` | `encode_wav`, `EXTENSIONS` |
| `VadFactory`, `WakeWordFactory`, `WakeWord` | `VadModel`, `WakeWordModel`, `WakeWordDetector` |
| `EnergyVadConfig::max_speech`, `SileroVadConfig::max_speech` | `VadModel::create(max_speech)`; `VadBackend::with_max_utterance` |
| `SherpaAsrConfig`, `SherpaAsr::load(&config)` | `sherpa::AsrConfig`, `config.load()` |
| `SherpaTtsConfig`, `KwsConfig`, `SherpaPunctuation` | `TtsConfig`, `KeywordSpotterConfig`, `PunctuationConfig`, each with `validate` and `load` |
| `SileroVad::load(config)` | `SileroVadConfig::load()` |
| `TtsModel`, `PunctuationModel` | `TtsFamily`, `PunctuationFamily` |
| `ExecutionProvider`, `Inference::with_num_threads` | `Provider`, `Inference::with_threads` |
| `process::ProcessBackend` | `process::IsolatedAsr` (`spawn`, `spawn_command`) |
| `OpenAiHttp`, `OpenAiHttpConfig::new(base_url, model)` | `OpenAiTranscription`, `OpenAiTranscriptionConfig::new(model).with_endpoint(..)` |
| `OpenAiSpeechConfig::new(base_url, model)` | `OpenAiSpeechConfig::new(model).with_endpoint(..)` |
| features `capture`, `playback` | feature `devices` |
| `io::input_devices()`, `io::output_devices()` | `Microphone::list()`, `Speaker::list()` |
| `Microphone::listen(&engine, options, deadline) -> Capture` | `Microphone::listen(&engine, options) -> Listening` |
| `Microphone::capture` then `Capture::attach(session)` | `capture.listen(&engine, options, ListenOptions)` |
| `CaptureOptions::{max_backlog, recording}` | `ListenOptions::{with_max_backlog, with_recording}`; `CaptureOptions::with_history` |
| `Capture::{finish, cancel, recording, updates}` | the same on `Listening`; `recording()` returns a `Recording {audio, truncated}` |
| `Player::speak(&engine, text, options, deadline)` | `speaker.speak(&engine, text, options)?.finish(deadline)` |
| `Player::play(output, deadline)` | `speaker.play(output)?.finish(deadline)` |
| `with_max_sessions(NonZeroUsize)` | `with_max_sessions(usize)`; 0 is raised to 1 |

## [0.4.0] - 2026-09-28

This release lets a push-to-talk app record from the moment the key goes down, and gives a model picker what it needs without loading a model. `Listening` became `Capture`, which is breaking.

### Added

- **io:** `Microphone::capture` starts the microphone with no session and returns a `Capture`; `Capture::attach` gives it a session later, which receives the held audio first and live audio after it, even after `stop`. `CaptureOptions` sets the backlog limit (`max_backlog`, 30 s by default) and an opt-in `recording`, which `Capture::recording` returns for a retry. `Capture::level` gives the level of the latest audio, for a meter.
- **asr:** `LiveTranscript` builds the text to show while a session runs: the committed segments followed by the words still being recognized.
- **sherpa:** `AsrFamily::detect` lists every family a model directory fits. `AsrFamily::supports_hotwords`, `supports_hotword_boost`, and `languages` tell what a family takes before a model is loaded.
- **sherpa:** `SherpaTtsConfig::validate` returns the `TtsModel`, `SherpaPunctuation::validate` the `PunctuationModel`, and `SileroVadConfig::validate` and `KwsConfig::validate` check their models, all without loading anything native. `KwsConfig::validate` also checks that the model can spell every keyword. Both model enums print their names.
- **sherpa:** `ExecutionProvider::is_supported` says whether the platform can have a provider (CUDA on Linux and Windows, CoreML on Apple platforms).
- **audio:** `DECODE_EXTENSIONS` lists the file extensions `read_audio` reads, for a file dialog. The list of formats now names Ogg FLAC, which already decoded.

### Changed

- **io:** [**breaking**] `Listening` is `Capture`, and `Capture::session` returns `Option<&AsrSession>`. `Microphone::listen` and `listen_until` start the microphone before they open the session, so words spoken during a cloud handshake are kept. To migrate, rename `Listening` to `Capture` and unwrap `session()` after `listen`, which always attaches one.
- **contract:** C-10 says a session never keeps input audio; only a `Capture` records audio, when its options ask for it, and it hands the recording to the caller.

### Fixed

- **io:** A session that falls behind the microphone fails with retryable `Capacity` and keeps its committed segments, instead of losing audio from the middle or end of the transcript and ending `Ok`. A `max_chunk` below 100 ms no longer loses all audio.

## [0.3.0] - 2026-09-28

This release makes the public API smaller and simpler before 1.0, and most of it is breaking. The [API reference](docs/book/src/api-reference.md) lists every public item and its default.

### Added

- **docs:** An API reference in the book, listing every public item with what it does and its default.
- **sherpa:** `SherpaAsrConfig` and `SherpaAsr` load every recognition family. The family is found from the files where they tell it apart, an offline family gets its silero VAD from the config, and hotwords become decoding bias for transducers or a prompt for Qwen3-ASR and FunASR-Nano.
- **io:** `Microphone::listen` and `listen_until` start a session fed from the microphone and return a `Listening`; `Player::speak` synthesizes text and plays it as it is produced.
- **server:** A `Server` builder that serves a recognition engine, a synthesis engine, or both. A route without an engine answers 404.

### Changed

- **api:** Each public item has one path. Modules that only group code are private, and their items are re-exported from the crate root, `asr`, `tts`, `sherpa`, and `cloud`. Items nothing outside the library used are private or removed.
- **core:** `AsrEngine::new(backend)` and `TtsEngine::new(backend)` take the backend alone; use `with_max_sessions` and `with_limits` for the rest. An `Arc` or `Box` of a backend is itself a backend. `AudioSpec` is replaced by `SampleRate`, and a session's rate is an argument of `start`. `SessionOptions::new(spec)` is `SessionOptions::default()`.
- **asr:** `session.input().try_push`, `push_wait`, and `close` are `session.try_push`, `push_until`, and `close_input`. `finish` returns a `SessionResult` or `SynthResult` by value, `result` lends one, and `SpeechError` is `Clone`. An observer that falls behind catches up from the committed segments instead of receiving `Update::Reset`, so each segment arrives once and in order (contract rules C-03, C-04, C-07, C-09, and C-12 changed).
- **tts:** The audio iterator no longer has a hidden 60 s timeout.
- **cloud:** API keys are `impl Into<Arc<Secret>>`. `OpenAiHttpConfig::new` and `OpenAiSpeechConfig::new` take no key; add one with `with_api_key`. `OpenAiHttpConfig::with_streaming(bool)` replaces `ResponseFormat`, and the CLI's `openai-http` config file takes `"stream": true` instead of `"response_format"`.
- **sherpa:** `SileroVadFactory` is `SileroVad`. `SherpaOffline` loads from `SherpaAsrConfig::offline_model` and also covers SenseVoice. `ProcessBackend::worker` takes a `SherpaAsrConfig`, so crash isolation covers hotwords, language, and VAD settings. The worker protocol is version 2, so rebuild `speechkit-worker`.
- **io:** `Microphone::default_input` is `Microphone::open_default`, and `Player::rate` is `sample_rate`. A lost microphone or speaker is logged at `warn` instead of reported on an event channel.
- **cli:** `--hotwords` is refused for families that cannot use hotwords instead of being ignored, and a flat model without SenseVoice markers needs `--family` instead of loading as SenseVoice.

### Removed

- **core:** `EngineLimits`, `AudioSpec`, `SpeechHints`, `session_limits()`, `AsrError`, `TtsError`, `AudioChunk`, `InvalidSample`, `SampleErrorKind`, and `SpeechError::duplicate`.
- **asr, tts:** `InputHandle`, `Update::Reset`, `Snapshot`, `AsrCapabilities::engine_hints`, `TtsCapabilities::incremental_text`, the public `SessionFailure::new` and `SynthFailure::new`, `asr::validate`, and `tts::{validate, Chunker, TextChunk}`.
- **audio:** `Recorder`, `rms`, `f32_to_pcm16`, `pcm16_to_f32`, `pcm16_bytes`, and `SUPPORTED_FORMATS`.
- **sherpa:** `SherpaStreaming`, `StreamingConfig`, `SenseVoice`, `SenseVoiceConfig`, `SenseVoiceLanguage`, `OfflineConfig`, `WorkerConfig`, the `layout` and `bias` modules, the `layout` accessors, `ModelingUnit` and `with_modeling_unit` (the unit is found from the files), `KeywordTokenizer`, `SherpaKws::keywords`, and the per-part `validate` methods.
- **cloud:** `ResponseFormat`, `OPENAI_VOICES`, `OpenAiHttp::config`, `OpenAiSpeech::config`, `CloudRuntime::handle`, and `CloudRuntime::is_owned`.
- **io:** `Microphone::start`, `Capture`, `CaptureEvent`, `PlaybackEvent`, and `io::convert`.
- **server:** `AppState`, `ServeOptions`, `TextDeltaBuilder`, and the free functions `router`, `start`, and `run`.
- **vad:** `VadBackend::recognizer`.
- **speech:** `EngineManager` and `Generation`. Engines are cheap clones, so a host swaps them with `ArcSwap` or a lock. The `arc-swap` dependency is gone.

## [0.2.1] - 2026-09-27

### Added

- **io:** Select audio devices by name.
- **sherpa:** Wake-word detection with keyword spotting.

### Changed

- **release:** Release archives no longer include `.sha256` files.
- **release:** Release archives no longer include `speechkit-worker`; `ProcessBackend` users build it with their own speechkit version.

## [0.2.0] - 2026-09-26

The first release of speechkit, a Rust library and CLI for speech recognition and speech synthesis. `0.1.0` was an empty placeholder; it is yanked, and this release is `0.2.0` rather than `0.1.1` because removing the placeholder's one function is a breaking change.

### Added

- **Recognition engine.** `AsrEngine` sessions with bounded input and backpressure, deadlines, cancellation, a single observer of partial results, and session limits, following the recognition contract.
- **Synthesis engine.** `TtsEngine` sessions that take text incrementally, split it into chunks, and stream audio through a bounded queue, following the synthesis contract.
- **Local backends** (feature `sherpa`, on sherpa-onnx): streaming Zipformer; SenseVoice, Paraformer, offline transducer, Qwen3-ASR, FunASR-Nano, and FireRedASR2 behind silero VAD; hotwords and prompt hints; punctuation; VITS/Piper, Matcha, and Kokoro voices; CPU, CUDA, and CoreML; model layout detection that validates a directory before loading it.
- **Crash isolation.** `sherpa::process::ProcessBackend` runs recognition in the `speechkit-worker` process, so a native abort fails one session instead of the host.
- **Cloud backends** (features `openai` and `dashscope`): OpenAI-compatible transcription (JSON, text, and SSE), OpenAI Realtime, and OpenAI speech; DashScope Paraformer real-time recognition and CosyVoice synthesis. Errors say whether they are retryable; speechkit never retries.
- **Audio.** WAV reading and writing, decoding of MP3, AAC, FLAC, Ogg Vorbis, and more (feature `decode`), resampling, an energy VAD, and engine hot reload.
- **Devices.** Microphone capture and speaker playback (features `capture` and `playback`).
- **Server** (feature `server`): an OpenAI-compatible HTTP API with `POST /v1/audio/transcriptions` (including streaming), `POST /v1/audio/speech`, `GET /v1/models`, and `GET /health`, with bearer-token authentication and limits on uploads, sessions, and slow clients.
- **CLI** (`cargo install speechkit-cli`): `speechkit transcribe`, `stream`, `mic`, `serve`, `speak`, and `voices`.
