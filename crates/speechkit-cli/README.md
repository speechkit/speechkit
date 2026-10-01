# speechkit-cli

The `speechkit` command: transcribe files, stream audio with backpressure, synthesize speech, and serve an OpenAI-compatible API, with local sherpa-onnx models or cloud services.

```sh
cargo install speechkit-cli

speechkit transcribe speech.mp3 --backend sherpa-streaming --model models/sherpa-onnx-streaming-zipformer-en-2023-06-26
speechkit transcribe speech.wav --backend sherpa-offline --model models/sense-voice --vad models/silero_vad.onnx --format srt
SPEECHKIT_API_KEY=sk-... speechkit transcribe speech.wav --backend openai-http --config openai.json
speechkit serve --backend sherpa-offline --model models/sense-voice --vad models/silero_vad.onnx --bind 127.0.0.1:8080

speechkit speak "Hello there." --model models/vits-piper-en_US-amy-low --out hello.wav
speechkit speak --file story.txt --model models/kokoro-en-v0_19 --voice 3 --speed 1.2 --play
speechkit voices --model models/kokoro-en-v0_19
speechkit serve --backend sherpa-offline --model models/sense-voice --vad models/silero_vad.onnx --tts-model models/vits-piper-en_US-amy-low

speechkit devices
speechkit mic --device "BlackHole 2ch" --backend sherpa-streaming --model models/sherpa-onnx-streaming-zipformer-en-2023-06-26
speechkit speak "Hello there." --model models/vits-piper-en_US-amy-low --play --device headphones
```

`cargo install speechkit-cli` also installs `speechkit-worker`, the crash-isolation process that `speechkit::sherpa::process::IsolatedAsr` starts; you do not run it by hand.

`mic` and `speak --play` use the system's default devices. `--device` picks another one by its name as `speechkit devices` lists it, ignoring case, or by a part of only one name. Two devices with the same name cannot be told apart. With a loopback device such as BlackHole, `mic` transcribes whatever another program plays. To record from or play to several devices at once, combine them in the operating system and select the result: on macOS, an Aggregate Device (input) or a Multi-Output Device (output) in Audio MIDI Setup; on Linux with PipeWire or PulseAudio, a combined sink (`pactl load-module module-combine-sink`). The operating system keeps their clocks in step.

Progress and partial results go to stderr; only the transcript (or, for `voices` and `devices`, the list) goes to stdout, so `speechkit transcribe a.wav > a.txt` stays clean.

Exit codes: 0 success, 2 usage error, 3 invalid input (a bad file, model, or option), 4 backend error.

`--config` for `openai-http` is a JSON file:

```json
{ "base_url": "https://api.openai.com/v1", "model": "gpt-4o-transcribe", "stream": true }
```

`stream` (default `false`) asks for server-sent events, so partial results appear as the server sends them.

For synthesis, `--config` for `openai` needs the base URL and model. `voices` (OpenAI's own by default) and `sample_rate` (the PCM rate, for servers other than OpenAI) are optional:

```json
{ "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini-tts" }
```

or, for another server:

```json
{ "base_url": "http://127.0.0.1:9000/v1", "model": "my-tts", "voices": ["alice", "bob"], "sample_rate": 22050 }
```

and for `dashscope` the model and its voices, the first being the default:

```json
{ "model": "cosyvoice-v2", "voices": ["longxiaochun_v2"] }
```
