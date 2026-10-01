An OpenAI-compatible HTTP server for speechkit engines (feature `server`). Point any OpenAI client's base URL at it. `Server` serves a recognition engine, a synthesis engine, or both; a route whose engine is missing answers 404, and `/health` and `/v1/models` list only what is there. With no engine at all, `/health` answers 503 and `start` refuses to serve.

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/transcriptions` | Multipart `file`, `model` (ignored), `language`, `response_format` (`json` or `text`), and `stream` (server-sent events) |
| `POST /v1/audio/speech` | JSON `input`, `voice`, `speed`, `model` (ignored), and `response_format` (`wav`, the default, or `pcm`) |
| `GET /v1/models` | The loaded models |
| `GET /health` | Status, active sessions, and the session limits |

Uploads are limited while they stream in (25 MiB by default, answered with 413). WebM and Opus are refused with 415 and a hint to transcode. When every session slot is busy the server answers 503 at once and never queues. A client that disconnects cancels its session. Errors use OpenAI's JSON error shape.

Speech is streamed as it is synthesized, always at 24 kHz, OpenAI's rate for `pcm`: `pcm` is raw 16-bit little-endian mono, and `wav` adds a streaming header whose size fields hold `0xFFFFFFFF` because the length is not known in advance. Input is limited to 4096 characters, OpenAI's documented limit; `mp3`, `opus`, `aac`, and `flac` are refused with 400. A failure after audio has started aborts the response, so a client sees a truncated body rather than a short one that looks complete.

`Server::with_auth` requires `Authorization: Bearer <token>` on `/v1/` routes.

```rust,no_run
use speechkit::server::Server;
# fn engine() -> speechkit::asr::AsrEngine { unimplemented!() }

# async fn run() -> Result<(), speechkit::SpeechError> {
Server::new()
    .with_asr(engine(), "my-model")
    // Add speech with `.with_tts(tts_engine, "my-voice")`.
    .run()
    .await?;
# Ok(())
# }
```
