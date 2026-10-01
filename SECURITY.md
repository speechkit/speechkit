# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose **Report a vulnerability** ([direct link](https://github.com/speechkit/speechkit/security/advisories/new)). Do not open a public issue.

Include what you found, how to reproduce it, and which versions and features are affected. A maintainer will acknowledge the report within 7 days and keep you informed until it is fixed. Fixes are released as patch versions of every affected crate, with a GitHub security advisory and a RustSec advisory, crediting you unless you ask otherwise.

## Supported versions

Before 1.0, only the latest release gets fixes. From 1.0, the latest minor release of each supported major version does.

## Scope

In scope: the speechkit crates, the `speechkit` command, and `speechkit-worker`. What each part trusts, and what it defends against, is described under [Threat model](#threat-model).

Out of scope: vulnerabilities in sherpa-onnx, ONNX Runtime, or model files themselves (report those upstream), and behavior of cloud services speechkit calls.

## Threat model

What each part of speechkit trusts, what it defends against, and what it leaves to you.

### The library

Rule IDs such as A-01, T-05, and D-02 name the contract checks in [`speechkit-testkit`](crates/speechkit-testkit); each check's doc comment states its rule.

**Audio and text from callers are untrusted.** Every sample is checked before it is queued (A-01): NaN, infinities, and out-of-range values are refused with the index of the first bad one, and the chunk is handed back. Input queues and TTS output queues are bounded. Readers read a session's history through their own cursors rather than through queues, and the history is capped at 8 MiB per session (A-11). So no input makes memory grow without limit.

**Microphone audio stays in memory, and is bounded.** A running capture holds audio only in its history (5 s by default), its wake-word reservations (30 s each), its listenings' backlogs (up to their `max_backlog`), and the 2 s before each wake-word watching. It frees that audio when it stops (A-06), and nothing is written to disk. A listening's recording, which is opt-in and capped, is the only audio handed to the caller.

**Audio files are untrusted.** `audio::read` and `audio::decode` enforce a maximum duration while decoding, for WAV and every other format. Together with the supported sample-rate range, this bounds decoded mono sample storage; it is not a separate byte limit. Both decoders are fuzzed nightly (the `decode_audio` and `read_wav_bytes` targets) with a 460 MB allocation cap.

**Model files are trusted.** sherpa-onnx and ONNX Runtime parse them in native code. Layouts are checked before the native library is called, so a missing or empty file is an error, not a crash. But a malicious or corrupt model can still crash the process, and some corrupt files make the native library abort (see `crates/speechkit/tests/sherpa_corrupt_model.rs`). Load models only from sources you trust. To survive a native crash, run the recognizer in worker processes with `IsolatedAsr` and `speechkit-worker`: a crash then fails only the session on that worker, instead of the host.

**Secrets** (API keys, the server's token) are held in `Secret`, whose `Debug` and `Display` print `Secret(***)`. Nothing is logged above `trace` that could contain audio, transcripts, synthesized text, or secrets (A-12, T-05, and the logging tests).

**Backends never retry**. A retryable error is reported as such, and the caller decides.

### The server

`speechkit::server` is meant to run behind your own network controls. It defends against misbehaving clients, not against a determined attacker with unlimited bandwidth.

| Threat | Defense |
|---|---|
| Huge uploads | The body is limited while it streams in (25 MiB by default); larger requests get 413 without being buffered. Speech input is limited to 4096 characters. |
| Slow clients holding connections (slowloris) | Headers must arrive within 30 s and the whole body within 120 s; the connection is closed otherwise. |
| Too much concurrent work | Each engine has fixed session slots (8 by default). The router also admits at most that many transcription requests before reading or decoding their uploads, rejecting excess work with 503. A detached decoder retains its admission permit until it exits. |
| Long synthesis or transcription | Request budgets include opening the session; transcription budgets start before upload and decode (120 s before the duration is known when no timeout is configured). Synthesis queue sends also expire, so unread responses cannot hold a slot forever. Disconnects cancel sessions; uninterruptible work retains its slot until it returns. |
| Malformed requests | Multipart and JSON parsing never panic; the `server_request` fuzz target sends arbitrary bodies to every endpoint nightly. Unsupported formats (WebM, Opus, MP3 output) are refused with a clear 4xx. |
| Unauthenticated use | With `Server::with_auth` (`--auth-token-env` in the CLI), every `/v1/` route requires a bearer token, compared in constant time. Without it, the server warns when bound to anything but loopback. |
| Error messages leaking internals | Errors use OpenAI's JSON shape. Backend errors name the backend and a short cause. OpenAI error bodies are reduced to their `type` and `code`; DashScope error messages are passed on, trimmed to 300 characters. |

What it does **not** do:

- **TLS.** Terminate TLS in a reverse proxy; the server speaks plain HTTP/1.1.
- **Per-client rate limiting or connection limits.** Put a proxy or load balancer in front for that. The session limit bounds CPU, not connections.
- **Multi-tenant isolation.** Every authenticated client shares the same engines and the same token.
- **HTTP/2.** Only HTTP/1.1 is served.

### Cloud backends

Requests go only to the configured endpoints, over HTTPS or WSS with certificate checks (rustls and the Mozilla root store); plain `http://` and `ws://` are accepted so tests and local servers work, and should not be used across a network. Redirects are not followed, so a key is never sent to a host you did not name. Streamed HTTP transcription (server-sent events) publishes each partial immediately and retains only the latest accumulated text, with a total wire-byte limit. Buffered response bodies are size-limited, streamed audio is consumed as it arrives, and WebSocket messages are capped at 1 MiB.

### Reviewing this model

Revisit this model when an endpoint, a parser, or a new source of untrusted input is added, and add a fuzz target for any new parser.
