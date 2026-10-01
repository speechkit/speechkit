Cloud speech backends: OpenAI-compatible transcription, OpenAI Realtime and speech, and DashScope recognition and synthesis.

| Feature | Contents |
|---|---|
| `openai` | `OpenAiTranscription`: any server implementing `POST /v1/audio/transcriptions`; `OpenAiRealtime`: streaming recognition over the Realtime WebSocket API; `OpenAiSpeech`: synthesis with `POST /v1/audio/speech` |
| `dashscope` | `DashScopeAsr`: Paraformer real-time recognition; `DashScopeTts`: streaming synthesis |

Each backend is configured by the `…Config` of the same name.

No feature is on by default.

Every backend takes a [`CloudRuntime`]: either a handle to your own Tokio runtime, or a runtime you create explicitly with `CloudRuntime::owned`. speechkit never starts a runtime you cannot see.

speechkit never retries. A failed request reports `SpeechError::retryable()`: true for timeouts, connection failures, and HTTP 408, 429, and 5xx.

The streaming recognizers, `OpenAiRealtime` and `DashScopeAsr`, read their connection in a task on that runtime, so results reach the session as they arrive, even while no audio is pushed, and a dropped connection fails the session at once. They report speech activity from the service: server VAD for Realtime (or a VAD of your own with `OpenAiRealtimeConfig::with_vad`), and sentence boundaries for DashScope. A service reports nothing while the user is silent, so each assumes a stated maximum lag, `with_activity_lag` (1 s by default); a service slower than that can end a turn or a session early.

OpenAI keeps a Realtime session for at most 60 minutes, so an `OpenAiRealtime` stream reconnects on its own: at the first pause after about 55 minutes, or anyway at 59. The old connection finishes its transcripts first, and times and utterance IDs go on across connections, so the session never notices; without a pause, the utterance in progress becomes two segments.
