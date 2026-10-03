# speechkit: the public API — design and scenario walkthrough

speechkit is a Rust library for speech recognition (ASR) and speech
synthesis (TTS). It covers:

- local models (sherpa-onnx);
- cloud services (OpenAI and DashScope);
- microphones and speakers;
- an OpenAI-compatible HTTP server.

This document designs its public API from first principles: what a caller
holds, how each piece is used, and what the library promises. Part III walks
through eleven applications using only this API, from a dictation input
method to a voice agent the user can interrupt.

---

## Part I — Principles

**Three verbs, two directions.** Everything the library does:

|              | One call                 | Session (streaming)                                | Device                                                    |
| ------------ | ------------------------ | -------------------------------------------------- | --------------------------------------------------------- |
| audio → text | `AsrEngine::transcribe`  | `start` → `push` → `updates` → `finish`            | `Microphone::listen`, `Capture::listen`, `Capture::watch` |
| text → audio | `TtsEngine::synthesize`  | `start` → `push_text` → read the output → `finish` | `Speaker::speak`, `speak_streaming`, `play`, `sink`       |

**One lifecycle.** Every interface follows:

> **start** (take a bounded resource) → **push** (input, with backpressure) →
> **read** (with a deadline) → **finish** or **cancel** → **result**
> (a failure still carries the progress made).

**Three layers.**

- A _backend_ is a loaded model or a service client.
- An _engine_ wraps a backend and limits how many sessions run on it at once.
- A _session_ is one recognition or one synthesis.

Applications hold engines and sessions. The backend traits are only for
people adding a model or a service, and the same engine type works with every
backend.

**Waiting is explicit.** A method that can wait takes a deadline, either a
`Duration` from now or an `Instant`. Where it has a `try_` twin, that twin
never waits. Only iterating a stream waits without a deadline, and every
stream ends. Nothing has a hidden timeout.

A timeout on a read or a `wait` only stops waiting. The deadline on `finish`
is different: finishing is a decision to end, so when its deadline passes the
session fails, for everyone holding it.

**Handles follow lifetimes.** A handle lives exactly as long as what it stands
for:

- an engine, as long as a loaded model;
- a `Capture`, as long as an open microphone;
- a `Listening`, as long as one request;
- a session, as long as one backend stream;
- a `Playback`, as long as one queued sound.

No handle changes type during its life. Control handles take `&self` and are
`Send + Sync`, so a key-up handler and the thread waiting for the result can
share one. Read handles are `Send` iterators with one reader each.

**A microphone is one timeline.** A capture counts _capture time_: the audio
captured since it started. It keeps the last few seconds. Every reader names
the position it starts at, and every time it reports is capture time, whether
for a segment or a wake word. So a handoff is "start at this position", for
example from key-down to session or from wake word to request. The times from
different readers line up. A wake word also reserves the audio after it the
moment it is heard, so the request can start there even if the application
is slow to get to it.

**Speech activity is data.** Backends report when speech starts and ends,
alongside partial and final text, and every reader receives every speech
event. Ending a session at a pause, the end of a speaker's turn, a "hearing
you" indicator, barge-in, and splitting at a pause are all built on those
events. Nothing guesses where speech is.

**Every stream says how it ended.** Three kinds of stream are read the same
way: recognition updates, synthesized audio, and wake words. Each is read
with `recv(deadline)`, `try_recv()`, or as an iterator. The last item is
always `Closed`, which carries the result. `RecvError` distinguishes `Empty`,
`Timeout`, and `Closed`.

**Nothing is lost silently.**

- A push that is refused always hands the chunk back, whatever the reason.
- A reader that starts at a position the capture still holds receives every
  sample from there on, in order, or fails. There is never a gap.
- A reader that falls too far behind fails with `Capacity`. It never skips
  audio.
- Readers receive every segment and every speech event. Only partials, which
  replace one another, are coalesced.
- A session that ends itself at a pause reports exactly where it stopped.
- A failure keeps what was confirmed before it.
- A stopped playback reports the text the listener heard.

**Honest about native code.** A native model call can't be interrupted. So
each session runs on its own thread and holds its slot until its backend has
stopped, including any work the backend runs in the background. Cancelling
ends the session at once, but the slot is freed only when that work has
stopped, so the limit always counts real work. A panic in a
backend fails its session, not the process. A model can also run in a worker
process, so that a native crash fails only one session.

**Policy stays with the application.** The library never:

- retries;
- falls back to another backend;
- splits a session on its own;
- guesses where speech is.

It reports exact positions, events, and whether an error is worth retrying.
That keeps an application's policy to a few lines.

**Private by default.** A session keeps no audio after processing it. Only a
listening that asks for a recording keeps any, and it hands that recording to
the caller. Audio, transcripts, and synthesized text are never logged above
`trace`. Secrets are never logged.

**A small core.** The core needs no features, no tokio, and no native library.
It contains engines, sessions, WAV files, and the VAD and wake-word traits.
Models, services, devices, and the server are features, and none of them
depends on another.

---

## Part II — The API surface

### Features

| Feature               | Adds                                                                         |
| --------------------- | ---------------------------------------------------------------------------- |
| _(none)_              | Engines and sessions, WAV files, the VAD and wake-word traits, an energy VAD |
| `decode`              | FLAC, MP3, AAC, Ogg, and Matroska files                                      |
| `sherpa`              | Local models with sherpa-onnx, linked statically                             |
| `sherpa-shared`       | sherpa-onnx linked dynamically, for CUDA or CoreML builds                    |
| `openai`, `dashscope` | Cloud services                                                               |
| `devices`             | Microphones and speakers                                                     |
| `server`              | The OpenAI-compatible HTTP server                                            |
| `serde`               | serde for the public data types                                              |

### Root

```rust
speechkit::{AudioBuffer, SampleRate, Deadline, RecvError, SpeechError, Secret, Flow}

AudioBuffer { sample_rate, samples }          // mono f32 in [-1, 1]; new(rate, samples), duration()
SampleRate                                    // 8 kHz to 192 kHz; new(hz), hz(),
                                              // HZ_16000, HZ_24000, HZ_48000, …

Deadline                                      // every waiting method takes impl Into<Deadline>
    From<Duration>                            // from now, fixed when the call starts
    From<Instant>                             // absolute, so several calls share one budget

RecvError { Empty, Timeout, Closed }          // from try_recv, from recv, and after Closed was read

SpeechError                                   // non_exhaustive; Clone
    InvalidInput(String)                      // bad options, audio, or text
    InvalidModel(String)                      // missing files, the wrong layout, or a failed load
    Unsupported(String)                       // the backend or this build can't do it
    DeadlineExceeded / Cancelled / Closed
    Capacity                                  // no free slot, a full queue, a reader too far
                                              // behind, or audio that is no longer held
    Backend { backend, retryable, source }    // source() is the backend's own error
    retryable() -> bool                       // Capacity, network errors, HTTP 408, 429,
                                              // and 5xx, and a crashed worker

Secret::new(value) / Secret::from_env(name)   // an API key or token; prints as Secret(***)
Flow { Continue, Stop }                       // what a backend's sink answers
```

### Recognition (`speechkit::asr`)

```rust
AsrEngine::new(backend)                       // any AsrBackend, or an Arc or Box of one; Clone
    .with_max_sessions(4)                     // default 8, at least 1 (0 becomes 1); gives it a pool of its own
    .with_limits(AsrLimits)                   // shares the pool
    .with_post_processor(processor)           // shares the pool; for example punctuation

engine.start(rate, AsrOptions, deadline) -> Result<AsrSession, SpeechError>
//  Waits for a free slot, then opens the backend's stream (a WebSocket
//  handshake, say), all within the deadline. Options the backend can't honor
//  fail at once with Unsupported.
engine.transcribe(&AudioBuffer, AsrOptions, deadline) -> AsrResult
//  start, push everything, and finish, all within the deadline
engine.name() / capabilities() / limits() / max_sessions() / active_sessions()
```

Clones of an engine share its backend and its pool of slots. Two engines can
share one model with separate limits: `with_max_sessions` gives an engine a
pool of its own. The other `with_` methods make a variant that shares the
pool.

```rust
AsrSession                                    // Send + Sync; dropping it cancels it
    push(chunk, deadline) -> Result<(), PushError>   // waits for room in the input queue
    try_push(chunk) -> Result<(), PushError>         // never waits
    close_input()                             // queued audio is still processed
    updates() -> AsrUpdates                   // any number of readers
    finish(deadline) -> AsrResult             // close_input, then wait for the result
    wait(deadline) -> Option<AsrResult>       // the result once it has ended by itself, or
                                              // None if it is still running; changes nothing
    cancel()
    result() -> Option<&AsrResult>
    sample_rate() / queued() -> Duration / id()

PushError { kind: PushErrorKind, chunk: Vec<f32> }   // every refusal hands the chunk back
    into_chunk() -> Vec<f32>                  // `?` turns it into a SpeechError
PushErrorKind                                 // non_exhaustive
    Full                                      // no room before the deadline
    TooLarge                                  // longer than the input queue
    Invalid { index }                         // NaN, infinite, or outside [-1, 1]
    Closed                                    // the session takes no more input
```

A chunk is mono audio at the session's rate, given as a `Vec<f32>` or a
slice (`impl Into<Vec<f32>>`). It is taken whole or handed back, whatever the
reason. So audio that a session refused because it had just closed can go to
another session. The session resamples to whatever rate the backend wants.

`finish` returns the same result on every call. If its deadline passes, the
session fails with `DeadlineExceeded` for everyone holding it, and keeps what
it confirmed. A `wait` that times out changes nothing.

```rust
AsrUpdates: Iterator<Item = AsrUpdate>        // Send; dropping it never affects the session
    recv(deadline) -> Result<AsrUpdate, RecvError>
    try_recv() -> Result<AsrUpdate, RecvError>

AsrUpdate                                     // non_exhaustive
    Partial(Partial)                          // pending text of an unfinished utterance
    Segment(Segment)                          // the final text of one utterance
    SpeechStarted { at: Duration }
    SpeechEnded { at: Duration, utterance: UtteranceId }
    //  where speech ended, sent once the backend is sure; `utterance` is the
    //  last utterance of that speech, whose segment may still be to come
    TurnEnded(Turn)                           // a turn ended (with_turn_end)
    Closed(AsrResult)                         // always last

Partial { utterance: UtteranceId, text }      // replaces the utterance's earlier partial
Segment { utterance: UtteranceId, text, start, end }
Turn { segments: Vec<Segment>, start, end }   // one turn's segments, all of them and no
    text()                                    // others; end is where its speech ended
LiveTranscript::new() / apply(&AsrUpdate) / text()   // committed text, then pending text;
                                                     // Default
```

**Readers.**

- Every reader receives every segment since the session started, and every
  speech and turn event since the reader started reading, exactly once and in
  order. A late reader first catches up on the segments. It isn't handed
  speech events from before it started, which could only be stale.
- Only partials are coalesced: a slow reader skips partials that a newer one
  replaced.
- A slow reader never slows the session. Each reader keeps its own place in
  the session's history, so there is no per-reader queue to overflow.

**History.** A session keeps its segments, and its speech and turn events,
for its readers.
`max_history_bytes` bounds them: 8 MiB, about 100 hours of speech. A session
that reaches the limit fails with `Capacity`, keeping what it confirmed.

**Times.** Times count from the session's first sample. For a session fed by
a microphone they are capture times (see "Devices, input").

```rust
AsrOptions                                    // Default; per session
    .with_language("zh")                      // needs capabilities().accepts_language
    .with_hints(["speechkit", "Zhang Wei"])   // words to favor; needs accepts_hints
    .with_turn_end(Duration)                  // report TurnEnded after this much silence
    .with_end_after_silence(Duration)         // end at the first pause this long
    .with_no_speech_timeout(Duration)         // end here if no speech has started
    .with_max_length(Duration)                // end after this much audio
```

**Utterances and turns.** One speaker's turn can hold several segments. People
pause mid-thought. And an unfinished utterance never grows without bound: a
backend cuts one that runs past its maximum and commits it as a segment (the
local models and `VadBackend` cut at 20 s by default). A cut commits a
segment, but the speech goes on. It adds a segment boundary inside the
session, never a new session.

`with_turn_end(silence)` reports where each turn ends, for a voice agent
deciding when to answer:

- A turn ends when speech has ended and the backend has confirmed `silence`
  of audio after it with no speech starting again: its report of how far
  speech activity is known has passed that point (see "Writing a backend").
  The input merely reaching that point proves nothing, because a backend
  reports the start of speech some time after it hears it. A shorter pause,
  or a cut utterance, doesn't end a turn.
- `AsrUpdate::TurnEnded` carries the `Turn`: every segment of it, including
  text the backend finishes after it reported the end of speech, and no
  segment of any other turn.
- `TurnEnded` can arrive after newer speech has started, when the turn's text
  is late. A controller that answers turns should check for that, by
  comparing the latest `SpeechStarted` with `turn.end` (scenario 11).
- Like the endings below, it is measured in audio time.

**Ending at a point in the audio.** The last three options make the session
end by itself. Two points in the audio matter:

- T, the endpoint the options ask for. It is whichever comes first: the end
  of speech plus the silence (where a turn with that silence would end), the
  no-speech timeout, or the maximum length.
- C, the cutoff where the session actually stops taking audio. For the
  maximum length, C is T. For a pause or the no-speech timeout, the backend
  must first confirm that no speech started before T, so C is how far the
  input had got when speech activity was known past T.

Then:

- The session transcribes the audio up to C and nothing after it. Audio
  already queued past C is dropped unheard, and `Closed` follows.
- The result's `duration` is the length of audio transcribed. For a
  listening, `end()` is C as a capture time. A listening that starts at 100 s
  and stops after 5 s reports a `duration` of 5 s and an `end()` of 105 s.
- A new listening can start at `end()` with nothing lost while the capture
  still holds that point: within its history, 5 s by default, or longer with
  `CaptureOptions::with_history`.
- If nobody speaks, the session ends with an empty transcript.
- With a local model, T and C depend only on the audio. The engine feeds the
  model in fixed blocks and checks after each block, so a file ends exactly
  where a live microphone would, however fast it was pushed. C trails T by
  the model's detection delay, such as the silero VAD's `min_speech`.
- With a cloud service, C trails T by as much as the service lags, which
  depends on timing. A service reports nothing while the user is silent, so
  its backend assumes a stated maximum lag (`with_activity_lag` on the
  service's config, 1 s by default). A service slower than that can end a
  turn or a session early; this is the one place the guarantee is weaker. Speech
  that resumed between T and C belongs to this session, cut at C. Audio
  pushed faster than real time makes this more likely, so a file can end at
  a different point than it would live.
- `with_turn_end`, `with_end_after_silence`, and `with_no_speech_timeout` need
  `capabilities().reports_activity`.

```rust
AsrResult = Result<Transcript, AsrFailure>
Transcript { segments: Vec<Segment>, duration }   // duration: the length of audio transcribed
    text()                                    // joined; Chinese, Japanese, Korean without spaces
AsrFailure { error: SpeechError, confirmed: Transcript }   // `?` keeps the error

AsrLimits                                     // per engine; a with_ for each
    input_queue: 2 s                          // audio waiting to be processed; the largest chunk
    max_history_bytes: 8 MiB                  // segments and events kept for readers
                                              // (the backend is fed in fixed 100 ms blocks, or the
                                              // whole queue if it is shorter: not a limit)
AsrCapabilities { sample_rate, reports_partials, reports_activity,
                  accepts_hints, accepts_language, punctuated }
```

### Synthesis (`speechkit::tts`)

```rust
TtsEngine::new(backend).with_max_sessions(n).with_limits(TtsLimits)   // as AsrEngine
engine.synthesize(text, TtsOptions, deadline) -> Result<AudioBuffer, TtsFailure>
engine.start(TtsOptions, deadline) -> Result<(TtsSession, TtsOutput), SpeechError>
//  Waits for a free slot, then opens the backend, all within the deadline.
//  Returns the text side and the audio side together.
engine.voices() / name() / capabilities() / limits() / max_sessions() / active_sessions()

TtsSession                                    // the text side; Send + Sync
    push_text(&str) -> Result<(), SpeechError>   // in pieces of any size, such as LLM tokens
    close_text()
    finish(deadline) -> TtsResult             // close_text, then wait for the end
    cancel() / result() -> Option<&TtsResult>
    sample_rate() / queued() -> Duration / id()

TtsOutput: Iterator<Item = TtsUpdate>         // the audio side; Send
    recv(deadline) / try_recv() -> Result<TtsUpdate, RecvError>
    sample_rate()

TtsUpdate                                     // non_exhaustive
    Audio(Vec<f32>)                           // mono, at sample_rate()
    Mark(Mark)                                // a piece of text is fully synthesized
    Closed(TtsResult)                         // always last

Mark { text: Range<usize>, audio: Range<Duration> }   // bytes of the pushed text, and their audio
```

Text is synthesized a sentence at a time, as each sentence is completed. So
audio for the first sentence arrives while an LLM is still writing the
second.

**The two halves.** The output is the product, not an observation, so it
works the opposite way from `AsrUpdates`:

- `start` returns both halves together, so no audio is ever made before it
  has a reader.
- There is one reader, and nothing is dropped. Synthesis waits while the
  output queue is full.
- `finish` waits for the reader to empty the queue. When its deadline passes,
  the synthesis fails.
- Dropping the session before `close_text()` cancels the synthesis. After
  `close_text()`, dropping it changes nothing, and the text is synthesized to
  the end.
- Dropping the output cancels the synthesis, since nobody can hear it any
  more.

```rust
TtsResult = Result<TtsSummary, TtsFailure>
TtsSummary { duration, marks: Vec<Mark> }
TtsFailure { error: SpeechError, text_done: usize, duration }   // bytes fully spoken, audio made

TtsOptions                                    // Default; checked before synthesis starts
    .with_voice(id)                           // one of engine.voices()
    .with_language(code)
    .with_speed(1.2)                          // within capabilities().speed
    .with_sample_rate(rate)                   // resample the output; default: the backend's rate
Voice { id, name, languages }
TtsLimits { output_queue: 2 s, max_text_chars: 10 000, chunk_chars: 300 }
TtsCapabilities { sample_rate, streams_audio, speed: Option<RangeInclusive<f32>>, max_chunk_chars }
```

### Audio files (`speechkit::audio`)

```rust
audio::read(path, DecodeLimits) -> Result<AudioBuffer, SpeechError>
audio::decode(bytes, DecodeLimits) -> Result<AudioBuffer, SpeechError>
//  Mono, at the file's rate. WAV always works; FLAC, MP3, AAC, Ogg, and
//  Matroska need the `decode` feature. The format comes from the content,
//  not the file name.
audio::encode_wav(&AudioBuffer) -> Result<Vec<u8>, SpeechError>   // 16-bit PCM
audio::EXTENSIONS                             // for a file dialog's filter
DecodeLimits::new(max_duration)               // longer audio is refused while decoding;
                                              // default 20 min
```

### Devices, input (`speechkit::io`)

```rust
Microphone::open_default() / open(name) / list() -> Vec<DeviceInfo>
    name() / sample_rate()
//  A name matches exactly, then ignoring case, then as part of a name, and
//  must match exactly one device.
DeviceInfo { name, is_default }               // also what Speaker::list returns

mic.capture(CaptureOptions) -> Result<Capture, SpeechError>
mic.listen(&engine, AsrOptions) -> Result<Listening, SpeechError>
//  The short way: capture, then listen from the start. The listening is the
//  capture's only user, so stopping it stops the microphone.

CaptureOptions::default().with_history(Duration)   // how far back a reader may start;
                                                   // default 5 s, about 1 MB at 48 kHz

Capture                                       // a running microphone; Clone + Send + Sync
    sample_rate()
    position() -> Duration                    // capture time: the audio captured so far
    level() -> f32                            // RMS of the latest audio, for a meter
    device_lost() -> bool
    listen(&engine, AsrOptions, ListenOptions) -> Result<Listening, SpeechError>
    watch(&wake_word_model) -> Result<Watching, SpeechError>
    watch_with(&wake_word_model, WatchOptions)   // WatchOptions::with_max_backlog (30 s): how far
                                                 // the detector may fall behind (D-02)
    stop()                                    // stops the microphone for every reader
```

The microphone runs while something needs it: a `Capture` handle, or a
listening or watching that hasn't stopped. The application decides how long
that is:

- per key press, so the microphone indicator shows only while dictating;
- or for as long as a text field has focus, or an assistant is armed.

```rust
ListenOptions
    ::default()                               // start now
    ::starting_at(Duration)                   // start at a capture time, up to `history` back;
                                              // an older one fails at once with Capacity
    .with_recording(Duration)                 // keep up to this much audio, for a retry
    .with_max_backlog(Duration)               // audio held while the session opens or falls
                                              // behind; past it, Capacity. Default 30 s

Listening                                     // one request; Send + Sync; dropping it cancels it
    updates() -> AsrUpdates
    stop()                                    // end the input here; returns at once
    finish(deadline) -> AsrResult             // stop, then wait for the result
    wait(deadline) -> Option<AsrResult>       // the result once it has ended by itself
    cancel()                                  // the recording is kept
    result() -> Option<&AsrResult>
    origin() -> Duration                      // capture time of its first sample, known at the start
    end() -> Option<Duration>                 // capture time of its last sample, once it has ended
    recording() -> Option<Recording>          // complete once it has ended
    level() / device_lost()

Recording { audio: AudioBuffer, truncated: bool }   // truncated: it reached its limit,
                                                    // and later audio is not in it
```

**Starting.** `listen` checks the options at once and returns. The session
opens in the background (waiting for a slot, loading, or connecting) while
the listening holds the audio. Then:

- The session receives the held audio first, and live audio after it.
- Its times are capture times, so they line up with wake words and with every
  other listening on the same capture.
- If the session can't open, that failure becomes the listening's result.

**Ending.** A listening ends in any of these cases, and its session then
finishes with what it heard:

- you call `stop`, `finish`, or `cancel`;
- its session ends by itself (at a pause, with no speech, or at its maximum
  length), and `end()` is then the cutoff C, as a capture time;
- the device is lost.

A session that fails doesn't end the listening. The recording continues until
you stop, so a retry gets the whole utterance, up to the recording's limit.

```rust
Watching: Iterator<Item = WakeUpdate>         // Send; runs until stopped or dropped
    recv(deadline) / try_recv() -> Result<WakeUpdate, RecvError>
    stop()

WakeUpdate                                    // non_exhaustive
    Heard(Wake)                               // a keyword, with the audio after it reserved
    Closed(Result<(), SpeechError>)           // Err if the detector failed or fell behind

Wake                                          // Send
    event() -> &WakeEvent                     // WakeEvent { keyword, start, end }, in capture time
    listen(self, &engine, AsrOptions) -> Result<Listening, SpeechError>
    //  a listening that starts right after the keyword, from the reserved
    //  audio. Dropping the Wake releases the reservation.
```

A watching runs its detector on its own thread. It keeps running while the
application listens and speaks, so a wake word can interrupt a reply.

When the watching hears a keyword, it reserves the audio from the keyword's
end at once, before the application has read the event. A reservation holds
up to 30 s. `wake.listen` starts from it, so the request starts right after
the keyword even if the application took a while to get to the event. After
30 s the reservation is gone, and `listen` fails with `Capacity`.

**The device invariant.** Every reader of a capture reads one timeline.

- A reader that starts at a position the capture still holds, in its history
  or in a reservation, receives every sample from there on, in order.
- Otherwise it fails with `Capacity`: at once if the position is no longer
  held, or later if it falls too far behind.
- Readers may overlap, and each gets the same samples.

The same rule covers every handoff:

- key-down to session;
- a session that connects after key-up;
- wake word to request.

None of these depends on who read a buffer first.

The device callback only converts samples, writes them to a lock-free ring
buffer, and stores the level. The capture's own thread moves the samples into
the history, into reservations, and on to each reader.

### Devices, output

```rust
Speaker::open_default() / open(name) / list() -> Vec<DeviceInfo>
    name() / sample_rate()
//  The output stream starts with the first sound and runs until the speaker
//  is dropped. Sounds play one after another in the order they were queued,
//  resampled to the device's rate.

speaker.speak(&engine, text, TtsOptions) -> Result<Playback, SpeechError>
//  checks the options, synthesizes in the background, and plays as audio arrives
speaker.speak_streaming(&engine, TtsOptions) -> Result<(TtsSession, Playback), SpeechError>
//  The same, for text that is still being written: push it into the session
//  as it arrives. Returns at once; text pushed before the synthesis has
//  opened waits for it.
speaker.play(sound) -> Result<Playback, SpeechError>          // impl Into<Sound>: a TtsOutput
                                                              // or an AudioBuffer
speaker.sink(rate) -> Result<(Sink, Playback), SpeechError>   // samples you push yourself
speaker.stop()                                // stop what is playing and everything queued
speaker.device_lost() / underruns()

Playback                                      // one queued sound; Send + Sync;
                                              // dropping it lets it play
    stop()                                    // skip it; a synthesis it plays is cancelled
    finish(deadline) -> Result<(), SpeechError>
    //  Wait until it has played to its end or was stopped. Fails on a lost
    //  device, on a failed synthesis (after playing what arrived), or at the
    //  deadline.
    is_done() -> bool
    played() -> Duration                      // audio the device has actually played
    text_played() -> String                   // the text whose audio has fully played:
                                              // what the listener heard

Sink                                          // Send + Sync
    push(&[f32], deadline) -> Result<(), SpeechError>   // waits while the queue is full
    close()                                   // its playback ends after what was pushed
```

`text_played` is what a voice agent needs when the user interrupts. It is the
part of the reply that was actually heard, to the nearest sentence, and the
application doesn't have to keep a copy of the text to get it. The rest can
be dropped from the conversation. Stopping is not an error: `finish` after
`stop` returns `Ok`.

### Local models (`speechkit::sherpa`)

```rust
sherpa::AsrConfig::streaming(dir)             // a streaming transducer: partials, low latency
sherpa::AsrConfig::offline(dir, vad_file)     // SenseVoice, Paraformer, FireRedASR, Qwen3-ASR,
                                              // FunASR-Nano, or an offline transducer, behind
                                              // the silero VAD
    .with_language(code) / .with_hotwords([...]) / .with_family(f) / .with_inference(i)
    .with_endpoint_silence(Duration)          // streaming: the pause that ends an utterance;
                                              // default 1 s
    .with_max_utterance(Duration)             // cut an utterance that runs this long; default 20 s
    .validate() -> Result<AsrFamily, SpeechError>   // checks the files; no native code runs
    .load() -> Result<sherpa::Asr, SpeechError>     // an AsrBackend
sherpa::TtsConfig::new(dir).load()            // VITS or Piper, Matcha, Kokoro: a TtsBackend
sherpa::KeywordSpotterConfig::new(dir).with_keywords(["Hi Jarvis", "小爱同学"]).load()
                                              // a WakeWordModel; it spells the phrases itself
sherpa::PunctuationConfig::new(dir).load()    // a PostProcessor
sherpa::SileroVadConfig::new(file).load()     // a VadModel
// Every config has validate(), which checks it without loading native code.

sherpa::inspect(dir) -> Result<ModelInfo, SpeechError>   // for a model picker; no native code
ModelInfo { asr: Vec<AsrFamily>, tts: Option<TtsFamily>,
            punctuation: Option<PunctuationFamily>, keyword_spotter: bool }
sherpa::Inference::default().with_provider(Provider::Cuda).with_threads(4)
sherpa::Provider::{Cpu, Cuda, CoreMl}.is_supported()

sherpa::process::IsolatedAsr::spawn(worker, &AsrConfig)
//  the model in a child process: a native crash fails one session, not the app
sherpa::process::worker_main()                // the whole worker binary
```

Where speech activity comes from:

| Backend                                                    | Speech activity from                                                                                                                                                  |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sherpa::Asr`, streaming                                   | Its endpoint detector. Speech starts at an utterance's first decoded token and ends at its last token, reported once the pause reaches `with_endpoint_silence`.       |
| `sherpa::Asr`, offline                                     | Its silero VAD, which reports a pause after `min_silence` (500 ms by default).                                                                                        |
| `VadBackend`, including `OpenAiTranscription` behind a VAD | Its `Vad`.                                                                                                                                                            |
| `OpenAiRealtime`                                           | The service's voice detection (`input_audio_buffer.speech_started` and `speech_stopped`). This is on by default.                                                      |
| `DashScopeAsr`                                             | The service's sentence boundaries.                                                                                                                                    |
| `sherpa::process::IsolatedAsr`                             | The model in the worker.                                                                                                                                              |

A backend that can't report activity says so in its capabilities. Then
`with_turn_end`, `with_end_after_silence`, and `with_no_speech_timeout` fail
at `start` with `Unsupported`.

### Cloud services (`speechkit::cloud`)

```rust
CloudRuntime::owned(threads)? / CloudRuntime::from_handle(tokio_handle)
//  Clone. The Tokio runtime the backends run on: never a hidden one.

OpenAiTranscription::new(config, runtime)     // file transcription over HTTP: an AsrBackend,
                                              // and an OfflineRecognizer behind a VAD. Works
                                              // with any OpenAI-compatible server.
OpenAiRealtime::new(config, runtime)          // streaming; the service reports speech activity
OpenAiSpeech::new(config, runtime)            // a TtsBackend
DashScopeAsr::new(config, runtime) / DashScopeTts::new(config, runtime)
// Each config's new() takes what the service requires: a model, and an API
// key (a Secret) where the service always needs one. A with_ covers the
// rest: the endpoint, a key for servers that take one, a timeout for each
// network call, and, for the streaming services, with_activity_lag: how far
// behind the audio the service may report speech (default 1 s).
// DashScopeTtsConfig::new also takes the voice list, a Vec<Voice>: the
// service cannot list its voices and no default list fits every model, so it
// is required, and DashScopeTts::new rejects an empty list.
```

### Server (`speechkit::server`)

```rust
Server::new()
    .with_asr(engine, "sense-voice")          // POST /v1/audio/transcriptions
    .with_tts(tts, "kokoro")                  // POST /v1/audio/speech
    .with_bind("0.0.0.0:8080")                // default 127.0.0.1:8080
    .with_auth(secret)                        // a bearer token on /v1/ routes
    .with_max_body_bytes(n) / .with_timeout(d)
    .run().await?                             // until Ctrl+C or SIGTERM
// or .start(stop_future).await? for a RunningServer, or .router() to nest in axum
```

Behavior:

- A route without an engine answers 404.
- `/health` lists what the server serves.
- A busy server answers 503 at once rather than queueing the request.

### Writing a backend

```rust
trait AsrBackend: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &AsrCapabilities;
    fn open(&self, options: &AsrOptions, events: AsrEvents)
        -> Result<Box<dyn AsrStream>, SpeechError>;
}
trait AsrStream: Send {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError>;
    fn finish(&mut self) -> Result<(), SpeechError>;   // input ended: returns once every
                                                       // final event has been sent
    fn cancel(&mut self) {}
}
AsrEvents                                     // Clone + Send: where a stream's results go
    send(AsrEvent) -> Flow                    // from any thread, at any time; Stop once the
                                              // session has ended
    fail(SpeechError)                         // the stream broke between calls, such as a
                                              // dropped connection
enum AsrEvent {                               // exhaustive: a new kind must break every backend
    Partial(Partial), Segment(Segment),
    SpeechStarted { at: Duration },
    SpeechEnded { at: Duration, utterance: UtteranceId },   // the last utterance of that speech
    ActivityKnown { through: Duration },      // every start and end of speech before
                                              // `through` has been sent
}
trait PostProcessor: Send + Sync + 'static {
    fn process(&self, text: &str) -> Result<String, SpeechError>;
}

trait TtsBackend: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &TtsCapabilities;
    fn voices(&self) -> &[Voice];
    fn open(&self, options: &TtsOptions) -> Result<Box<dyn TtsStream>, SpeechError>;
}
trait TtsStream: Send {
    fn synthesize(&mut self, text: &str, audio: &mut dyn FnMut(&[f32]) -> Flow)
        -> Result<(), SpeechError>;
    fn cancel(&mut self) {}
}

trait VadModel: Send + Sync + 'static {        // creates one Vad per session
    fn sample_rate(&self) -> SampleRate;
    fn start_delay(&self) -> Duration;         // how late the detector reports the start of speech:
                                               // the engine knows activity only up to the audio fed
                                               // minus this, so an underestimate can end a turn
                                               // in the middle of speech. Required, with no default.
    fn create(&self, max_speech: Duration) -> Result<Box<dyn Vad>, SpeechError>;
}
trait Vad: Send {
    fn accept(&mut self, samples: &[f32]) -> Vec<SpeechSegment>;   // none longer than max_speech
    fn flush(&mut self) -> Vec<SpeechSegment>;
    fn speaking_since(&self) -> Option<Duration>;   // the start of the speech in progress
}
trait OfflineRecognizer: Send + Sync + 'static {   // one utterance at a time
    fn name(&self) -> &str;
    fn capabilities(&self) -> &AsrCapabilities;
    fn recognize(&self, samples: &[f32], options: &AsrOptions) -> Result<String, SpeechError>;
}
VadBackend::new(recognizer, vad_model)?       // an AsrBackend (Err if the VAD's rate is not the
                                              // recognizer's): the VAD cuts utterances,
    .with_max_utterance(Duration)             // and the recognizer transcribes each one;
                                              // default 20 s
EnergyVad::new(EnergyVadConfig)               // a VadModel with no model file

trait WakeWordModel: Send + Sync + 'static {   // creates one detector per capture reader
    fn sample_rate(&self) -> SampleRate;
    fn create(&self) -> Result<Box<dyn WakeWordDetector>, SpeechError>;
}
trait WakeWordDetector: Send {
    fn accept(&mut self, samples: &[f32]) -> Vec<WakeEvent>;
    fn flush(&mut self) -> Vec<WakeEvent>;
}
```

**What the engine does.** Queueing, deadlines, cancellation, readers and
their history, resampling, endpointing, turn ends, and checking options
against capabilities before `open`. Events a stream sends reach readers at
once, whether or not audio is being pushed.

**What the engine promises a stream.**

- It calls the stream from one thread.
- It never calls `accept` after `finish`.
- It passes samples already at the backend's rate.

**What a stream must do.**

- Send results through `AsrEvents`, either from `accept` or from a thread of
  its own. A cloud stream sends what the service returns as it arrives, so
  nothing waits for the next chunk of audio.
- End every utterance with exactly one `Segment`, even one with no words;
  readers never see the empty ones. Name the last utterance of a stretch of
  speech in its `SpeechEnded`, which may come before that utterance's
  segment. Together, these tell the engine when a turn's text is complete.
- Send `ActivityKnown { through }` as detection progresses. The engine ends a
  turn, or a session at a pause or the no-speech timeout, only once activity
  is known past the point it needs, so a start of speech reported late can't
  be missed. A local model sends it after each block, trailing the input by
  its detection delay. A cloud stream sends it from the service's own
  reports, and while the service is silent, from the audio it has sent minus
  the config's `activity_lag`.
- Bound an unfinished utterance: cut it at its maximum and send it as a
  segment, without `SpeechEnded`, since the speech goes on.
- Return from `finish` only after sending every final event.
- Report a failure between calls, such as a dropped connection, with `fail`,
  so the session fails at once.
- Stop as soon as it can once `send` answers `Flow::Stop`.
- Stop its own threads and tasks, and close its connections, before its
  destructor returns. The engine keeps the session's slot until the stream
  is dropped, so the limit counts work still running in the background.
- If the service limits how long one connection lasts, reconnect inside the
  stream, at the first pause once most of the allowance is used. If speech
  runs on to the hard limit, reconnect anyway and send the utterance in
  progress as two segments. Either way, the session never notices.

### Guarantees

Each guarantee is a test of the same name. The tests run against a fake
backend and against every real one.

| ID   | Guarantee                                                                                                                                                                                                                                                                                                                                                                                             |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A-01 | A refused chunk comes back unchanged, whatever the reason.                                                                                                                                                                                                                                                                                                                                            |
| A-02 | A full input queue refuses `try_push` at once. `push` waits until there is room, the deadline passes, or the session ends.                                                                                                                                                                                                                                                                            |
| A-03 | `finish` returns the same result on every call. `wait` and `result()` return it once the session has ended. A `recv` or `wait` that times out changes nothing.                                                                                                                                                                                                                                        |
| A-04 | A failure keeps the confirmed transcript. An error is never reported as an empty success.                                                                                                                                                                                                                                                                                                             |
| A-05 | Every reader receives every segment since the session started and every speech and turn event since it started reading, exactly once and in order. Only partials are coalesced. A slow reader never slows the session.                                                                                                                                                                                |
| A-06 | A session keeps no audio after processing it.                                                                                                                                                                                                                                                                                                                                                         |
| A-07 | Endpointing stops a session at a cutoff C, no earlier than the endpoint T. At a pause or the no-speech timeout, C comes only once the backend has confirmed activity past T, so speech that restarted before T and was reported late never ends a session. Nothing after C is transcribed, and the result reports it. With a local model, the same audio stops at the same C whatever the push speed. |
| A-08 | A turn ends only once the backend has confirmed its silence: a shorter pause, including one whose restart is reported late, or a cut utterance, never ends it. `TurnEnded` carries every segment of its turn and none of another, even when its text arrives after newer speech started.                                                                                                              |
| A-09 | A backend panic fails its session, never the process. A slot is freed only once the backend's stream has been dropped, which stops any work it runs in the background. If the session ended by itself, the slot is freed before `finish` or `wait` returns.                                                                                                                                           |
| A-10 | Events a backend sends reach readers, and a stream failure ends the session, even while no audio is being pushed. `finish` returns only after every final event has been sent.                                                                                                                                                                                                                        |
| A-11 | An unfinished utterance is cut at its maximum and committed. The history never exceeds `max_history_bytes`: a session that reaches it fails with `Capacity`, keeping what it confirmed.                                                                                                                                                                                                               |
| A-12 | Transcripts and audio are never logged above `trace`.                                                                                                                                                                                                                                                                                                                                                 |
| T-01 | Audio is mono at the session's rate, in text order. Nothing is dropped, and synthesis waits while the output queue is full.                                                                                                                                                                                                                                                                           |
| T-02 | Every sample has a reader: `start` returns the output with the session. Dropping the session after `close_text` changes nothing. Dropping it before, or dropping the output, cancels the synthesis.                                                                                                                                                                                                   |
| T-03 | A `Mark` follows the last audio of its text. `Closed` is always last.                                                                                                                                                                                                                                                                                                                                 |
| T-04 | Bad options, unknown voices, and empty text are rejected before synthesis starts.                                                                                                                                                                                                                                                                                                                     |
| T-05 | Synthesized text is never logged above `trace`.                                                                                                                                                                                                                                                                                                                                                       |
| D-01 | A reader that starts at a held position receives every sample from there on, in order. A position that is no longer held fails at once with `Capacity`.                                                                                                                                                                                                                                               |
| D-02 | A reader further behind than its limit fails with `Capacity`. Audio is never skipped.                                                                                                                                                                                                                                                                                                                 |
| D-03 | A wake word, once heard, reserves the audio after it for 30 s, whether or not the application has read the event yet.                                                                                                                                                                                                                                                                                 |
| D-04 | `stop()` returns at once on every handle. `finish(deadline)` bounds the rest.                                                                                                                                                                                                                                                                                                                         |
| D-05 | A lost device ends every reader at its last sample and fails every queued playback.                                                                                                                                                                                                                                                                                                                   |
| D-06 | `played()` and `text_played()` never exceed what the device has played.                                                                                                                                                                                                                                                                                                                               |

The device tests run against a fake device. Scenarios 5d, 7, and 11 also run
as tests, against fake backends, a fake device, and a stand-in LLM. They cover
the cases where these contracts are easiest to get wrong:

- synthesis faster than real time, before the reader starts reading (T-02);
- an LLM that stalls while the user interrupts, where the loop must still
  handle the next update at once;
- a cloud event that arrives while no audio is being pushed (A-10);
- the same audio pushed at different speeds, and a report of a pause that
  arrives after the input has passed T (A-07);
- speech that ends at 1.0 s and restarts at 1.6 s, with a 700 ms threshold,
  where the restart is reported only after the input passes 1.7 s: neither
  the turn nor a session at a pause may end (A-07, A-08);
- `SpeechEnded(A)`, 700 ms of silence, `SpeechStarted(B)`, `Partial(B)`,
  then A's late segment: `TurnEnded(A)` holds only A's text, and scenario 11
  waits for B before answering (A-08);
- a pause mid-sentence and an utterance cut at its maximum, neither of which
  may end a turn (A-08).

---

## Part III — Scenario walkthroughs

Imports are left out. Each scenario runs in a function that returns
`Result<(), Box<dyn Error>>`, or in an `async fn` for the two servers. Names
such as `ui`, `worker`, `brain`, and `llm` are application code.

### 1. Dictation input method (voice IME)

Push-to-talk. Key-up never blocks the UI thread:

```rust
let listening = mic.listen(&engine, AsrOptions::default().with_hints(ui.contact_names()))?;
let mut updates = listening.updates();
let mut live = LiveTranscript::new();
while !ui.key_is_up() {
    while let Ok(update) = updates.try_recv() {
        live.apply(&update);
    }
    ui.show_candidates(&live.text());               // committed + pending words
    ui.set_meter(listening.level());
}
listening.stop();                                   // in a key-up handler: returns at once
let transcript = listening.finish(Duration::from_secs(5))?;   // the tail, bounded
ui.insert(&transcript.text());
```

Toggle-to-talk. The first press starts listening; the second stops it. Keep
an `Option<(u64, Listening)>` named `active` in the UI state, initially `None`.
The number identifies a round and is never reused. Both key-down and a
self-ended session use this helper:

```rust
fn finish_dictation(active: &mut Option<(u64, Listening)>, worker: &Worker) {
    if let Some((id, listening)) = active.take() {
        listening.stop();                           // returns at once
        let deadline = Deadline::from(Duration::from_secs(5));
        worker.submit(id, move || listening.finish(deadline));
    }
}
```

Run this handler on key-down, ignoring auto-repeat; key-up does nothing:

```rust
if active.is_some() {
    finish_dictation(&mut active, &worker);
} else {
    let listening = mic.listen(&engine, AsrOptions::default().with_hints(ui.contact_names()))?;
    let id = ui.begin_dictation();
    ui.forward_updates(id, listening.updates());
    active = Some((id, listening));
}
```

`ui.begin_dictation` allocates the ID, remembers the insertion target, and
reserves its place in the result order. Call it only after `mic.listen`
succeeds, so a failed start leaves `active` empty without reserving a
result. `ui.forward_updates` reads updates off the UI thread and posts
`(id, update)` to it. On the UI thread, handle a `Closed` update like this:

```rust
if matches!(update, AsrUpdate::Closed(_))
    && active.as_ref().is_some_and(|(current, _)| *current == id)
{
    finish_dictation(&mut active, &worker);
}
```

Taking `active` schedules a round's finish only once. A late `Closed`
from a stopped round cannot clear a newer round. Partial text and meters
should also update only the matching active round. Do not insert text
from `Closed`: the worker is the sole source of completion results.

`worker.submit` runs the closure off the UI thread and posts `(id, result)`
back to it. It must accept and promptly dispatch each submitted task; the
five-second deadline starts before queueing. The result handler is:

```rust
ui.complete_dictation(id, result);
```

`ui.complete_dictation` accepts each pending ID once, holds results that
arrive out of order, and delivers them in start order to their remembered
targets, inserting the transcript or showing the failure. A failure also
releases its place in the order. This handler never changes `active`.

A new round may start while an older one finishes. Each gets a fresh
listening; unfinished backend work still holds an engine slot. When all
slots are busy, `mic.listen` waits for one in the background while holding
audio, up to the listening's backlog limit. Handle any eventual failure
through the same result path.

Hands-free. There is no key; a pause ends the dictation:

```rust
let options = AsrOptions::default()
    .with_end_after_silence(Duration::from_millis(800))
    .with_no_speech_timeout(Duration::from_secs(5));
let listening = mic.listen(&engine, options)?;
let mut live = LiveTranscript::new();
for update in listening.updates() {                 // Closed arrives by itself, at the pause
    live.apply(&update);
    ui.show_candidates(&live.text());
}
let transcript = listening.finish(Duration::from_secs(1))?;   // already ended: returns at once
ui.insert(&transcript.text());                      // empty if nobody spoke
```

A cloud backend. This one shows three things:

- A word spoken while the key goes down is kept.
- The session may connect after the key is up.
- A failure is retried on another engine with the same audio.

```rust
// While the text field has focus, the microphone stays open with 5 s of history.
let capture = mic.capture(CaptureOptions::default())?;

// Key down: listen from 300 ms back, and record, so a failure can be retried.
// The session connects in the background.
let from = capture.position().saturating_sub(Duration::from_millis(300));
let listen = ListenOptions::starting_at(from).with_recording(Duration::from_secs(60));
let listening = capture.listen(&cloud, AsrOptions::default(), listen)?;

// Key up: returns at once, even if the session hasn't connected yet.
ui.wait_for_key_up();
listening.stop();

// In an app, the rest runs on a worker thread.
let text = match listening.finish(Duration::from_secs(10)) {
    Ok(transcript) => transcript.text(),
    Err(failure) if failure.error.retryable() => {
        let recording = listening.recording().ok_or("no recording")?;
        backup.transcribe(&recording.audio, AsrOptions::default(), Duration::from_secs(30))?.text()
    }
    Err(failure) => return Err(failure.into()),
};
ui.insert(&text);
```

No thread, channel, or `Arc` is needed. The listening holds the 300 ms before
the key and everything spoken during the handshake. The session receives all
of it once it connects. A dictation longer than the recording's limit comes
back with `truncated` set, and the retry then covers only the first minute.

### 2. Meeting transcription

Hours long, hands-free, with timestamped minutes. A session keeps no audio,
so one session can run all day:

```rust
let asr = sherpa::AsrConfig::offline("sense-voice", "silero_vad.onnx").load()?;
let punctuation = sherpa::PunctuationConfig::new("punct-ct-transformer").load()?;
let engine = AsrEngine::new(asr).with_post_processor(punctuation);

let listening = mic.listen(&engine, AsrOptions::default().with_language("zh"))?;
let mut live = LiveTranscript::new();
for update in listening.updates() {
    live.apply(&update);
    match &update {
        AsrUpdate::Segment(segment) => {
            writeln!(minutes, "[{}] {}", clock(segment.start), segment.text)?;   // capture time
        }
        AsrUpdate::Closed(Err(failure)) => {
            save(&failure.confirmed);               // what was confirmed is kept
            notify(&failure.error);
        }
        _ => panel.show(&live.text()),
    }
    if listening.device_lost() {
        prompt_replug();                            // the session ends with what it heard
    }
}
```

To write one file per hour, route each segment by its capture time; there's
no need to end the session for that. The session's history is capped at about
100 hours of speech. A recording that could run longer should start a new
listening at a pause, with `ListenOptions::starting_at`, while the capture
still holds that point, and nothing is lost between the two.

### 3. Transcribe a file

```rust
let engine = AsrEngine::new(sherpa::AsrConfig::offline("sense-voice", "silero_vad.onnx").load()?);
let recording = audio::read("meeting.mp3", DecodeLimits::new(Duration::from_secs(7200)))?;
let transcript = engine.transcribe(&recording, AsrOptions::default(), Duration::from_secs(600))?;
for segment in &transcript.segments {
    write_srt(segment.start, segment.end, &segment.text)?;   // times from the file's start
}
```

To use a cloud service instead, change only the first line:
`AsrEngine::new(OpenAiTranscription::new(config, runtime)?)`.

### 4. ASR service (OpenAI-compatible HTTP)

```rust
Server::new()
    .with_asr(engine, "sense-voice")
    .with_bind("0.0.0.0:8080")
    .with_auth(Secret::from_env("SPEECHKIT_TOKEN")?)
    .run()
    .await?;
```

```sh
curl -H "Authorization: Bearer $SPEECHKIT_TOKEN" \
     -F file=@meeting.wav http://127.0.0.1:8080/v1/audio/transcriptions
```

### 5. Text to speech

(a) One call, to a file:

```rust
let options = TtsOptions::default().with_voice("45").with_speed(1.1);
let speech = tts.synthesize("Hello from speechkit.", options, Duration::from_secs(60))?;
std::fs::write("out.wav", audio::encode_wav(&speech)?)?;
```

(b) Speak it, and stop it partway through:

```rust
let playback = speaker.speak(&tts, "Done.", TtsOptions::default())?;
if ui.skip_pressed() {
    playback.stop();                                // stops synthesizing too
}
playback.finish(Duration::from_secs(30))?;          // Ok after a stop
```

(c) Speak an LLM's reply while it is still being written:

```rust
let (session, playback) = speaker.speak_streaming(&tts, TtsOptions::default())?;   // returns at once
for piece in llm.stream("Tell me a story.") {
    session.push_text(&piece)?;
}
session.close_text();
playback.finish(Duration::from_secs(300))?;
```

(d) Your own processing, through a sink:

```rust
let (session, output) = tts.start(TtsOptions::default(), Duration::from_secs(5))?;
session.push_text("Processed on its way to the speaker.")?;
session.close_text();
let (sink, playback) = speaker.sink(output.sample_rate())?;
for update in output {
    match update {
        TtsUpdate::Audio(samples) => sink.push(&with_gain(&samples, 0.8), Duration::from_secs(2))?,
        TtsUpdate::Closed(result) => {
            result?;
        }
        _ => {}
    }
}
sink.close();
playback.finish(Duration::from_secs(10))?;
```

The output exists from the start, so audio made before the loop begins
reading waits in the output queue.

(e) A sound file uses the same handle:

```rust
speaker
    .play(audio::read("chime.wav", DecodeLimits::default())?)?
    .finish(Duration::from_secs(5))?;
```

### 6. TTS service (OpenAI-compatible HTTP)

```rust
Server::new()
    .with_tts(tts, "kokoro")
    .with_bind("0.0.0.0:8080")
    .with_auth(Secret::from_env("SPEECHKIT_TOKEN")?)
    .run()
    .await?;
```

```sh
curl -H "Authorization: Bearer $SPEECHKIT_TOKEN" -H "Content-Type: application/json" \
     -d '{"model":"kokoro","input":"Hello."}' \
     http://127.0.0.1:8080/v1/audio/speech -o speech.wav
```

### 7. Voice assistant (wake word → one request → reply)

This scenario shows three things:

- The request starts right after the keyword, even though the detector
  reports the keyword later: the audio after it is reserved as soon as it is
  heard.
- The wake word can interrupt a reply.
- A keyword said inside a request isn't taken as a new one.

```rust
let capture = mic.capture(CaptureOptions::default())?;   // stays open, with 5 s of history
let speaker = Speaker::open_default()?;
let request = AsrOptions::default()
    .with_end_after_silence(Duration::from_secs(1))
    .with_no_speech_timeout(Duration::from_secs(5))
    .with_max_length(Duration::from_secs(20));
let mut handled_until = Duration::ZERO;
for update in capture.watch(&kws)? {                // keeps running while it listens and speaks
    let wake = match update {
        WakeUpdate::Heard(wake) => wake,
        WakeUpdate::Closed(result) => return Ok(result?),
        _ => continue,
    };
    let keyword_end = wake.event().end;
    if keyword_end < handled_until {
        continue;                                   // said inside a request already handled
    }
    speaker.stop();                                 // barge-in: the wake word stops a reply
    let listening = match wake.listen(&asr, request.clone()) {
        Ok(listening) => listening,                 // starts right after the keyword
        Err(SpeechError::Capacity) => continue,     // reserved over 30 s ago: skip it
        Err(error) => return Err(error.into()),
    };
    let heard = listening
        .wait(Duration::from_secs(30))              // ends by itself at a pause
        .unwrap_or_else(|| listening.finish(Duration::from_secs(5)))?;   // or cut it short
    handled_until = listening.end().unwrap_or(keyword_end);
    if heard.segments.is_empty() {
        continue;                                   // woken, but nothing was said
    }
    let reply = brain.answer(&heard.text());
    speaker.speak(&tts, &reply, TtsOptions::default())?;
    // Not waited for: back to the watcher, which can interrupt the reply.
}
```

The reservation is what makes the handoff exact. `wake.listen` starts from
audio reserved when the keyword was heard, however long the loop took to get
to it. The limit is 30 s. After that the audio is gone, `listen` fails with
`Capacity`, and the loop skips the event. An assistant that may think for
longer than that should answer on another thread, so the loop stays
responsive.

### 8. Live captions (your own audio source, no device)

```rust
let session = asr.start(SampleRate::HZ_48000, AsrOptions::default(), Duration::from_secs(5))?;
let mut updates = session.updates();
let mut live = LiveTranscript::new();
for packet in rtp.audio() {
    session.push(packet, Duration::from_secs(2))?;  // waits while the queue is full
    while let Ok(update) = updates.try_recv() {
        live.apply(&update);
    }
    overlay.show(&live.text());
}
session.finish(Duration::from_secs(10))?;           // closes the input and drains it
```

A session keeps no audio, so captions can run for days, up to the history cap
of about 100 hours of speech. A stream that runs longer starts a new session.

### 9. Batch transcription pipeline

Engine slots are the throttle. Decoding needs no slot, so running more
workers than slots keeps every slot busy:

```rust
let engine = AsrEngine::new(backend).with_max_sessions(4);
let queue = Mutex::new(files.iter());
std::thread::scope(|scope| {
    for _ in 0..8 {
        scope.spawn(|| {
            while let Some(path) = queue.lock().ok().and_then(|mut paths| paths.next()) {
                report(path, transcribe_file(&engine, path));
            }
        });
    }
});

fn transcribe_file(engine: &AsrEngine, path: &Path) -> Result<Transcript, Box<dyn Error + Send + Sync>> {
    let recording = audio::read(path, DecodeLimits::default())?;
    // Waits here for a free slot, within the deadline.
    Ok(engine.transcribe(&recording, AsrOptions::default(), Duration::from_secs(1800))?)
}
```

### 10. Settings UI (model picker, devices, acceleration)

```rust
let found = sherpa::inspect(model_dir)?;            // no native code runs
if found.asr.contains(&sherpa::AsrFamily::SenseVoice) {
    ui.offer("SenseVoice");
}
if found.tts.is_some() {
    ui.offer("its voices");
}
ui.enable_gpu(sherpa::Provider::Cuda.is_supported());
for device in Microphone::list()? {
    ui.add_microphone(&device.name, device.is_default);
}
```

### 11. Voice agent: an LLM you can interrupt

One cloud session hears the whole conversation. The assistant answers when
the user's turn ends, not after every segment. When the user starts talking
over a reply, the service reports it, the reply stops mid-sentence, and the
conversation keeps only the part the user heard:

```rust
let asr = AsrEngine::new(OpenAiRealtime::new(
    OpenAiRealtimeConfig::new("gpt-4o-transcribe", Secret::from_env("OPENAI_API_KEY")?),
    CloudRuntime::owned(2)?,
)?);
let speaker = Speaker::open_default()?;
let options = AsrOptions::default().with_turn_end(Duration::from_millis(700));
let listening = mic.listen(&asr, options)?;         // the whole conversation
let mut chat = Conversation::default();
let mut unanswered: Vec<String> = Vec::new();         // turns heard, not answered yet
let mut newest_speech = Duration::ZERO;              // where the user last started talking
let mut reply: Option<(Playback, LlmCancel)> = None;
for update in listening.updates() {
    match update {
        AsrUpdate::SpeechStarted { at } => {
            newest_speech = at;
            retire(&mut reply, &mut chat);          // barge-in
        }
        AsrUpdate::TurnEnded(turn) => {
            let said = turn.text();                 // this turn's segments, and only them
            if !said.is_empty() {
                unanswered.push(said);
            }
            if newest_speech > turn.end || unanswered.is_empty() {
                continue;                           // the user went on: answer when they stop
            }
            retire(&mut reply, &mut chat);          // never leave a reply playing
            chat.user(&std::mem::take(&mut unanswered).join(" "));
            let (session, playback) = speaker.speak_streaming(&tts, TtsOptions::default())?;
            let (tokens, cancel) = llm.reply(&chat);   // the request runs in the background
            std::thread::spawn(move || {
                for piece in tokens {
                    if session.push_text(&piece).is_err() {
                        break;                      // the reply was stopped
                    }
                }
                session.close_text();               // the reply plays to its end
            });
            reply = Some((playback, cancel));
        }
        AsrUpdate::Closed(result) => {
            retire(&mut reply, &mut chat);          // the last reply is kept, and stopped
            result?;
        }
        _ => {}
    }
}

/// Stops a reply that may still be playing, and keeps only what was heard.
fn retire(reply: &mut Option<(Playback, LlmCancel)>, chat: &mut Conversation) {
    if let Some((playback, cancel)) = reply.take() {
        playback.stop();                            // a no-op if it already finished
        cancel.cancel();                            // the LLM request stops too
        chat.assistant(&playback.text_played());    // only what was heard
    }
}
```

The assistant answers at `TurnEnded`, which carries every segment of the
turn, even one the service transcribes late, and nothing of the next turn. A
pause shorter than 700 ms, or an utterance cut at its maximum, doesn't end the
turn, so the assistant never answers half a sentence. If the turn's text comes
late and the user has already started speaking again, the loop keeps the turn
and answers both once the user stops. A turn with no words, such as a cough,
still releases a waiting answer.

The loop never waits:

- `speak_streaming` opens the synthesis in the background;
- the LLM streams on its own thread;
- an interruption only stops things and reads what was heard.

Stopping the LLM is the application's part. speechkit stops the synthesis and
the sound, but only the LLM client can abandon a request that is waiting for
its next token.

Every `SpeechStarted` reaches this loop even if it was busy. A reply is also
retired before the next one starts and when the conversation closes. So two
replies never play one after the other, and the part of every reply that was
heard enters the conversation. Without echo cancellation, the agent's own
voice counts as speech (see Part IV).

---

## Part IV — Deliberately not added

- **Sessions that split themselves.** A session keeps no audio, so nothing
  forces it to end early. Meeting minutes split into files by capture time,
  and the history cap of about 100 hours of speech is the only limit. If a
  service limits how long one connection lasts, its backend reconnects inside
  the stream (see "Writing a backend"). An application that wants separate
  sessions anyway starts the next listening at a pause with
  `ListenOptions::starting_at`, and nothing falls between them. Cutting an
  overlong utterance is different: it adds a segment boundary inside the
  session.
- **Handles that change type.** A capture could turn into a listening, but
  that stops the key-up thread and the session thread from sharing a handle.
  A failed transition would also lose the audio being held.
- **Guessing speech activity.** An energy detector placed behind a backend
  that reports nothing would end sessions and turns mid-sentence in a noisy
  room. Ending at a pause and reporting turn ends need `reports_activity`,
  and fail with `Unsupported` without it.
- **Raw taps on a capture.** Listenings and watchings cover the scenarios. A
  public raw reader would need a policy for readers that fall behind. Add one
  when someone needs it.
- **Mixing.** Sounds queue on a speaker; two never play at once.
- **Async streams.** Readers wait on condition variables, and a `Stream`
  would need wakers in the same places. A thread and a channel bridge to
  async code, as the server does. Add `into_stream()` behind a feature when
  an async user asks for it.
- **Callbacks.** A callback runs on the library's thread, where a slow one
  stalls audio. Handles and iterators make the threading plain.
- **Echo cancellation.** It belongs in the platform's voice-processing audio
  path, such as macOS voice processing or Windows communications mode.
  Without it, an agent's own voice counts as speech (scenario 11). A capture
  option that asks for the platform's processing is the natural next step.
- **Retries and fallback.** Retrying means replaying audio, and falling back
  changes the transcript, the timing, and the cost. The application decides,
  using `retryable()` and a listening's recording (scenario 1).
- **Downloading models.** Users fetch models. The library validates and loads
  what it is given.
- **Speaker diarization.** It works on whole recordings, so it belongs with
  `transcribe`, not with sessions. Later.
- **Streaming file decode.** `audio::read` holds the whole file in memory:
  two hours at 48 kHz is about 1.4 GB. A decoder that yields chunks would let
  `transcribe` take audio of any length. It is independent of this design.
- **More than one model per kind in `Server`, and WebSocket endpoints.**
  Later.

---

## Summary

The three verbs and one lifecycle carry every scenario above. Six ideas make
the API easy to use:

1. **Waiting is explicit.** A method either takes a deadline or is a `try_`
   that never waits. A timeout only stops waiting; only the deadline on
   `finish` ends a session.
2. **A microphone is one timeline.** A `Capture` holds the device and its
   recent history. `Listening` and `Watching` are readers that name where they
   start, and a heard wake word reserves the audio after it. One rule, no gap
   from a held position, covers key-down, a session that connects late, and
   wake word to request.
3. **Sessions open in the background** behind `listen`, `speak`, and
   `speak_streaming`, so no device call ever blocks on a model or a network.
4. **Speech activity is an event** that every reader receives, from every
   backend, even while no audio flows. Backends also say how far activity is
   known, so a turn or a session ends only at a silence the backend has
   confirmed. A turn end carries exactly that turn's words, endpointing stops
   at a point in the audio that the result reports, and barge-in is built on
   the same events.
5. **Playback is a queue of handles** that never block, can be stopped, and
   report the text that was heard.
6. **Every stream reads the same way** and ends with `Closed`. A refused chunk
   always comes back, failures keep what was confirmed, and synthesized audio
   has a reader from its first sample.

The capture timeline is the piece the device layer rests on; everything else
composes on top of it.
