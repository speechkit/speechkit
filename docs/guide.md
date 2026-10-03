# speechkit guide

Eleven applications built on the public API, from a dictation input method
to a voice agent the user can interrupt. Each shows which handles to hold and
where the waiting happens; the rustdoc has the full signatures.

Imports are left out. Each scenario runs in a function that returns
`Result<(), Box<dyn Error>>`, or in an `async fn` for the two servers. Names
such as `ui`, `worker`, `brain`, and `llm` are application code.

## 1. Dictation input method (voice IME)

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

## 2. Meeting transcription

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

## 3. Transcribe a file

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

## 4. ASR service (OpenAI-compatible HTTP)

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

## 5. Text to speech

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

## 6. TTS service (OpenAI-compatible HTTP)

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

## 7. Voice assistant (wake word → one request → reply)

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

## 8. Live captions (your own audio source, no device)

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

## 9. Batch transcription pipeline

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

## 10. Settings UI (model picker, devices, acceleration)

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

## 11. Voice agent: an LLM you can interrupt

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
heard enters the conversation. speechkit does no echo cancellation, so
without the platform's voice processing (macOS voice processing, Windows
communications mode) the agent's own voice counts as speech.
