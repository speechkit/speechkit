Microphones and speakers (feature `devices`), built on cpal.

`Microphone::capture` starts a microphone and returns a `Capture`: a running microphone with one timeline. Capture time is the audio captured since it started, and the capture keeps the last few seconds (`CaptureOptions::with_history`, 5 s by default). Every reader of a capture starts at a position the capture still holds and receives every sample from there on, in order; an older position fails at once with `SpeechError::Capacity`. Readers may overlap, and each gets the same samples.

- `Capture::listen` starts one request, a `Listening`. It checks the options and returns at once; its session opens in the background (waiting for a slot, loading, or connecting) while the listening holds the audio, up to `ListenOptions::with_max_backlog` (30 s). The session then receives the held audio first and live audio after it, and its times are capture times. `ListenOptions::starting_at` starts it in the past, for a word spoken as the key went down, and `with_recording` keeps the audio for a retry: a session that fails doesn't end the listening, so the recording goes on until it is stopped, and it ends where the listening does, at `Listening::end`.
- `Capture::watch` runs a wake-word detector on its own thread, a `Watching`. When it hears a keyword, it reserves the audio after it at once, before the application reads the event, so `Wake::listen` starts right after the keyword. A reservation lapses once the capture is 30 s past the keyword.
- `Microphone::listen` is the short way to one request: it captures, then listens from the start.

```rust,no_run
use std::time::Duration;

use speechkit::asr::{AsrEngine, AsrOptions};
use speechkit::io::{CaptureOptions, ListenOptions, Microphone, WakeUpdate};
use speechkit::wake::WakeWordModel;

fn push_to_talk(engine: &AsrEngine, key_is_down: impl Fn() -> bool) -> Result<String, Box<dyn std::error::Error>> {
    let mic = Microphone::open_default()?;
    let capture = mic.capture(CaptureOptions::default())?;
    // Key down: start 300 ms back, so the first word is kept.
    let from = capture.position().saturating_sub(Duration::from_millis(300));
    let listening = capture.listen(engine, AsrOptions::default(), ListenOptions::starting_at(from))?;
    while key_is_down() {
        std::thread::sleep(Duration::from_millis(20));
    }
    listening.stop(); // returns at once
    Ok(listening.finish(Duration::from_secs(10))?.text())
}

fn assistant(engine: &AsrEngine, keywords: &dyn WakeWordModel, answer: impl Fn(&str)) -> Result<(), Box<dyn std::error::Error>> {
    let capture = Microphone::open_default()?.capture(CaptureOptions::default())?;
    let request = AsrOptions::default().with_end_after_silence(Duration::from_secs(1));
    for update in capture.watch(keywords)? {
        match update {
            WakeUpdate::Heard(wake) => {
                let listening = wake.listen(engine, request.clone())?;
                let heard = listening
                    .wait(Duration::from_secs(30))
                    .unwrap_or_else(|| listening.finish(Duration::from_secs(5)))?;
                answer(&heard.text());
            }
            WakeUpdate::Closed(result) => return Ok(result?),
            _ => {}
        }
    }
    Ok(())
}
```

A listening ends when it is stopped, finished, or cancelled; when its session ends by itself (at a pause, with no speech, or at its maximum length), and `Listening::end` is then the cutoff; or when the capture stops or the device is lost, at the last sample. The microphone runs while a `Capture` handle or an unfinished listening or watching exists, so a listening that is the capture's only user stops the microphone when it ends. `Capture::stop` stops it for every reader. Every `stop` returns at once, and `finish` bounds the rest with its deadline.

The cpal callback only converts samples to mono f32, writes them to a lock-free ring buffer, and stores their level: no allocation, locking, or logging. The capture's own thread moves the audio into the timeline every 20 ms and frees what no reader, reservation, or the history needs. Each listening and watching reads from the timeline on its own thread, so a slow session or detector never holds up the others; one that falls further behind than its `max_backlog` fails with `SpeechError::Capacity` rather than skip audio. So does one that would read across samples the microphone lost, when its 2 s buffer overflowed because the machine fell behind; no reader can start before them. Lost samples and a lost device are logged at `warn`; `device_lost` says so, and a lost device ends every reader at its last sample.

A `Speaker` plays sounds one after another, in the order they were queued, each resampled to the device's rate: `speak` synthesizes text and plays it as the audio arrives, `speak_streaming` does the same for text that is still being written (an LLM's reply, say), `play` takes a `TtsOutput` or an `AudioBuffer`, and `sink` takes samples you push yourself. Each returns a `Playback` at once; the synthesis opens in the background. `Playback::stop` skips the sound and cancels its synthesis, and `finish` waits for it to play out. `played` counts only audio whose playback time has passed, so `text_played`, the text up to the last mark whose audio has played, is what the listener heard: what a voice agent keeps when the user interrupts.

```rust,no_run
use std::time::Duration;

use speechkit::io::Speaker;
use speechkit::tts::{TtsEngine, TtsOptions};

fn reply(tts: &TtsEngine, tokens: impl Iterator<Item = String>, interrupted: impl Fn() -> bool) -> Result<String, Box<dyn std::error::Error>> {
    let speaker = Speaker::open_default()?;
    let (session, playback) = speaker.speak_streaming(tts, TtsOptions::default())?;
    for token in tokens {
        if interrupted() {
            playback.stop(); // returns at once; the synthesis is cancelled
            break;
        }
        session.push_text(&token)?;
    }
    session.close_text();
    playback.finish(Duration::from_secs(60))?;
    Ok(playback.text_played()) // only what was heard
}
```

The speaker's output stream starts with the first sound and runs until the speaker is dropped. A player thread moves each sound into a lock-free ring buffer, and the output callback only reads it, skips the audio of stopped sounds, and publishes how far it got with the time that audio will play: no allocation, locking, or logging. Silence while a sound was due counts as an underrun, which `Speaker::underruns` counts and the player logs at `warn`. A lost device fails every queued playback and closes its sinks.

`Microphone::open_default` and `Speaker::open_default` use the system's default devices. `Microphone::list` and `Speaker::list` list the others, and `Microphone::open` and `Speaker::open` take a name from those lists. A name matches exactly, then ignoring case, then as a part of a name; whichever of these first finds a device must find only one.

On Linux, building needs `libasound2-dev` and `pkg-config`.
