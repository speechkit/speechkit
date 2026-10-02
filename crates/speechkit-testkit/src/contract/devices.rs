//! The device contract suite: rules `D-01` to `D-06`.
//!
//! Each `dXX_*` function checks one rule on the hidden fake devices,
//! [`Microphone::fake`] and [`Speaker::fake`], with [`FakeAsr`] sessions,
//! a [`FakeWakeWord`] detector, and [`FakeTts`] syntheses. The fake
//! microphone delivers audio as fast as a test pushes it, and the fake
//! speaker plays only when told to, so the checks never wait on real time
//! for the audio itself. Real devices are checked by hand, with the
//! checklists in `docs/manual-tests.md`.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use speechkit::{
    AudioBuffer, RecvError, SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions, AsrResult},
    io::{
        Capture, CaptureOptions, FakeMicrophone, FakeSpeaker, ListenOptions, Microphone, Playback,
        Speaker, WakeUpdate, WatchOptions, Watching,
    },
    tts::{TtsEngine, TtsOptions},
};

use super::SETTLE;
use crate::{
    Gate,
    asr::{FakeAsr, FakeStats, Script, Step, Trigger},
    eventually, secs,
    tts::{FakeTts, FakeTtsStats, level, samples_for},
    wake::FakeWakeWord,
};

/// The rate of every fake microphone here.
pub const RATE: SampleRate = SampleRate::HZ_16000;

/// How long a call that returns at once may take.
const AT_ONCE: Duration = Duration::from_millis(100);

/// Frames in `millis` milliseconds at [`RATE`].
fn ms(millis: u64) -> usize {
    usize::try_from(RATE.frames_in(Duration::from_millis(millis))).expect("fits")
}

/// Distinct samples below [`FakeWakeWord::MARK`], so that order can be
/// checked: sample `i` of the stream is `ramp(i, 1)[0]`.
pub fn ramp(from: usize, len: usize) -> Vec<f32> {
    (from..from + len)
        .map(|i| (i % 30_000) as f32 / 32_768.0)
        .collect()
}

fn capture(history: Duration) -> (Capture, FakeMicrophone) {
    let (microphone, fake) = Microphone::fake(RATE);
    let capture = microphone
        .capture(CaptureOptions::default().with_history(history))
        .expect("start the fake microphone");
    (capture, fake)
}

/// Waits until the capture has moved `frames` frames into its timeline.
fn moved(capture: &Capture, frames: usize) {
    let position = RATE.duration_of(frames as u64);
    assert!(
        eventually(SETTLE, || capture.position() >= position),
        "the capture thread did not move the audio"
    );
}

/// Waits until the capture thread has stopped moving audio: its position
/// has not changed for 100 ms.
fn quiet(capture: &Capture) {
    let mut last = capture.position();
    let mut since = Instant::now();
    assert!(
        eventually(SETTLE, || {
            let now = capture.position();
            if now != last {
                last = now;
                since = Instant::now();
            }
            since.elapsed() >= Duration::from_millis(100)
        }),
        "the capture thread kept moving audio"
    );
}

/// Delivers `samples` 100 ms at a time, each once the capture has moved
/// the one before: faster than a device, but in steps like one, so a
/// reader is only as far behind as its session makes it.
fn deliver(capture: &Capture, mic: &FakeMicrophone, samples: &[f32]) {
    let mut at = usize::try_from(RATE.frames_in(capture.position())).expect("fits");
    for piece in samples.chunks(ms(100)) {
        mic.push(piece);
        at += piece.len();
        moved(capture, at);
    }
}

/// An engine whose sessions record what they hear, with stats to read it.
fn recorder() -> (AsrEngine, Arc<FakeStats>) {
    engine(Script::new())
}

fn engine(script: Script) -> (AsrEngine, Arc<FakeStats>) {
    let fake = FakeAsr::new(script);
    let stats = fake.stats();
    (AsrEngine::new(fake), stats)
}

/// An engine whose sessions open only once `gate` is released.
fn opening_late(gate: &Gate) -> (AsrEngine, Arc<FakeStats>) {
    engine(Script::new().then(Trigger::OnOpen, Step::BlockUntilReleased(gate.clone())))
}

/// An engine whose sessions block in their first `accept` until `gate`
/// is released.
fn stuck(gate: &Gate) -> (AsrEngine, Arc<FakeStats>) {
    engine(Script::new().then(
        Trigger::AfterSamples(1),
        Step::BlockUntilReleased(gate.clone()),
    ))
}

fn error_of(result: &AsrResult) -> Option<&SpeechError> {
    result.as_ref().err().map(|failure| &failure.error)
}

/// The next update of `watching`, within [`SETTLE`].
fn next_update(watching: &mut Watching) -> WakeUpdate {
    watching.recv(SETTLE).expect("an update")
}

/// D-01: a reader that starts at a held position receives every sample
/// from there on, in order, even from a session that opens late, and
/// readers that overlap receive the same samples. A position no longer
/// held fails at once with `Capacity`.
pub fn d01_held_position_complete() {
    let (capture, mic) = capture(Duration::from_secs(1));
    mic.push(&ramp(0, ms(1_500)));
    moved(&capture, ms(1_500));

    let gate = Gate::new();
    let (late, late_stats) = opening_late(&gate);
    let (prompt, prompt_stats) = recorder();
    let first = capture
        .listen(
            &late,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::from_millis(800)),
        )
        .expect("0.8 s is held");
    let second = capture
        .listen(
            &prompt,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::from_millis(1_200)),
        )
        .expect("1.2 s is held");
    assert_eq!(first.origin(), Duration::from_millis(800));
    assert_eq!(second.origin(), Duration::from_millis(1_200));

    // More than the input queue arrives before the first session opens.
    mic.push(&ramp(ms(1_500), ms(3_000)));
    moved(&capture, ms(4_500));
    first.stop();
    second.stop();
    assert!(
        gate.wait_entered(1, SETTLE),
        "the first session never opened"
    );
    gate.release();
    first.finish(secs(10)).expect("the first listening");
    second.finish(secs(10)).expect("the second listening");
    assert_eq!(first.end(), Some(Duration::from_millis(4_500)));

    assert!(
        late_stats.heard() == ramp(ms(800), ms(3_700)),
        "the late session missed or reordered samples"
    );
    assert!(
        prompt_stats.heard() == ramp(ms(1_200), ms(3_300)),
        "the overlapping reader missed or reordered samples"
    );

    let error = capture
        .listen(
            &prompt,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::from_millis(3_000)),
        )
        .expect_err("3 s is older than the history");
    assert!(matches!(error, SpeechError::Capacity), "{error:?}");
}

/// D-02: a listening or a watching further behind than its limit, or one
/// that would read across samples the microphone lost, fails with
/// `Capacity`, and never skips audio: what its session heard is a prefix
/// of what it was given.
pub fn d02_behind_fails() {
    behind_its_limit_fails();
    overflow_fails_readers();
}

fn behind_its_limit_fails() {
    let (capture, mic) = capture(Duration::ZERO);
    let gate = Gate::new();
    let (stuck, stats) = stuck(&gate);
    let listening = capture
        .listen(
            &stuck,
            AsrOptions::default(),
            ListenOptions::default()
                .with_max_backlog(Duration::from_millis(500))
                .with_recording(Duration::from_secs(10)),
        )
        .expect("listen");
    let mut watching = capture
        .watch_with(
            &FakeWakeWord::new().with_delay(Duration::from_millis(200)),
            WatchOptions::default().with_max_backlog(Duration::from_millis(500)),
        )
        .expect("watch");

    // The session takes one block, then its input queue (2 s), then
    // nothing; the listening holds at most 500 ms more.
    deliver(&capture, &mic, &ramp(0, ms(4_000)));
    let result = listening.wait(SETTLE).expect("the listening failed");
    assert!(
        matches!(error_of(&result), Some(SpeechError::Capacity)),
        "{result:?}"
    );
    assert!(
        eventually(SETTLE, || listening.end().is_some()),
        "a listening behind its limit ends"
    );
    assert!(
        listening.recording().expect("a recording").truncated,
        "the recording cannot go on past the gap"
    );
    gate.release();
    assert!(
        eventually(SETTLE, || stats.alive() == 0),
        "the session did not end"
    );
    let heard = stats.heard();
    assert!(
        !heard.is_empty() && heard == ramp(0, heard.len()),
        "the session skipped audio"
    );

    match next_update(&mut watching) {
        WakeUpdate::Closed(Err(SpeechError::Capacity)) => {}
        other => panic!("expected Closed(Err(Capacity)), got {other:?}"),
    }
    assert!(matches!(watching.try_recv(), Err(RecvError::Closed)));
}

/// The microphone's buffer overflows: the samples that do not fit are
/// lost, which fails every reader that had not read past them, and keeps
/// any reader from starting before them. Readers that start after them
/// are not affected.
fn overflow_fails_readers() {
    let (capture, mic) = capture(Duration::from_secs(5));
    let (whole, stats) = recorder();
    let listening = capture
        .listen(
            &whole,
            AsrOptions::default(),
            ListenOptions::default().with_recording(Duration::from_secs(30)),
        )
        .expect("listen");
    let mut watching = capture.watch(&FakeWakeWord::new()).expect("watch");

    // Ten seconds arrive in one callback, and the buffer holds two.
    mic.push_burst(&ramp(0, ms(10_000)));
    let result = listening.wait(SETTLE).expect("the listening failed");
    assert!(
        matches!(error_of(&result), Some(SpeechError::Capacity)),
        "{result:?}"
    );
    assert!(
        eventually(SETTLE, || listening.end().is_some()),
        "a listening that lost audio ends"
    );
    assert!(
        listening.recording().expect("a recording").truncated,
        "the recording cannot go on past the gap"
    );
    assert!(
        eventually(SETTLE, || stats.alive() == 0),
        "the session did not end"
    );
    let heard = stats.heard();
    assert!(
        heard == ramp(0, heard.len()),
        "the session skipped audio (heard {} frames)",
        heard.len()
    );
    match next_update(&mut watching) {
        WakeUpdate::Closed(Err(SpeechError::Capacity)) => {}
        other => panic!("expected Closed(Err(Capacity)), got {other:?}"),
    }

    // The audio before the loss is still held, but is no longer whole from
    // its start.
    let error = capture
        .listen(
            &whole,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::ZERO),
        )
        .expect_err("the audio before the loss is not whole");
    assert!(matches!(error, SpeechError::Capacity), "{error:?}");

    // Once the capture thread has moved what the buffer held, a reader
    // that starts after the loss hears everything from there on.
    quiet(&capture);
    let (later, later_stats) = recorder();
    let fresh = capture
        .listen(&later, AsrOptions::default(), ListenOptions::default())
        .expect("a reader can start after the loss");
    mic.push(&ramp(20_000, ms(300)));
    fresh.finish(SETTLE).expect("the later listening finishes");
    assert_eq!(later_stats.heard(), ramp(20_000, ms(300)));
}

/// D-03: a heard wake word reserves the audio after it for 30 s of
/// capture, whether or not the application has read the event yet.
pub fn d03_wake_reservation() {
    // Almost no history: only the reservation can hold the audio.
    let (capture, mic) = capture(Duration::from_millis(100));
    let mut watching = capture.watch(&FakeWakeWord::new()).expect("watch");
    let mut keyword = ramp(0, ms(1_000));
    keyword.push(FakeWakeWord::MARK);
    let after = ms(1_000) + 1;
    mic.push(&keyword);
    // Four seconds pass before the application reads the event.
    mic.push(&ramp(after, ms(4_000)));
    moved(&capture, after + ms(4_000));

    let WakeUpdate::Heard(wake) = next_update(&mut watching) else {
        panic!("expected a wake word");
    };
    let end = RATE.duration_of(after as u64);
    assert_eq!(wake.event().end, end, "the event is in capture time");
    assert_eq!(wake.event().keyword, FakeWakeWord::KEYWORD);
    let (engine, stats) = recorder();
    let listening = wake
        .listen(&engine, AsrOptions::default())
        .expect("the audio after the keyword is reserved");
    assert_eq!(listening.origin(), end);
    listening.stop();
    listening.finish(secs(10)).expect("the request");
    assert!(
        stats.heard() == ramp(after, ms(4_000)),
        "the request did not start right after the keyword"
    );

    // A reservation lapses after 30 s of capture.
    let start = after + ms(4_000);
    let mut keyword = ramp(start, ms(100));
    keyword.push(FakeWakeWord::MARK);
    mic.push(&keyword);
    let later = start + ms(100) + 1;
    mic.push(&ramp(later, ms(31_000)));
    moved(&capture, later + ms(31_000));
    let WakeUpdate::Heard(wake) = next_update(&mut watching) else {
        panic!("expected a second wake word");
    };
    let error = wake
        .listen(&engine, AsrOptions::default())
        .expect_err("the reservation lapsed");
    assert!(matches!(error, SpeechError::Capacity), "{error:?}");
    watching.stop();
}

/// D-04: `stop()` returns at once on every handle, even while a session
/// is still opening or a detector is busy; `finish(deadline)` bounds the
/// rest.
pub fn d04_stop_returns_at_once() {
    let (capture, mic) = capture(Duration::from_secs(1));
    let gate = Gate::new();
    let (stuck, _) = opening_late(&gate);
    let first = capture
        .listen(&stuck, AsrOptions::default(), ListenOptions::default())
        .expect("listen");
    let second = capture
        .listen(&stuck, AsrOptions::default(), ListenOptions::default())
        .expect("listen");
    let watching = capture
        .watch(&FakeWakeWord::new().with_delay(Duration::from_millis(500)))
        .expect("watch");
    mic.push(&ramp(0, ms(500)));
    assert!(gate.wait_entered(1, SETTLE), "no session started opening");

    let started = Instant::now();
    first.stop();
    watching.stop();
    capture.stop();
    assert!(started.elapsed() < AT_ONCE, "stop blocked");

    let started = Instant::now();
    let result = second.finish(Duration::from_millis(200));
    let waited = started.elapsed();
    assert!(
        matches!(error_of(&result), Some(SpeechError::DeadlineExceeded)),
        "{result:?}"
    );
    assert!(waited < Duration::from_secs(1), "finish took {waited:?}");
    gate.release();
    first.finish(secs(10)).expect("the first listening");
}

/// D-05: a lost device ends every reader at its last sample, and each
/// session finishes with what it heard; a lost speaker fails every queued
/// playback and closes its sink to further samples.
pub fn d05_device_lost() {
    let (capture, mic) = capture(Duration::from_secs(1));
    let (engine, stats) = recorder();
    let listening = capture
        .listen(&engine, AsrOptions::default(), ListenOptions::default())
        .expect("listen");
    let mut watching = capture.watch(&FakeWakeWord::new()).expect("watch");
    mic.push(&ramp(0, ms(500)));
    mic.lose();
    assert!(capture.device_lost() && listening.device_lost());

    assert!(
        eventually(SETTLE, || listening.end().is_some()),
        "the listening did not end"
    );
    assert_eq!(listening.end(), Some(Duration::from_millis(500)));
    listening
        .wait(SETTLE)
        .expect("the session ended by itself")
        .expect("with what it heard");
    assert!(
        stats.heard() == ramp(0, ms(500)),
        "the session missed audio"
    );
    match next_update(&mut watching) {
        WakeUpdate::Closed(Ok(())) => {}
        other => panic!("expected Closed(Ok), got {other:?}"),
    }
    mic.push(&ramp(ms(500), ms(100)));
    assert_eq!(capture.position(), Duration::from_millis(500));

    let (speaker, fake) = Speaker::fake(RATE);
    let playing = speaker
        .play(AudioBuffer::new(RATE, ramp(0, ms(400))))
        .expect("play");
    let queued = speaker
        .play(AudioBuffer::new(RATE, ramp(0, ms(400))))
        .expect("play");
    fake.play(Duration::from_millis(100));
    fake.lose();
    for playback in [playing, queued] {
        let error = playback.finish(SETTLE).expect_err("the speaker was lost");
        assert!(error.retryable(), "{error:?}");
    }
    assert!(speaker.device_lost());
    assert!(
        speaker.play(AudioBuffer::new(RATE, ramp(0, 10))).is_err(),
        "a lost speaker plays nothing more"
    );

    let (speaker, fake) = Speaker::fake(RATE);
    let (playing_sink, playing) = speaker.sink(RATE).expect("first sink");
    let (queued_sink, queued) = speaker.sink(RATE).expect("queued sink");
    // The first sink stays open, so the second cannot start playing.
    fake.lose();
    for (sink, playback) in [(playing_sink, playing), (queued_sink, queued)] {
        assert!(
            playback
                .finish(SETTLE)
                .expect_err("device lost")
                .retryable()
        );
        assert!(matches!(
            sink.push(&[0.1], SETTLE),
            Err(SpeechError::Closed)
        ));
    }
}

/// What the fake speaker has played of a [`FakeTts`] synthesis of `text`.
struct Heard {
    /// Samples heard of each chunk, by the chunk's level.
    per_chunk: Vec<usize>,
    total: usize,
}

impl Heard {
    fn new() -> Self {
        Self {
            per_chunk: Vec::new(),
            total: 0,
        }
    }

    fn add(&mut self, samples: &[f32]) {
        for &sample in samples {
            // Played unchanged, since every rate is the same.
            let Some(index) = (0..900).find(|&index| level(index).to_bits() == sample.to_bits())
            else {
                continue;
            };
            if self.per_chunk.len() <= index {
                self.per_chunk.resize(index + 1, 0);
            }
            self.per_chunk[index] += 1;
            self.total += 1;
        }
    }

    /// The bytes of `text` whose chunks have been heard in full.
    fn text_heard(&self, text: &str, stats: &FakeTtsStats) -> usize {
        let mut end = 0;
        let mut from = 0;
        for (index, chunk) in stats.chunks().iter().enumerate() {
            let chunk = chunk.trim();
            let Some(at) = text[from..].find(chunk) else {
                break;
            };
            from += at + chunk.len();
            if self.per_chunk.get(index).copied().unwrap_or(0) < samples_for(chunk) {
                break;
            }
            end = from;
        }
        end
    }
}

/// Plays 10 ms at a time until `until` holds, checking D-06 after each.
fn play_checking(
    fake: &FakeSpeaker,
    playback: &Playback,
    text: &str,
    stats: &FakeTtsStats,
    heard: &mut Heard,
    mut until: impl FnMut(&Heard) -> bool,
) {
    let started = Instant::now();
    while !until(heard) {
        assert!(started.elapsed() < SETTLE, "the playback did not progress");
        heard.add(&fake.play(Duration::from_millis(10)));
        let played = playback.played();
        assert!(
            played <= RATE.duration_of(heard.total as u64),
            "played() says {played:?}, but the device played {} samples",
            heard.total
        );
        let said = playback.text_played();
        assert!(text.starts_with(&said), "{said:?} is not what was pushed");
        assert!(
            said.trim_end().len() <= heard.text_heard(text, stats),
            "text_played() says {said:?}, but less was heard"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// D-06: `played()` and `text_played()` never exceed what the device has
/// played, and stay put once the playback is stopped.
pub fn d06_played_bounded() {
    let text = "One two. Three four five. Six seven.";
    let (speaker, fake) = Speaker::fake(RATE);
    let tts = FakeTts::plain();
    let stats = tts.stats();
    let engine = TtsEngine::new(tts);

    let playback = speaker
        .speak(&engine, text, TtsOptions::default())
        .expect("speak");
    let mut heard = Heard::new();
    play_checking(&fake, &playback, text, &stats, &mut heard, |_| {
        playback.is_done()
    });
    playback.finish(SETTLE).expect("played to the end");
    assert_eq!(playback.text_played(), text);
    assert_eq!(playback.played(), RATE.duration_of(heard.total as u64));

    // Stopped halfway through the second sentence, only the first was
    // heard.
    let before = stats.chunks().len();
    let playback = speaker
        .speak(&engine, text, TtsOptions::default())
        .expect("speak");
    let mut heard = Heard::new();
    let first = samples_for("One two.");
    let half = first + samples_for("Three four five.") / 2;
    play_checking(&fake, &playback, text, &stats, &mut heard, |heard| {
        heard.total >= half
    });
    playback.stop();
    assert_eq!(stats.chunks().len() - before, 3, "the synthesis ran ahead");
    let said = playback.text_played();
    assert_eq!(said.trim_end(), "One two.");
    let played = playback.played();
    fake.play(Duration::from_millis(500));
    assert_eq!(
        playback.played(),
        played,
        "a stopped playback plays no more"
    );
    assert_eq!(playback.text_played(), said);
    playback.finish(SETTLE).expect("stopping is not an error");
}
