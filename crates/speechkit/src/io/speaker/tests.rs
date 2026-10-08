use std::time::{Duration, Instant};

use super::*;
use crate::AudioBuffer;

// The speaker's rules with synthesis (D-05, D-06) run in
// tests/io_contract.rs, with the testkit's fake TTS.

const RATE: SampleRate = SampleRate::HZ_16000;

fn ms(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

fn frames(millis: u64) -> usize {
    usize::try_from(RATE.frames_in(ms(millis))).unwrap()
}

fn tone(level: f32, millis: u64) -> AudioBuffer {
    AudioBuffer::new(RATE, vec![level; frames(millis)])
}

/// Polls `done` for up to five seconds.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// Plays until the ring holds `frames` frames of audio, or fails. The
/// player fills the ring on its own thread.
fn play_filled(fake: &FakeSpeaker, shared: &Shared, frames: u64) -> Vec<f32> {
    let _ = shared;
    assert!(
        eventually(|| fake.consumer_len() as u64 >= frames),
        "the player did not fill the ring"
    );
    fake.play(RATE.duration_of(frames))
}

impl FakeSpeaker {
    fn consumer_len(&self) -> usize {
        use ringbuf::traits::Observer;
        lock(&self.consumer).occupied_len()
    }
}

#[test]
fn sounds_play_in_order() {
    let (speaker, fake) = Speaker::fake(RATE);
    let first = speaker.play(tone(0.1, 100)).unwrap();
    let second = speaker.play(tone(0.2, 100)).unwrap();
    let heard = play_filled(&fake, &speaker.shared, 2 * frames(100) as u64);
    assert_eq!(heard[..frames(100)], vec![0.1; frames(100)][..]);
    assert_eq!(heard[frames(100)..], vec![0.2; frames(100)][..]);
    first.finish(ms(1_000)).unwrap();
    second.finish(ms(1_000)).unwrap();
    assert_eq!(first.played(), ms(100));
    assert_eq!(second.played(), ms(100));
}

#[test]
fn a_sound_is_resampled_to_the_device_rate() {
    let (speaker, fake) = Speaker::fake(SampleRate::HZ_48000);
    let playback = speaker
        .play(AudioBuffer::new(RATE, vec![0.25; frames(200)]))
        .unwrap();
    assert!(eventually(|| {
        fake.play(ms(10));
        playback.is_done()
    }));
    playback.finish(ms(100)).unwrap();
    assert_eq!(playback.played(), ms(200));
}

#[test]
fn played_counts_only_what_the_device_played() {
    let (speaker, fake) = Speaker::fake(RATE);
    let playback = speaker.play(tone(0.1, 300)).unwrap();
    assert_eq!(playback.played(), Duration::ZERO);
    play_filled(&fake, &speaker.shared, frames(100) as u64);
    assert_eq!(playback.played(), ms(100));
    assert!(!playback.is_done());
    assert!(matches!(
        playback.finish(ms(20)),
        Err(SpeechError::DeadlineExceeded)
    ));
    assert!(playback.is_done(), "a finish that times out stops it");
    fake.play(ms(300));
    assert_eq!(
        playback.played(),
        ms(100),
        "nothing more plays once stopped"
    );
}

/// A `finish` that races a `stop` reports one of the two: `Ok` for the
/// stop, or `DeadlineExceeded` for its own deadline; never `Closed`, and
/// never an `Ok` from the player that beat the deadline's result. A stop and
/// its result are set together, so the player, which settles a stopped
/// sound with `Ok`, cannot get in between. The window is a few
/// microseconds, so the race is repeated.
#[test]
fn a_finish_racing_a_stop_reports_one_of_the_two() {
    use std::sync::Barrier;

    let (speaker, _fake) = Speaker::fake(RATE);
    for round in 0..1_000 {
        let playback = speaker.play(tone(0.1, 400)).unwrap();
        let barrier = Barrier::new(2);
        let finished = std::thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                playback.stop();
            });
            barrier.wait();
            playback.finish(Instant::now())
        });
        assert!(
            matches!(finished, Ok(()) | Err(SpeechError::DeadlineExceeded)),
            "round {round}: {finished:?}"
        );
    }
}

/// The player sleeps a tick while the ring is full, even with more sounds
/// queued behind the one playing. A queued sound is work only while nothing
/// is playing; counting the player's loops stands in for its CPU use.
#[test]
fn a_queued_sound_behind_a_playing_one_does_not_make_the_player_spin() {
    let (speaker, _fake) = Speaker::fake(RATE);
    // 30 s each: far more than the 500 ms ring, and nobody plays it.
    let _first = speaker.play(tone(0.1, 30_000)).unwrap();
    let _second = speaker.play(tone(0.2, 30_000)).unwrap();
    std::thread::sleep(ms(200));
    let before = speaker.shared.loops.load(Ordering::Relaxed);
    std::thread::sleep(ms(300));
    let loops = speaker.shared.loops.load(Ordering::Relaxed) - before;
    // A tick of 5 ms allows about 60 in 300 ms; a spinning player runs
    // tens of thousands.
    assert!(loops < 500, "the player looped {loops} times in 300 ms");
}

#[test]
fn stop_skips_the_rest_of_a_sound() {
    let (speaker, fake) = Speaker::fake(RATE);
    let first = speaker.play(tone(0.1, 400)).unwrap();
    let second = speaker.play(tone(0.2, 100)).unwrap();
    play_filled(&fake, &speaker.shared, frames(100) as u64);
    first.stop();
    first.finish(ms(100)).expect("stopping is not an error");
    // The player skips what is left of it in the ring.
    assert!(eventually(|| speaker.shared.skip.read()[1] > 0));
    let heard = fake.play(ms(100));
    assert_eq!(heard, vec![0.2; frames(100)]);
    second.finish(ms(1_000)).unwrap();
    assert_eq!(first.played(), ms(100));
}

#[test]
fn speaker_stop_clears_the_queue() {
    let (speaker, fake) = Speaker::fake(RATE);
    let playing = speaker.play(tone(0.1, 400)).unwrap();
    let queued = speaker.play(tone(0.2, 400)).unwrap();
    play_filled(&fake, &speaker.shared, frames(50) as u64);
    speaker.stop();
    playing.finish(ms(1_000)).unwrap();
    queued.finish(ms(1_000)).unwrap();
    assert_eq!(queued.played(), Duration::ZERO);
    assert!(eventually(|| speaker.shared.skip.read()[1] > 0));
    assert!(fake.play(ms(100)).iter().all(|&sample| sample == 0.0));
}

#[test]
fn a_sink_plays_what_is_pushed_until_closed() {
    let (speaker, fake) = Speaker::fake(RATE);
    let (sink, playback) = speaker.sink(RATE).unwrap();
    sink.push(&vec![0.3; frames(100)], ms(1_000)).unwrap();
    let heard = play_filled(&fake, &speaker.shared, frames(100) as u64);
    assert_eq!(heard, vec![0.3; frames(100)]);
    assert!(!playback.is_done());
    sink.close();
    assert!(eventually(|| {
        fake.play(ms(10));
        playback.is_done()
    }));
    assert!(matches!(
        sink.push(&[0.1], ms(10)),
        Err(SpeechError::Closed)
    ));
    assert!(matches!(
        sink.push(&[f32::NAN], ms(10)),
        Err(SpeechError::InvalidInput(_))
    ));
}

#[test]
fn a_full_sink_waits_until_its_deadline() {
    let (speaker, _fake) = Speaker::fake(RATE);
    let (sink, _playback) = speaker.sink(RATE).unwrap();
    // The ring takes 500 ms and the sink 2 s; nothing plays.
    let started = Instant::now();
    let pushed = sink.push(&vec![0.1; frames(4_000)], ms(200));
    assert!(
        matches!(pushed, Err(SpeechError::DeadlineExceeded)),
        "{pushed:?}"
    );
    assert!(started.elapsed() >= ms(200));
}

#[test]
fn stopped_and_dropped_playbacks_close_current_and_queued_sinks() {
    for queued in [false, true] {
        for action in ["playback", "speaker", "drop"] {
            let (speaker, fake) = Speaker::fake(RATE);
            let (first_sink, first) = speaker.sink(RATE).unwrap();
            first_sink.push(&vec![0.2; frames(100)], ms(1_000)).unwrap();
            play_filled(&fake, &speaker.shared, frames(50) as u64);
            // An open first sink keeps the second sound queued.
            let (sink, playback) = if queued {
                speaker.sink(RATE).unwrap()
            } else {
                (first_sink, first)
            };
            match action {
                "playback" => playback.stop(),
                "speaker" => speaker.stop(),
                "drop" => drop(speaker),
                _ => unreachable!(),
            }
            if action != "drop" {
                // A stop refuses pushes before it returns, playing or not.
                assert!(
                    matches!(sink.push(&[0.1], ms(10)), Err(SpeechError::Closed)),
                    "queued={queued}, action={action}"
                );
            }
            let result = playback.finish(ms(1_000));
            if action == "drop" {
                assert!(matches!(result, Err(SpeechError::Closed)), "{result:?}");
            } else {
                result.unwrap();
            }
            assert!(
                eventually(|| matches!(sink.push(&[0.1], ms(10)), Err(SpeechError::Closed))),
                "queued={queued}, action={action}"
            );
        }
    }
}

#[test]
fn stopping_a_queued_sink_wakes_a_full_queue_producer() {
    let (speaker, _fake) = Speaker::fake(RATE);
    let (_first_sink, _first) = speaker.sink(RATE).unwrap();
    let (sink, playback) = speaker.sink(RATE).unwrap();
    sink.push(&vec![0.1; frames(2_000)], ms(1_000)).unwrap();
    std::thread::scope(|scope| {
        let (send, result) = std::sync::mpsc::channel();
        scope.spawn(move || {
            let started = Instant::now();
            let pushed = sink.push(&[0.1], ms(2_000));
            send.send((started, pushed)).unwrap();
        });
        // Give the producer time to wait on the full queue.
        std::thread::sleep(ms(100));
        let stopped = Instant::now();
        playback.stop();
        let (started, pushed) = result.recv_timeout(ms(1_000)).unwrap();
        assert!(matches!(pushed, Err(SpeechError::Closed)), "{pushed:?}");
        // A push begun after the stop fails without waiting, and would not
        // show the wakeup; one begun well before it was waiting.
        assert!(
            started + ms(50) <= stopped,
            "the producer started {:?} before the stop; rerun on a less loaded machine",
            stopped.saturating_duration_since(started)
        );
    });
}

#[test]
fn a_lost_device_fails_every_playback() {
    let (speaker, fake) = Speaker::fake(RATE);
    let playing = speaker.play(tone(0.1, 400)).unwrap();
    let queued = speaker.play(tone(0.2, 400)).unwrap();
    fake.lose();
    for playback in [playing, queued] {
        let error = playback.finish(ms(1_000)).unwrap_err();
        assert!(error.retryable(), "{error:?}");
    }
    assert!(speaker.device_lost());
    assert!(speaker.play(tone(0.1, 10)).is_err());
}

#[test]
fn silence_while_a_sound_is_due_is_an_underrun() {
    let (speaker, fake) = Speaker::fake(RATE);
    let (sink, _playback) = speaker.sink(RATE).unwrap();
    sink.push(&vec![0.3; frames(200)], ms(1_000)).unwrap();
    play_filled(&fake, &speaker.shared, frames(200) as u64);
    assert_eq!(speaker.underruns(), 0);
    fake.play(ms(50));
    assert_eq!(speaker.underruns(), 1);
}

#[test]
fn dropping_the_speaker_ends_its_playbacks() {
    let (speaker, _fake) = Speaker::fake(RATE);
    let playback = speaker.play(tone(0.1, 400)).unwrap();
    drop(speaker);
    assert!(matches!(
        playback.finish(ms(1_000)),
        Err(SpeechError::Closed)
    ));
}

#[test]
fn the_sequence_cell_reads_what_was_written() {
    let cell = SeqCell::<3>::new();
    assert_eq!(cell.read(), [0, 0, 0]);
    cell.write([1, 2, 3]);
    assert_eq!(cell.read(), [1, 2, 3]);
}

#[test]
fn a_reader_that_must_not_wait_gets_nothing_during_a_write() {
    let cell = SeqCell::<2>::new();
    cell.write([1, 2]);
    assert_eq!(cell.try_read(), Some([1, 2]));
    // A write in progress leaves the sequence odd.
    cell.seq.fetch_add(1, Ordering::Relaxed);
    assert_eq!(cell.try_read(), None);
    cell.seq.fetch_add(1, Ordering::Relaxed);
    assert_eq!(cell.try_read(), Some([1, 2]));
}

/// The output callback runs in real time and never waits for the player:
/// while the player publishes new skips it keeps the ranges it saw last,
/// and takes the new ones once the write is done.
#[test]
fn the_callback_keeps_its_last_skips_while_the_player_publishes() {
    let shared = Shared::new(RATE);
    let (mut producer, mut consumer) = ring(RATE);
    // Distinct samples within the range the output keeps unclipped.
    let ramp: Vec<f32> = (0..100_u8).map(|index| f32::from(index) / 200.0).collect();
    assert_eq!(producer.push_slice(&ramp), 100);
    // The callback last saw one range, 10..20.
    let mut skips: Skips = [0; 2 * SKIPS];
    skips[..2].copy_from_slice(&[10, 20]);
    // The player is part-way through publishing another.
    shared.skip.seq.fetch_add(1, Ordering::Relaxed);
    let mut out = [0.0_f32; 30];
    shared.render(
        &mut consumer,
        &mut out,
        1,
        Duration::ZERO,
        Duration::ZERO,
        &mut skips,
    );
    assert_eq!(
        out[9..11],
        [ramp[9], ramp[20]],
        "the range it knew is still skipped"
    );
    assert_eq!(skips[..2], [10, 20]);
    // The write completes with the range 50..60; the next buffer takes it.
    shared.skip.values[0].store(50, Ordering::Relaxed);
    shared.skip.values[1].store(60, Ordering::Relaxed);
    shared.skip.seq.fetch_add(1, Ordering::Release);
    let mut next = [0.0_f32; 30];
    shared.render(
        &mut consumer,
        &mut next,
        1,
        Duration::ZERO,
        Duration::ZERO,
        &mut skips,
    );
    assert_eq!(skips[..2], [50, 60]);
    assert_eq!(
        next[9..11],
        [ramp[49], ramp[60]],
        "the new range is skipped"
    );
}
