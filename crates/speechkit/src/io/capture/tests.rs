use std::time::{Duration, Instant};

use super::timeline::{Read, Timeline};
use super::*;
use crate::{
    Deadline, SpeechError,
    asr::{AsrBackend, AsrCapabilities, AsrEvents, AsrStream},
};

// speechkit-testkit depends on this crate, so its fakes cannot be used in
// unit tests: they would implement the traits of a second copy. The
// device contract (D-01 to D-05) runs in tests/io_contract.rs.

const RATE: SampleRate = SampleRate::HZ_16000;

fn soon() -> Deadline {
    Deadline::from(Duration::from_millis(50))
}

/// Polls `done` for up to five seconds.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if done() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

/// Distinct samples, so that order can be checked.
fn ramp(from: usize, len: usize) -> Vec<f32> {
    (from..from + len)
        .map(|i| f32::from(u16::try_from(i % 32_768).unwrap()) / 32_768.0)
        .collect()
}

/// Reads everything reader `id` gets until it ends.
fn read_all(timeline: &Timeline, id: u64) -> (Vec<f32>, Read) {
    let mut heard = Vec::new();
    loop {
        match timeline.read(id, 1_000, soon()) {
            Read::Audio(chunk) => heard.extend(chunk),
            Read::Idle => panic!("the reader did not end"),
            end => return (heard, end),
        }
    }
}

#[test]
fn a_reader_gets_every_sample_from_its_start_in_order() {
    let timeline = Timeline::new(1_000, 10_000);
    timeline.append(&ramp(0, 500));
    let (id, start) = timeline.add_reader(Some(200), 0, 10_000).unwrap();
    assert_eq!(start, 200);
    timeline.append(&ramp(500, 700));
    timeline.stop_reader(id);
    timeline.append(&[]);
    timeline.append(&ramp(1_200, 100));
    let (heard, end) = read_all(&timeline, id);
    assert_eq!(heard, ramp(200, 1_000));
    assert_eq!(end, Read::End(1_200));
}

#[test]
fn a_start_before_the_history_fails_and_a_future_one_is_invalid() {
    let timeline = Timeline::new(1_000, 10_000);
    timeline.append(&ramp(0, 3_000));
    assert!(matches!(
        timeline.add_reader(Some(1_999), 0, 10_000),
        Err(SpeechError::Capacity)
    ));
    assert!(timeline.add_reader(Some(2_000), 0, 10_000).is_ok());
    assert!(matches!(
        timeline.add_reader(Some(3_001), 0, 10_000),
        Err(SpeechError::InvalidInput(_))
    ));
}

#[test]
fn only_the_history_is_held_without_readers() {
    let timeline = Timeline::new(1_000, 10_000);
    timeline.append(&ramp(0, 5_000));
    assert_eq!(timeline.held(), 1_000);
    let (id, _) = timeline.add_reader(None, 0, 100_000).unwrap();
    timeline.append(&ramp(5_000, 5_000));
    assert_eq!(timeline.held(), 5_000, "the reader's audio stays");
    timeline.remove_reader(id);
    assert_eq!(timeline.held(), 1_000);
}

#[test]
fn a_reader_behind_its_limit_fails_and_frees_the_audio() {
    let timeline = Timeline::new(0, 10_000);
    let (id, _) = timeline.add_reader(None, 0, 1_000).unwrap();
    timeline.append(&ramp(0, 600));
    assert_eq!(timeline.read(id, 100, soon()), Read::Audio(ramp(0, 100)));
    timeline.append(&ramp(600, 500));
    assert!(!timeline.is_lost(id), "exactly at its limit");
    timeline.append(&ramp(1_100, 1));
    assert!(timeline.is_lost(id));
    assert_eq!(timeline.read(id, 100, soon()), Read::Lost(100));
    assert_eq!(timeline.held(), 0);
}

#[test]
fn samples_lost_fail_the_readers_that_would_read_across_them() {
    let timeline = Timeline::new(10_000, 10_000);
    timeline.append(&ramp(0, 500));
    let (early, _) = timeline.add_reader(Some(100), 0, 100_000).unwrap();
    let (stopped, _) = timeline.add_reader(Some(200), 0, 100_000).unwrap();
    timeline.stop_reader(stopped);
    timeline.append(&[]);

    // The device lost samples after position 500, where the audio so far ends.
    timeline.append_with_loss(&[], 500);
    timeline.append(&ramp(500, 100));
    let (after, start) = timeline.add_reader(Some(500), 0, 100_000).unwrap();
    assert_eq!(start, 500);

    assert!(timeline.is_lost(early));
    assert_eq!(timeline.read(early, 1_000, soon()), Read::Lost(100));
    // A reader that stopped before the loss got all of its audio.
    let (heard, end) = read_all(&timeline, stopped);
    assert_eq!(heard, ramp(200, 300));
    assert_eq!(end, Read::End(500));
    // One that starts after it is not affected.
    assert!(!timeline.is_lost(after));
    assert_eq!(
        timeline.read(after, 1_000, soon()),
        Read::Audio(ramp(500, 100))
    );
    // Nothing can start before it, although the audio is still held.
    assert!(matches!(
        timeline.add_reader(Some(499), 0, 100_000),
        Err(SpeechError::Capacity)
    ));
    assert!(timeline.reserve(499).is_none());
    assert!(timeline.reserve(500).is_some());
}

#[test]
fn samples_lost_ahead_of_the_audio_moved_so_far_fail_the_readers_at_once() {
    let timeline = Timeline::new(10_000, 10_000);
    timeline.append(&ramp(0, 500));
    let (id, _) = timeline.add_reader(None, 0, 100_000).unwrap();
    // The device queued 300 more frames before it lost samples, and the
    // capture thread has yet to move them.
    timeline.append_with_loss(&[], 800);
    assert_eq!(timeline.read(id, 100, soon()), Read::Lost(500));
    assert!(matches!(
        timeline.add_reader(None, 0, 100_000),
        Err(SpeechError::Capacity)
    ));
    timeline.append(&ramp(500, 300));
    assert!(timeline.add_reader(None, 0, 100_000).is_ok());
}

#[test]
fn audio_from_after_a_loss_never_reaches_a_reader_that_started_before_it() {
    let timeline = Timeline::new(10_000, 10_000);
    timeline.append(&ramp(0, 500));
    let (id, _) = timeline.add_reader(Some(0), 0, 100_000).unwrap();
    // One step brings 300 frames from before the loss and 200 from after.
    timeline.append_with_loss(&ramp(500, 500), 800);
    assert_eq!(
        timeline.read(id, 1_000, soon()),
        Read::Lost(0),
        "not even the frames before it, which it could have read"
    );
}

#[test]
fn a_reservation_before_lost_samples_lapses() {
    let timeline = Timeline::new(10_000, 100_000);
    timeline.append(&ramp(0, 500));
    let before = timeline.reserve(200).unwrap();
    let after = timeline.reserve(500).unwrap();
    timeline.append_with_loss(&[], 500);
    assert!(matches!(
        timeline.claim(before, 10_000),
        Err(SpeechError::Capacity)
    ));
    assert!(timeline.claim(after, 10_000).is_ok());
}

#[test]
fn a_callback_that_overflows_records_where_the_samples_were_lost() {
    let device = Device::new(RATE, &CaptureOptions::default());
    let (mut producer, mut consumer) = HeapRb::<f32>::new(100).split();
    let lost_at = || device.lost_at.load(Ordering::Relaxed);

    device.input(&ramp(0, 60), 1, &mut producer);
    assert_eq!(lost_at(), NOTHING_LOST);
    // 40 of the next 60 fit: the loss is where the 41st would have landed.
    device.input(&ramp(60, 60), 1, &mut producer);
    assert_eq!(lost_at(), 100);
    assert_eq!(device.dropped.load(Ordering::Relaxed), 20);
    // The capture thread empties the ring, and the callback goes on.
    assert_eq!(consumer.skip(100), 100);
    device.input(&ramp(120, 30), 1, &mut producer);
    assert_eq!(lost_at(), 100, "the capture thread resets it");
    // A second loss moves it on, past the 30 frames queued since.
    device.input(&ramp(150, 100), 1, &mut producer);
    assert_eq!(lost_at(), 200);
}

#[test]
fn a_stop_takes_effect_at_the_next_append() {
    let timeline = Timeline::new(0, 10_000);
    let (id, _) = timeline.add_reader(None, 0, 10_000).unwrap();
    timeline.append(&ramp(0, 100));
    timeline.stop_reader(id);
    // Audio delivered before the stop but moved after it still counts.
    timeline.append(&ramp(100, 50));
    timeline.append(&ramp(150, 50));
    let (heard, end) = read_all(&timeline, id);
    assert_eq!(heard, ramp(0, 150));
    assert_eq!(end, Read::End(150));
}

#[test]
fn a_reservation_holds_its_audio_until_it_lapses() {
    let timeline = Timeline::new(100, 1_000);
    timeline.append(&ramp(0, 500));
    let reservation = timeline.reserve(450).unwrap();
    timeline.append(&ramp(500, 900));
    assert_eq!(timeline.held(), 950);
    let (id, start) = timeline.claim(reservation, 10_000).unwrap();
    assert_eq!(start, 450);
    assert_eq!(timeline.read(id, 10, soon()), Read::Audio(ramp(450, 10)));

    let lapsing = timeline.reserve(1_300).unwrap();
    timeline.append(&ramp(1_400, 899));
    timeline.append(&ramp(2_299, 1));
    assert!(matches!(
        timeline.claim(lapsing, 10_000),
        Err(SpeechError::Capacity)
    ));
    assert!(timeline.reserve(0).is_none(), "no longer held");
}

#[test]
fn releasing_a_reservation_frees_its_audio() {
    let timeline = Timeline::new(100, 100_000);
    timeline.append(&ramp(0, 500));
    let reservation = timeline.reserve(450).unwrap();
    timeline.append(&ramp(500, 500));
    assert_eq!(timeline.held(), 550);
    timeline.release(reservation);
    assert_eq!(timeline.held(), 100);
}

#[test]
fn the_end_ends_every_reader_at_the_last_sample_and_frees_the_rest() {
    let timeline = Timeline::new(1_000, 1_000);
    timeline.append(&ramp(0, 300));
    let (id, _) = timeline.add_reader(Some(100), 0, 10_000).unwrap();
    let reservation = timeline.reserve(200).unwrap();
    timeline.end();
    let (heard, end) = read_all(&timeline, id);
    assert_eq!(heard, ramp(100, 200));
    assert_eq!(end, Read::End(300));
    assert_eq!(timeline.held(), 0);
    assert!(matches!(
        timeline.claim(reservation, 10),
        Err(SpeechError::Closed)
    ));
    assert!(matches!(
        timeline.add_reader(None, 0, 10),
        Err(SpeechError::Closed)
    ));
}

/// A backend that keeps every sample it is given.
struct Collect(AsrCapabilities, Arc<Mutex<Vec<f32>>>);

struct CollectStream(Arc<Mutex<Vec<f32>>>);

impl AsrBackend for Collect {
    fn name(&self) -> &'static str {
        "collect"
    }
    fn capabilities(&self) -> &AsrCapabilities {
        &self.0
    }
    fn open(&self, _: &AsrOptions, _: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        Ok(Box::new(CollectStream(self.1.clone())))
    }
}

impl AsrStream for CollectStream {
    fn accept(&mut self, samples: &[f32]) -> Result<(), SpeechError> {
        self.0.lock().unwrap().extend_from_slice(samples);
        Ok(())
    }
    fn finish(&mut self) -> Result<(), SpeechError> {
        Ok(())
    }
}

fn collect() -> (AsrEngine, Arc<Mutex<Vec<f32>>>) {
    let heard = Arc::new(Mutex::new(Vec::new()));
    let engine = AsrEngine::new(Collect(AsrCapabilities::new(RATE), heard.clone()));
    (engine, heard)
}

/// A backend whose streams fail on the first audio.
struct Failing(AsrCapabilities);

struct FailingStream;

impl AsrBackend for Failing {
    fn name(&self) -> &'static str {
        "failing"
    }
    fn capabilities(&self) -> &AsrCapabilities {
        &self.0
    }
    fn open(&self, _: &AsrOptions, _: AsrEvents) -> Result<Box<dyn AsrStream>, SpeechError> {
        Ok(Box::new(FailingStream))
    }
}

impl AsrStream for FailingStream {
    fn accept(&mut self, _: &[f32]) -> Result<(), SpeechError> {
        Err(SpeechError::backend("failing", true, "dropped"))
    }
    fn finish(&mut self) -> Result<(), SpeechError> {
        Ok(())
    }
}

fn fake(options: CaptureOptions) -> (Capture, FakeMicrophone) {
    let (microphone, fake) = Microphone::fake(RATE);
    (microphone.capture(options).unwrap(), fake)
}

/// Waits until the capture has moved `frames` frames into its timeline.
fn moved(capture: &Capture, frames: u64) {
    assert!(
        eventually(|| capture.position() >= RATE.duration_of(frames)),
        "the capture thread did not move the audio"
    );
}

#[test]
fn a_listening_feeds_its_session_from_its_start_in_capture_time() {
    let (capture, mic) = fake(CaptureOptions::default());
    mic.push(&ramp(0, 8_000));
    moved(&capture, 8_000);
    let (engine, heard) = collect();
    let listening = capture
        .listen(
            &engine,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::from_millis(200)),
        )
        .unwrap();
    assert_eq!(listening.origin(), Duration::from_millis(200));
    mic.push(&ramp(8_000, 8_000));
    moved(&capture, 16_000);
    listening.stop();
    let transcript = listening.finish(Duration::from_secs(5)).unwrap();
    assert_eq!(*heard.lock().unwrap(), ramp(3_200, 12_800));
    assert_eq!(transcript.duration, Duration::from_millis(800));
    assert_eq!(listening.end(), Some(Duration::from_secs(1)));
}

#[test]
fn listen_checks_its_options_at_once() {
    let (capture, _mic) = fake(CaptureOptions::default());
    let (engine, _) = collect();
    let error = capture
        .listen(
            &engine,
            AsrOptions::default(),
            ListenOptions::default().with_max_backlog(Duration::from_millis(10)),
        )
        .unwrap_err();
    assert!(matches!(error, SpeechError::InvalidInput(_)), "{error:?}");
    let error = capture
        .listen(
            &engine,
            AsrOptions::default().with_hints(["kit"]),
            ListenOptions::default(),
        )
        .unwrap_err();
    assert!(matches!(error, SpeechError::Unsupported(_)), "{error:?}");
}

#[test]
fn a_failed_session_keeps_recording_until_stop() {
    let (capture, mic) = fake(CaptureOptions::default());
    let engine = AsrEngine::new(Failing(AsrCapabilities::new(RATE)));
    let listening = capture
        .listen(
            &engine,
            AsrOptions::default(),
            ListenOptions::default().with_recording(Duration::from_secs(60)),
        )
        .unwrap();
    mic.push(&ramp(0, 4_000));
    assert!(eventually(|| listening.result().is_some()));
    mic.push(&ramp(4_000, 4_000));
    moved(&capture, 8_000);
    assert_eq!(
        listening.end(),
        None,
        "a failure does not end the listening"
    );
    let failure = listening.finish(Duration::from_secs(5)).unwrap_err();
    assert!(failure.error.retryable());
    let recording = listening.recording().unwrap();
    assert_eq!(recording.audio.samples, ramp(0, 8_000));
    assert!(!recording.truncated);
}

#[test]
fn a_recording_stops_at_its_limit() {
    let (capture, mic) = fake(CaptureOptions::default());
    let (engine, _) = collect();
    let listening = capture
        .listen(
            &engine,
            AsrOptions::default(),
            ListenOptions::default().with_recording(Duration::from_millis(100)),
        )
        .unwrap();
    mic.push(&ramp(0, 4_000));
    moved(&capture, 4_000);
    listening.finish(Duration::from_secs(5)).unwrap();
    let recording = listening.recording().unwrap();
    assert_eq!(recording.audio.samples, ramp(0, 1_600));
    assert!(recording.truncated);
}

#[test]
fn a_listening_without_a_recording_has_none() {
    let (capture, _mic) = fake(CaptureOptions::default());
    let (engine, _) = collect();
    let listening = capture
        .listen(&engine, AsrOptions::default(), ListenOptions::default())
        .unwrap();
    assert!(listening.recording().is_none());
}

#[test]
fn level_is_the_rms_of_the_latest_buffer_and_zero_once_lost() {
    let (capture, mic) = fake(CaptureOptions::default());
    mic.push(&[0.5, -0.5, 0.5, -0.5]);
    assert!((capture.level() - 0.5).abs() < 1e-6);
    mic.lose();
    assert!(capture.device_lost());
    assert!(capture.level().abs() < f32::EPSILON);
}

#[test]
fn the_last_handle_stops_the_microphone() {
    let (microphone, mic) = Microphone::fake(RATE);
    let (engine, heard) = collect();
    let listening = microphone.listen(&engine, AsrOptions::default()).unwrap();
    mic.push(&ramp(0, 1_600));
    listening.stop();
    // The microphone stopped, so this audio is lost.
    mic.push(&ramp(1_600, 1_600));
    listening.finish(Duration::from_secs(5)).unwrap();
    assert_eq!(*heard.lock().unwrap(), ramp(0, 1_600));
}

#[test]
fn a_capture_stop_ends_every_listening_at_the_last_sample() {
    let (capture, mic) = fake(CaptureOptions::default());
    let (engine, heard) = collect();
    let listening = capture
        .listen(&engine, AsrOptions::default(), ListenOptions::default())
        .unwrap();
    mic.push(&ramp(0, 3_200));
    capture.stop();
    assert!(eventually(|| listening.end().is_some()));
    assert_eq!(listening.end(), Some(Duration::from_millis(200)));
    listening.finish(Duration::from_secs(5)).unwrap();
    assert_eq!(*heard.lock().unwrap(), ramp(0, 3_200));
    assert!(matches!(
        capture.listen(&engine, AsrOptions::default(), ListenOptions::default()),
        Err(SpeechError::Closed)
    ));
}
