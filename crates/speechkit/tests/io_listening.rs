//! Listenings on the fake microphone: how they end.
#![cfg(feature = "devices")]

use std::time::Duration;

use speechkit::{
    SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions, AsrUpdate},
    io::{
        Capture, CaptureOptions, FakeMicrophone, ListenOptions, Microphone, Wake, WakeUpdate,
        Watching,
    },
};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    contract::devices::ramp,
    eventually,
    wake::FakeWakeWord,
};

const RATE: SampleRate = SampleRate::HZ_16000;
const SETTLE: Duration = Duration::from_secs(10);

fn ms(millis: u64) -> usize {
    usize::try_from(RATE.frames_in(Duration::from_millis(millis))).expect("fits")
}

fn capture() -> (Capture, FakeMicrophone) {
    let (microphone, mic) = Microphone::fake(RATE);
    let capture = microphone
        .capture(CaptureOptions::default())
        .expect("start the fake microphone");
    (capture, mic)
}

/// A fake that hears speech from 0.1 s to 0.5 s of its input, and
/// confirms that nobody spoke since, through 1 s.
fn one_utterance() -> AsrEngine {
    AsrEngine::new(
        FakeAsr::new(
            Script::new()
                .then(
                    Trigger::AfterSamples(1_600),
                    Step::started(Duration::from_millis(100)),
                )
                .then(Trigger::AfterSamples(8_000), Step::Segment(0, "hello"))
                .then(
                    Trigger::AfterSamples(8_000),
                    Step::ended(Duration::from_millis(500), 0),
                )
                .then(
                    Trigger::AfterSamples(16_000),
                    Step::known(Duration::from_secs(1)),
                ),
        )
        .reporting_activity(),
    )
}

#[test]
fn a_session_that_ends_itself_ends_the_listening_at_its_cutoff() {
    let (capture, mic) = capture();
    mic.push(&ramp(0, ms(200)));
    assert!(eventually(SETTLE, || capture.position() >= Duration::from_millis(200)));
    let options = AsrOptions::default().with_end_after_silence(Duration::from_millis(300));
    let listening = capture
        .listen(
            &one_utterance(),
            options,
            ListenOptions::starting_at(Duration::from_millis(200)),
        )
        .unwrap();
    // The listening is now the capture's only user.
    drop(capture);
    let mut updates = listening.updates();
    mic.push(&ramp(ms(200), ms(2_000)));

    let transcript = listening.wait(SETTLE).expect("it ended by itself").unwrap();
    assert_eq!(transcript.text(), "hello");
    assert_eq!(
        transcript.segments[0].end,
        Duration::from_millis(700),
        "capture time"
    );
    assert_eq!(transcript.duration, Duration::from_secs(1));
    assert_eq!(
        listening.end(),
        Some(Duration::from_millis(1_200)),
        "set once wait returns"
    );
    let started = updates.find_map(|update| match update {
        AsrUpdate::SpeechStarted { at } => Some(at),
        _ => None,
    });
    assert_eq!(started, Some(Duration::from_millis(300)));
    // Its end stopped the microphone.
    assert!(listening.level().abs() < f32::EPSILON);
    assert_eq!(
        listening.finish(Duration::from_secs(1)).unwrap(),
        transcript,
        "finish returns the same result"
    );
}

#[test]
fn a_listening_on_a_stopped_capture_is_closed() {
    let (capture, _mic) = capture();
    capture.stop();
    assert!(eventually(SETTLE, || {
        capture
            .listen(
                &one_utterance(),
                AsrOptions::default(),
                ListenOptions::default(),
            )
            .is_err_and(|error| matches!(error, speechkit::SpeechError::Closed))
    }));
}

#[test]
fn a_stop_while_recognition_is_blocked_ends_at_the_cutoff() {
    let (capture, mic) = capture();
    // The model is stuck in its first 100 ms while a second of audio queues
    // behind it.
    let gate = Gate::new();
    let engine = AsrEngine::new(FakeAsr::new(Script::new().then(
        Trigger::AfterSamples(1_600),
        Step::BlockUntilReleased(gate.clone()),
    )));
    let options = AsrOptions::default().with_max_length(Duration::from_millis(100));
    let listening = capture
        .listen(
            &engine,
            options,
            ListenOptions::default().with_recording(Duration::from_secs(10)),
        )
        .unwrap();
    mic.push(&ramp(0, ms(1_000)));
    assert!(eventually(SETTLE, || capture.position() >= Duration::from_secs(1)));
    assert!(gate.wait_entered(1, SETTLE));

    listening.stop();
    // The listening has read everything up to the stop, but the session has
    // not reached its cutoff, so nothing is known about the end yet.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(listening.end(), None);
    gate.release();

    let transcript = listening.finish(SETTLE).unwrap();
    assert_eq!(transcript.duration, Duration::from_millis(100));
    assert_eq!(
        listening.end(),
        Some(Duration::from_millis(100)),
        "the cutoff, not where the input stopped"
    );
    // The recording ends there too: a retry must not hear what the session
    // never did.
    let recording = listening.recording().expect("a recording");
    assert_eq!(recording.audio.samples, ramp(0, ms(100)));
    assert!(!recording.truncated);
}

#[test]
fn a_finish_that_times_out_still_ends_the_listening() {
    let (capture, mic) = capture();
    let gate = Gate::new();
    let engine = AsrEngine::new(FakeAsr::new(Script::new().then(
        Trigger::AfterSamples(1_600),
        Step::BlockUntilReleased(gate.clone()),
    )));
    let listening = capture
        .listen(&engine, AsrOptions::default(), ListenOptions::default())
        .unwrap();
    mic.push(&ramp(0, ms(200)));
    assert!(gate.wait_entered(1, SETTLE));

    let failure = listening
        .finish(Duration::from_millis(100))
        .expect_err("the session is stuck");
    assert!(matches!(
        failure.error,
        speechkit::SpeechError::DeadlineExceeded
    ));
    // The failed session wakes the listening's thread, which reports where
    // the input stopped.
    assert!(eventually(SETTLE, || listening.end().is_some()));
    gate.release();
}

#[test]
fn a_wait_that_returns_a_transcript_has_the_end() {
    // The session ends at exactly the audio it was given, so the listening's
    // thread is idle in a read and notices only within a tick.
    for _ in 0..20 {
        let (capture, mic) = capture();
        let options = AsrOptions::default().with_max_length(Duration::from_millis(100));
        let engine = AsrEngine::new(FakeAsr::hello_world());
        let listening = capture
            .listen(&engine, options, ListenOptions::default())
            .unwrap();
        drop(capture);
        mic.push(&ramp(0, ms(100)));
        let started = std::time::Instant::now();
        let transcript = loop {
            if let Some(result) = listening.wait(Duration::ZERO) {
                break result.unwrap();
            }
            assert!(started.elapsed() < SETTLE, "the session never ended");
            std::thread::sleep(Duration::from_micros(100));
        };
        assert_eq!(transcript.duration, Duration::from_millis(100));
        assert_eq!(
            listening.end(),
            Some(Duration::from_millis(100)),
            "set whenever wait returns the transcript"
        );
    }
}

#[test]
fn a_recording_holds_every_frame_up_to_the_end_at_any_rate() {
    // At 44.1 kHz a frame count does not survive a `Duration` rounded down:
    // 4411 frames last 100 022 675 ns, which hold 4410 whole frames.
    let rate = SampleRate::new(44_100).unwrap();
    let (microphone, mic) = Microphone::fake(rate);
    let capture = microphone.capture(CaptureOptions::default()).unwrap();
    let engine = AsrEngine::new(FakeAsr::hello_world());
    let listening = capture
        .listen(
            &engine,
            AsrOptions::default(),
            ListenOptions::default().with_recording(Duration::from_secs(10)),
        )
        .unwrap();
    mic.push(&ramp(0, 4_411));
    listening.stop();
    listening.finish(SETTLE).unwrap();

    let recording = listening.recording().expect("a recording");
    assert_eq!(recording.audio.samples.len(), 4_411);
    assert_eq!(recording.audio.samples, ramp(0, 4_411));
}

/// The next keyword `watching` hears.
fn heard(watching: &mut Watching) -> Wake {
    match watching.recv(SETTLE).expect("an update") {
        WakeUpdate::Heard(wake) => Some(wake),
        _ => None,
    }
    .expect("a keyword")
}

#[test]
fn a_listening_from_a_keyword_takes_listen_options() {
    let (capture, mic) = capture();
    let mut watching = capture.watch(&FakeWakeWord::new()).expect("watch");
    let engine = AsrEngine::new(FakeAsr::new(Script::new()));
    let keyword = |from: usize| {
        let mut samples = ramp(from, ms(400));
        samples.push(FakeWakeWord::MARK);
        samples
    };

    // A start of its own is refused: the listening starts at the keyword.
    mic.push(&keyword(0));
    mic.push(&ramp(0, ms(400)));
    let error = heard(&mut watching)
        .listen_with(
            &engine,
            AsrOptions::default(),
            ListenOptions::starting_at(Duration::ZERO),
        )
        .expect_err("a start of its own");
    assert!(matches!(error, SpeechError::InvalidInput(_)), "{error:?}");

    mic.push(&keyword(0));
    mic.push(&ramp(0, ms(400)));
    let listening = heard(&mut watching)
        .listen_with(
            &engine,
            AsrOptions::default(),
            ListenOptions::default().with_recording(Duration::from_secs(5)),
        )
        .expect("listen after the keyword");
    let recording = listening.recording().expect("it records");
    assert!(!recording.truncated);
    listening.finish(SETTLE).expect("the listening finishes");
    watching.stop();
}
