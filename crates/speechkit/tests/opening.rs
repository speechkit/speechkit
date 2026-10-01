//! Deadlines include initialization and retain slots until it finishes.

use speechkit::{
    AudioBuffer, SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions},
    tts::{TtsEngine, TtsOptions},
};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    eventually,
    tts::{FakeTts, TtsStep, TtsTrigger},
};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

#[test]
fn c05_transcribe_deadline_includes_open_and_keeps_slot() {
    let gate = Gate::new();
    let fake =
        FakeAsr::new(Script::new().then(Trigger::OnOpen, Step::BlockUntilReleased(gate.clone())));
    let engine = AsrEngine::new(fake).with_max_sessions(1);
    let copy = engine.clone();
    let (sender, receiver) = mpsc::channel();
    let task = std::thread::spawn(move || {
        let spec = SampleRate::HZ_16000;
        let result = copy.transcribe(
            &AudioBuffer::new(spec, vec![0.0; 160]),
            AsrOptions::default(),
            Duration::from_millis(300),
        );
        sender.send(result).unwrap();
    });
    assert!(gate.wait_entered(1, Duration::from_secs(5)));
    let result = receiver.recv_timeout(Duration::from_secs(2));
    let active = engine.active_sessions();
    gate.release();
    task.join().unwrap();
    assert!(matches!(
        result.unwrap().unwrap_err().error,
        SpeechError::DeadlineExceeded
    ));
    assert_eq!(active, 1);
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
}

#[test]
fn t05_synthesis_deadline_includes_open_and_keeps_slot() {
    let gate = Gate::new();
    let fake = FakeTts::new(vec![(
        TtsTrigger::OnOpen,
        TtsStep::BlockUntilReleased(gate.clone()),
    )]);
    let engine = TtsEngine::new(fake).with_max_sessions(1);
    let copy = engine.clone();
    let (sender, receiver) = mpsc::channel();
    let task = std::thread::spawn(move || {
        sender
            .send(copy.synthesize("hello", TtsOptions::default(), Duration::from_millis(300)))
            .unwrap();
    });
    assert!(gate.wait_entered(1, Duration::from_secs(5)));
    let result = receiver.recv_timeout(Duration::from_secs(2));
    let active = engine.active_sessions();
    gate.release();
    task.join().unwrap();
    assert!(matches!(
        result.unwrap().unwrap_err().error,
        SpeechError::DeadlineExceeded
    ));
    assert_eq!(active, 1);
    assert!(eventually(Duration::from_secs(5), || engine
        .active_sessions()
        == 0));
}

#[test]
fn expired_open_never_calls_backend() {
    let fake = FakeTts::plain();
    let stats = fake.stats();
    let engine = TtsEngine::new(fake).with_max_sessions(1);
    assert!(matches!(
        engine.start(TtsOptions::default(), Instant::now()),
        Err(SpeechError::DeadlineExceeded)
    ));
    assert_eq!(stats.opened(), 0);
    assert_eq!(engine.active_sessions(), 0);
}

#[test]
fn t08_marks_cover_resampled_audio() {
    for rate in [
        SampleRate::HZ_8000,
        SampleRate::HZ_24000,
        SampleRate::HZ_44100,
        SampleRate::HZ_48000,
    ] {
        let engine = TtsEngine::new(FakeTts::plain()).with_max_sessions(1);
        let (session, mut output) = engine
            .start(
                TtsOptions::default().with_sample_rate(rate),
                Duration::from_secs(10),
            )
            .unwrap();
        session.push_text("hi。世界。").unwrap();
        session.close_text();
        let read = speechkit_testkit::contract::tts::read_all(&mut output);
        let summary = read.result.unwrap().unwrap();
        let marks = &summary.marks;
        assert_eq!(marks.len(), 2);
        assert_eq!(marks[0].audio.start, Duration::ZERO);
        assert_eq!(marks[0].audio.end, Duration::from_millis(30));
        assert_eq!(marks[1].audio.start, marks[0].audio.end);
        assert_eq!(marks[1].audio.end, summary.duration);
        assert_eq!(marks[1].audio.end, Duration::from_millis(60));
        // Each mark comes after all of its audio, resampler tail included.
        for (mark, before) in &read.marks {
            assert!(rate.frames_in(mark.audio.end) <= *before as u64, "{rate:?}");
        }
    }
}
