//! Scenarios 5d, 7, and 11 of the API design, on the fake devices with
//! fake backends and a stand-in LLM. Each
//! scenario's code follows the design; the test around it plays the user
//! and the devices.
#![cfg(feature = "devices")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{
    error::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use speechkit::{
    SampleRate, SpeechError,
    asr::{AsrEngine, AsrOptions, AsrUpdate},
    io::{
        Capture, CaptureOptions, FakeMicrophone, FakeSpeaker, Microphone, Playback, Speaker,
        WakeUpdate,
    },
    tts::{TtsEngine, TtsOptions, TtsUpdate},
};
use speechkit_testkit::{
    asr::{FakeAsr, Script, Step, Trigger},
    contract::devices::ramp,
    eventually,
    tts::{FakeTts, level, samples_for},
    wake::FakeWakeWord,
};

type Failure = Box<dyn Error + Send + Sync>;

const RATE: SampleRate = SampleRate::HZ_16000;
const SETTLE: Duration = Duration::from_secs(10);

fn frames(millis: u64) -> usize {
    usize::try_from(RATE.frames_in(Duration::from_millis(millis))).unwrap()
}

/// Delivers `samples` 100 ms at a time, each once the capture has moved
/// the one before.
fn deliver(capture: &Capture, mic: &FakeMicrophone, samples: &[f32]) {
    for piece in samples.chunks(frames(100)) {
        let at = capture.position() + RATE.duration_of(piece.len() as u64);
        mic.push(piece);
        assert!(eventually(SETTLE, || capture.position() >= at));
    }
}

/// Runs `.0` when dropped, so a failing check still ends the scenario's
/// loop instead of leaving `thread::scope` waiting for it.
struct OnDrop<F: FnMut()>(F);

impl<F: FnMut()> Drop for OnDrop<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}

/// Samples the fake speaker played that were not silence.
fn audible(samples: &[f32]) -> Vec<f32> {
    samples
        .iter()
        .copied()
        .filter(|sample| sample.abs() > 0.000_01)
        .collect()
}

/// Plays the fake speaker 10 ms at a time until `done` holds for what was
/// heard.
fn play_until(speaker: &FakeSpeaker, heard: &mut Vec<f32>, mut done: impl FnMut(&[f32]) -> bool) {
    let started = Instant::now();
    while !done(heard) {
        assert!(started.elapsed() < SETTLE, "the speaker never played that");
        heard.extend(audible(&speaker.play(Duration::from_millis(10))));
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// Scenario 5d: synthesis runs faster than real time and fills its output
/// before the loop starts reading it (T-02); the audio waits, and every
/// sample reaches the speaker through the application's processing.
#[test]
fn scenario_5d_own_processing_through_a_sink() {
    let (speaker, device) = Speaker::fake(RATE);
    let tts = TtsEngine::new(FakeTts::plain());
    let text = "Processed on its way to the speaker.";
    let with_gain = |samples: &[f32], gain: f32| -> Vec<f32> {
        samples.iter().map(|sample| sample * gain).collect()
    };

    let heard = Arc::new(Mutex::new(Vec::new()));
    let playing = Arc::new(AtomicBool::new(true));
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while playing.load(Ordering::Relaxed) {
                let samples = device.play(Duration::from_millis(10));
                heard.lock().unwrap().extend(audible(&samples));
                std::thread::sleep(Duration::from_millis(2));
            }
        });
        let run = || -> Result<(), Failure> {
            let (session, output) = tts.start(TtsOptions::default(), Duration::from_secs(5))?;
            session.push_text(text)?;
            session.close_text();
            // The synthesis finishes before anything reads it.
            std::thread::sleep(Duration::from_millis(100));
            let (sink, playback) = speaker.sink(output.sample_rate())?;
            for update in output {
                match update {
                    TtsUpdate::Audio(samples) => {
                        sink.push(&with_gain(&samples, 0.8), Duration::from_secs(2))?;
                    }
                    TtsUpdate::Closed(result) => {
                        result?;
                    }
                    _ => {}
                }
            }
            sink.close();
            playback.finish(Duration::from_secs(10))?;
            Ok(())
        };
        let result = run();
        playing.store(false, Ordering::Relaxed);
        result.unwrap();
    });
    let heard = heard.lock().unwrap();
    let expected = vec![level(0) * 0.8; samples_for(text)];
    assert!(
        *heard == expected,
        "the speaker played {} samples",
        heard.len()
    );
}

/// A request heard after each keyword: "what time is it", then a pause
/// that ends the session a second into it.
fn request_engine() -> AsrEngine {
    AsrEngine::new(
        FakeAsr::new(
            Script::new()
                .then(
                    Trigger::AfterSamples(1_600),
                    Step::started(Duration::from_millis(100)),
                )
                .then(
                    Trigger::AfterSamples(8_000),
                    Step::Segment(0, "what time is it"),
                )
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

/// What the assistant did, for the test to check.
#[derive(Default)]
struct Log {
    /// Where each request started, in capture time.
    origins: Vec<Duration>,
    replies: Vec<Playback>,
}

/// Scenario 7, as designed: wake word, one request, a reply the next wake
/// word interrupts.
fn voice_assistant(
    capture: &Capture,
    kws: &FakeWakeWord,
    asr: &AsrEngine,
    speaker: &Speaker,
    tts: &TtsEngine,
    log: &Mutex<Log>,
) -> Result<(), Failure> {
    let answer = |question: &str| format!("You asked: {question}.");
    let request = AsrOptions::default()
        .with_end_after_silence(Duration::from_millis(300))
        .with_no_speech_timeout(Duration::from_secs(5))
        .with_max_length(Duration::from_secs(20));
    let mut handled_until = Duration::ZERO;
    for update in capture.watch(kws)? {
        let wake = match update {
            WakeUpdate::Heard(wake) => wake,
            WakeUpdate::Closed(result) => return Ok(result?),
            _ => continue,
        };
        let keyword_end = wake.event().end;
        if keyword_end < handled_until {
            continue; // said inside a request already handled
        }
        speaker.stop(); // barge-in: the wake word stops a reply
        let listening = match wake.listen(asr, request.clone()) {
            Ok(listening) => listening,
            Err(SpeechError::Capacity) => continue,
            Err(error) => return Err(error.into()),
        };
        log.lock().unwrap().origins.push(listening.origin());
        let heard = listening
            .wait(Duration::from_secs(30))
            .unwrap_or_else(|| listening.finish(Duration::from_secs(5)))?;
        handled_until = listening.end().unwrap_or(keyword_end);
        if heard.segments.is_empty() {
            continue;
        }
        let reply = answer(&heard.text());
        let playback = speaker.speak(tts, &reply, TtsOptions::default())?;
        log.lock().unwrap().replies.push(playback);
    }
    Ok(())
}

/// Scenario 7: the request starts right after the keyword, a keyword said
/// inside a request is not a new one, and a wake word interrupts a reply.
#[test]
fn scenario_7_voice_assistant() {
    let (microphone, mic) = Microphone::fake(RATE);
    let capture = microphone.capture(CaptureOptions::default()).unwrap();
    let (speaker, device) = Speaker::fake(RATE);
    let tts = TtsEngine::new(FakeTts::plain());
    let asr = request_engine();
    let kws = FakeWakeWord::new();
    let log = Mutex::new(Log::default());

    std::thread::scope(|scope| {
        let assistant = scope.spawn(|| voice_assistant(&capture, &kws, &asr, &speaker, &tts, &log));
        let _end = OnDrop(|| capture.stop());

        // "Hey kit" ends at 1 s; the request follows, with the keyword
        // said again halfway through it.
        let mut audio = ramp(0, frames(1_000));
        audio.push(FakeWakeWord::MARK);
        let first_end = RATE.duration_of(audio.len() as u64);
        audio.extend(ramp(audio.len(), frames(500)));
        audio.push(FakeWakeWord::MARK);
        audio.extend(ramp(audio.len(), frames(1_500)));
        deliver(&capture, &mic, &audio);
        assert!(eventually(SETTLE, || log.lock().unwrap().replies.len() == 1));

        // The reply starts playing, and the keyword interrupts it.
        let mut heard = Vec::new();
        play_until(&device, &mut heard, |heard| heard.len() >= frames(100));
        let mut more = vec![FakeWakeWord::MARK];
        let second_end = capture.position() + RATE.duration_of(1);
        more.extend(ramp(audio.len() + 1, frames(1_500)));
        deliver(&capture, &mic, &more);
        assert!(eventually(SETTLE, || log.lock().unwrap().replies.len() == 2));

        let log = log.lock().unwrap();
        assert_eq!(
            log.origins,
            [first_end, second_end],
            "requests start at keyword ends"
        );
        let interrupted = &log.replies[0];
        assert!(interrupted.is_done(), "the wake word stopped the reply");
        assert!(interrupted.played() > Duration::ZERO);
        assert!(interrupted.played() <= RATE.duration_of(heard.len() as u64));
        assert_eq!(
            interrupted.text_played(),
            "",
            "no sentence was heard in full"
        );
        drop(log);

        capture.stop();
        assistant.join().unwrap().unwrap();
    });
}

/// A stand-in conversation.
#[derive(Default, Debug)]
struct Conversation {
    turns: Vec<(&'static str, String)>,
}

impl Conversation {
    fn user(&mut self, text: &str) {
        self.turns.push(("user", text.to_owned()));
    }

    fn assistant(&mut self, text: &str) {
        self.turns.push(("assistant", text.to_owned()));
    }
}

/// A stand-in LLM: each reply streams the tokens the test sends it, and
/// stalls until it does.
#[derive(Default)]
struct Llm {
    requests: Mutex<Vec<(String, mpsc::Sender<String>)>>,
}

struct Tokens {
    tokens: mpsc::Receiver<String>,
    cancelled: Arc<AtomicBool>,
}

impl Iterator for Tokens {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        loop {
            if self.cancelled.load(Ordering::Relaxed) {
                return None;
            }
            match self.tokens.recv_timeout(Duration::from_millis(5)) {
                Ok(token) => return Some(token),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
            }
        }
    }
}

struct LlmCancel(Arc<AtomicBool>);

impl LlmCancel {
    fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Llm {
    fn reply(&self, chat: &Conversation) -> (Tokens, LlmCancel) {
        let prompt = chat
            .turns
            .last()
            .map(|(_, text)| text.clone())
            .unwrap_or_default();
        let (send, tokens) = mpsc::channel();
        self.requests.lock().unwrap().push((prompt, send));
        let cancelled = Arc::new(AtomicBool::new(false));
        let tokens = Tokens {
            tokens,
            cancelled: cancelled.clone(),
        };
        (tokens, LlmCancel(cancelled))
    }

    fn prompts(&self) -> Vec<String> {
        let requests = self.requests.lock().unwrap();
        requests.iter().map(|(prompt, _)| prompt.clone()).collect()
    }
}

/// Scenario 11, as designed: a voice agent the user can interrupt.
fn voice_agent(
    mic: &Microphone,
    asr: &AsrEngine,
    speaker: &Speaker,
    tts: &TtsEngine,
    llm: &Llm,
    shared_chat: &Mutex<Conversation>,
) -> Result<(), Failure> {
    let options = AsrOptions::default().with_turn_end(Duration::from_millis(700));
    let listening = mic.listen(asr, options)?; // the whole conversation
    let mut unanswered: Vec<String> = Vec::new();
    let mut newest_speech = Duration::ZERO;
    let mut reply: Option<(Playback, LlmCancel)> = None;
    for update in listening.updates() {
        let mut chat = shared_chat.lock().unwrap();
        match update {
            AsrUpdate::SpeechStarted { at } => {
                newest_speech = at;
                retire(&mut reply, &mut chat); // barge-in
            }
            AsrUpdate::TurnEnded(turn) => {
                let said = turn.text();
                if !said.is_empty() {
                    unanswered.push(said);
                }
                if newest_speech > turn.end || unanswered.is_empty() {
                    continue; // the user went on: answer when they stop
                }
                retire(&mut reply, &mut chat);
                chat.user(&std::mem::take(&mut unanswered).join(" "));
                let (session, playback) = speaker.speak_streaming(tts, TtsOptions::default())?;
                let (tokens, cancel) = llm.reply(&chat);
                std::thread::spawn(move || {
                    for piece in tokens {
                        if session.push_text(&piece).is_err() {
                            break; // the reply was stopped
                        }
                    }
                    session.close_text();
                });
                reply = Some((playback, cancel));
            }
            AsrUpdate::Closed(result) => {
                retire(&mut reply, &mut chat);
                result?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// Stops a reply that may still be playing, and keeps only what was heard.
fn retire(reply: &mut Option<(Playback, LlmCancel)>, chat: &mut Conversation) {
    if let Some((playback, cancel)) = reply.take() {
        playback.stop();
        cancel.cancel();
        chat.assistant(playback.text_played().trim_end());
    }
}

/// The conversation's speech, in its session's time: "hello" (a turn),
/// the user talking over the reply, "tell me", whose text comes only after
/// the user started speaking again, and "a joke", whose text comes while
/// no audio is pushed.
fn conversation_engine() -> AsrEngine {
    let ms = Duration::from_millis;
    AsrEngine::new(
        FakeAsr::new(
            Script::new()
                .then(Trigger::AfterSamples(1_600), Step::started(ms(100)))
                .then(Trigger::AfterSamples(16_000), Step::Segment(0, "hello"))
                .then(Trigger::AfterSamples(16_000), Step::ended(ms(1_000), 0))
                .then(Trigger::AfterSamples(30_400), Step::known(ms(1_800)))
                .then(Trigger::AfterSamples(40_000), Step::started(ms(2_500)))
                .then(Trigger::AfterSamples(48_000), Step::ended(ms(3_000), 1))
                .then(Trigger::AfterSamples(60_800), Step::known(ms(3_800)))
                .then(Trigger::AfterSamples(62_400), Step::started(ms(3_900)))
                .then(Trigger::AfterSamples(64_000), Step::Segment(1, "tell me"))
                .then(Trigger::AfterSamples(72_000), Step::ended(ms(4_500), 2))
                .then(Trigger::AfterSamples(84_800), Step::known(ms(5_300)))
                .then(
                    Trigger::AfterSamples(84_800),
                    Step::SegmentLater(ms(300), 2, "a joke"),
                ),
        )
        .reporting_activity(),
    )
}

/// Scenario 11: the agent answers at the end of a turn; an interruption
/// stops the reply while the LLM stalls, and keeps only the sentence the
/// user heard (D-06); a turn whose text comes after newer speech is kept
/// and answered with the next (A-08), whose text arrives while no audio
/// is pushed (A-10).
#[test]
fn scenario_11_voice_agent() {
    let (microphone, mic) = Microphone::fake(RATE);
    let (speaker, device) = Speaker::fake(RATE);
    let tts = TtsEngine::new(FakeTts::plain());
    let asr = conversation_engine();
    let llm = Llm::default();
    let chat = Mutex::new(Conversation::default());

    std::thread::scope(|scope| {
        let agent = scope.spawn(|| voice_agent(&microphone, &asr, &speaker, &tts, &llm, &chat));
        let _end = OnDrop(|| mic.lose());
        let pushed = Arc::new(Mutex::new(0_usize));
        let push = |samples: usize| {
            let mut pushed = pushed.lock().unwrap();
            for piece in ramp(*pushed, samples).chunks(frames(100)) {
                mic.push(piece);
                std::thread::sleep(Duration::from_millis(5));
            }
            *pushed += samples;
        };
        // Audio before the agent's capture starts would be lost.
        assert!(eventually(SETTLE, || asr.active_sessions() == 1));

        // "hello", then silence: the turn ends and the LLM is asked.
        push(frames(2_000));
        assert!(
            eventually(SETTLE, || llm.prompts() == ["hello"]),
            "{:?}",
            llm.prompts()
        );
        let tokens = llm.requests.lock().unwrap()[0].1.clone();
        tokens.send("Hi there. ".into()).unwrap();
        tokens.send("How can I help you? ".into()).unwrap();
        tokens.send("I".into()).unwrap();
        // The LLM stalls here, so the second sentence is complete but the
        // third is not. The first sentence plays in full, and a
        // little of the second.
        let mut heard = Vec::new();
        let first = samples_for("Hi there.");
        play_until(&device, &mut heard, |heard| {
            heard.len() >= first + frames(50)
        });

        // The user talks over the reply: it stops at once, though the LLM
        // is still waiting for its next token.
        push(frames(600));
        assert!(
            eventually(SETTLE, || chat.lock().unwrap().turns.len() == 2),
            "{:?}",
            chat.lock().unwrap()
        );
        assert_eq!(
            chat.lock().unwrap().turns[1],
            ("assistant", "Hi there.".to_owned())
        );
        // The player skips what is left of the reply within its 5 ms tick;
        // a real device would play at most that much more.
        std::thread::sleep(Duration::from_millis(100));
        let after = device.play(Duration::from_millis(200));
        assert!(audible(&after).is_empty(), "the reply went on playing");

        // "tell me" comes after newer speech started; "a joke" comes while
        // no audio is pushed. The agent answers both at once.
        push(frames(2_800));
        assert!(
            eventually(SETTLE, || llm.prompts().len() == 2),
            "{:?}",
            llm.prompts()
        );
        assert_eq!(llm.prompts(), ["hello", "tell me a joke"]);
        drop(tokens);

        mic.lose();
        agent.join().unwrap().unwrap();
    });
    let chat = chat.into_inner().unwrap();
    let roles: Vec<_> = chat.turns.iter().map(|(role, _)| *role).collect();
    assert_eq!(
        roles,
        ["user", "assistant", "user", "assistant"],
        "{chat:?}"
    );
}
