//! The scenarios of the public API design, compiled and never run.
//!
//! Each function below is a scenario written the way an application would
//! write it, against the real API. If a call changes shape, this file stops
//! compiling: the API and its documented usage cannot drift apart. Nothing
//! here runs, so no device, model, key, or network is needed.
//!
//! The scenario code is as written in the design, except where an
//! application object is borrowed here and owned there (`&engine` is
//! `engine`). The types at the top are the application's own code, stubbed.
#![cfg(all(
    feature = "devices",
    feature = "openai",
    feature = "server",
    feature = "sherpa"
))]
#![expect(dead_code, reason = "the scenarios are compiled and never run")]
#![expect(
    clippy::unused_self,
    clippy::unnecessary_wraps,
    reason = "the stubs stand in for application code"
)]

use std::{
    error::Error,
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
    time::Duration,
};

use speechkit::{
    SampleRate, Secret, SpeechError,
    asr::{AsrEngine, AsrOptions, AsrUpdate, LiveTranscript, Transcript},
    audio::{self, DecodeLimits},
    cloud::{CloudRuntime, OpenAiRealtime, OpenAiRealtimeConfig},
    io::{CaptureOptions, ListenOptions, Microphone, Playback, Speaker, WakeUpdate},
    server::Server,
    sherpa,
    tts::{TtsEngine, TtsOptions, TtsUpdate},
    wake::WakeWordModel,
};

// ---- The application's own code, stubbed.

struct Ui;

impl Ui {
    fn contact_names(&self) -> Vec<String> {
        Vec::new()
    }
    fn key_is_up(&self) -> bool {
        true
    }
    fn wait_for_key_up(&self) {}
    fn skip_pressed(&self) -> bool {
        false
    }
    fn show_candidates(&self, _text: &str) {}
    fn set_meter(&self, _level: f32) {}
    fn insert(&self, _text: &str) {}
    fn offer(&self, _what: &str) {}
    fn enable_gpu(&self, _on: bool) {}
    fn add_microphone(&self, _name: &str, _is_default: bool) {}
}

struct Panel;

impl Panel {
    fn show(&self, _text: &str) {}
}

struct Overlay;

impl Overlay {
    fn show(&self, _text: &str) {}
}

struct Rtp;

impl Rtp {
    fn audio(&self) -> Vec<&'static [f32]> {
        Vec::new()
    }
}

struct Brain;

impl Brain {
    fn answer(&self, _said: &str) -> String {
        String::new()
    }
}

struct LlmCancel;

impl LlmCancel {
    fn cancel(&self) {}
}

#[derive(Default)]
struct Conversation {
    lines: Vec<String>,
}

impl Conversation {
    fn user(&mut self, said: &str) {
        self.lines.push(said.to_owned());
    }
    fn assistant(&mut self, said: &str) {
        self.lines.push(said.to_owned());
    }
}

struct Llm;

impl Llm {
    fn stream(&self, _prompt: &str) -> Vec<String> {
        Vec::new()
    }
    fn reply(&self, _chat: &Conversation) -> (Box<dyn Iterator<Item = String> + Send>, LlmCancel) {
        (Box::new(std::iter::empty()), LlmCancel)
    }
}

fn clock(_time: Duration) -> String {
    String::new()
}
fn save(_confirmed: &Transcript) {}
fn notify(_error: &SpeechError) {}
fn prompt_replug() {}
fn write_srt(_start: Duration, _end: Duration, _text: &str) -> Result<(), Box<dyn Error>> {
    Ok(())
}
fn with_gain(samples: &[f32], _gain: f32) -> Vec<f32> {
    samples.to_vec()
}
fn report(_path: &Path, _result: Result<Transcript, Box<dyn Error + Send + Sync>>) {}

// ---- 1. Dictation input method

fn dictation_push_to_talk(
    mic: &Microphone,
    engine: &AsrEngine,
    ui: &Ui,
) -> Result<(), Box<dyn Error>> {
    let listening = mic.listen(engine, AsrOptions::default().with_hints(ui.contact_names()))?;
    let mut updates = listening.updates();
    let mut live = LiveTranscript::new();
    while !ui.key_is_up() {
        while let Ok(update) = updates.try_recv() {
            live.apply(&update);
        }
        ui.show_candidates(&live.text());
        ui.set_meter(listening.level());
    }
    listening.stop();
    let transcript = listening.finish(Duration::from_secs(5))?;
    ui.insert(&transcript.text());
    Ok(())
}

fn dictation_hands_free(
    mic: &Microphone,
    engine: &AsrEngine,
    ui: &Ui,
) -> Result<(), Box<dyn Error>> {
    let options = AsrOptions::default()
        .with_end_after_silence(Duration::from_millis(800))
        .with_no_speech_timeout(Duration::from_secs(5));
    let listening = mic.listen(engine, options)?;
    let mut live = LiveTranscript::new();
    for update in listening.updates() {
        live.apply(&update);
        ui.show_candidates(&live.text());
    }
    let transcript = listening.finish(Duration::from_secs(1))?;
    ui.insert(&transcript.text());
    Ok(())
}

fn dictation_with_a_cloud_backend(
    mic: &Microphone,
    cloud: &AsrEngine,
    backup: &AsrEngine,
    ui: &Ui,
) -> Result<(), Box<dyn Error>> {
    let capture = mic.capture(CaptureOptions::default())?;

    let from = capture
        .position()
        .saturating_sub(Duration::from_millis(300));
    let listen = ListenOptions::starting_at(from).with_recording(Duration::from_secs(60));
    let listening = capture.listen(cloud, AsrOptions::default(), listen)?;

    ui.wait_for_key_up();
    listening.stop();

    let text = match listening.finish(Duration::from_secs(10)) {
        Ok(transcript) => transcript.text(),
        Err(failure) if failure.error.retryable() => {
            let recording = listening.recording().ok_or("no recording")?;
            backup
                .transcribe(
                    &recording.audio,
                    AsrOptions::default(),
                    Duration::from_secs(30),
                )?
                .text()
        }
        Err(failure) => return Err(failure.into()),
    };
    ui.insert(&text);
    Ok(())
}

// ---- 2. Meeting transcription

fn meeting_transcription(
    mic: &Microphone,
    minutes: &mut impl Write,
    panel: &Panel,
) -> Result<(), Box<dyn Error>> {
    let asr = sherpa::AsrConfig::offline("sense-voice", "silero_vad.onnx").load()?;
    let punctuation = sherpa::PunctuationConfig::new("punct-ct-transformer").load()?;
    let engine = AsrEngine::new(asr).with_post_processor(punctuation);

    let listening = mic.listen(&engine, AsrOptions::default().with_language("zh"))?;
    let mut live = LiveTranscript::new();
    for update in listening.updates() {
        live.apply(&update);
        match &update {
            AsrUpdate::Segment(segment) => {
                writeln!(minutes, "[{}] {}", clock(segment.start), segment.text)?;
            }
            AsrUpdate::Closed(Err(failure)) => {
                save(&failure.confirmed);
                notify(&failure.error);
            }
            _ => panel.show(&live.text()),
        }
        if listening.device_lost() {
            prompt_replug();
        }
    }
    Ok(())
}

// ---- 3. Transcribe a file

fn transcribe_a_file() -> Result<(), Box<dyn Error>> {
    let engine =
        AsrEngine::new(sherpa::AsrConfig::offline("sense-voice", "silero_vad.onnx").load()?);
    let recording = audio::read("meeting.mp3", DecodeLimits::new(Duration::from_secs(7200)))?;
    let transcript =
        engine.transcribe(&recording, AsrOptions::default(), Duration::from_secs(600))?;
    for segment in &transcript.segments {
        write_srt(segment.start, segment.end, &segment.text)?;
    }
    Ok(())
}

// ---- 4. ASR service

async fn asr_service(engine: AsrEngine) -> Result<(), Box<dyn Error>> {
    Server::new()
        .with_asr(engine, "sense-voice")
        .with_bind("0.0.0.0:8080")
        .with_auth(Secret::from_env("SPEECHKIT_TOKEN")?)
        .run()
        .await?;
    Ok(())
}

// ---- 5. Text to speech

fn speak_to_a_file(tts: &TtsEngine) -> Result<(), Box<dyn Error>> {
    let options = TtsOptions::default().with_voice("45").with_speed(1.1);
    let speech = tts.synthesize("Hello from speechkit.", options, Duration::from_secs(60))?;
    std::fs::write("out.wav", audio::encode_wav(&speech)?)?;
    Ok(())
}

fn speak_and_stop(speaker: &Speaker, tts: &TtsEngine, ui: &Ui) -> Result<(), Box<dyn Error>> {
    let playback = speaker.speak(tts, "Done.", TtsOptions::default())?;
    if ui.skip_pressed() {
        playback.stop();
    }
    playback.finish(Duration::from_secs(30))?;
    Ok(())
}

fn speak_a_reply_as_it_is_written(
    speaker: &Speaker,
    tts: &TtsEngine,
    llm: &Llm,
) -> Result<(), Box<dyn Error>> {
    let (session, playback) = speaker.speak_streaming(tts, TtsOptions::default())?;
    for piece in llm.stream("Tell me a story.") {
        session.push_text(&piece)?;
    }
    session.close_text();
    playback.finish(Duration::from_secs(300))?;
    Ok(())
}

fn speak_through_your_own_processing(
    speaker: &Speaker,
    tts: &TtsEngine,
) -> Result<(), Box<dyn Error>> {
    let (session, output) = tts.start(TtsOptions::default(), Duration::from_secs(5))?;
    session.push_text("Processed on its way to the speaker.")?;
    session.close_text();
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
}

fn play_a_sound_file(speaker: &Speaker) -> Result<(), Box<dyn Error>> {
    speaker
        .play(audio::read("chime.wav", DecodeLimits::default())?)?
        .finish(Duration::from_secs(5))?;
    Ok(())
}

// ---- 6. TTS service

async fn tts_service(tts: TtsEngine) -> Result<(), Box<dyn Error>> {
    Server::new()
        .with_tts(tts, "kokoro")
        .with_bind("0.0.0.0:8080")
        .with_auth(Secret::from_env("SPEECHKIT_TOKEN")?)
        .run()
        .await?;
    Ok(())
}

// ---- 7. Voice assistant

fn voice_assistant(
    mic: &Microphone,
    kws: &impl WakeWordModel,
    asr: &AsrEngine,
    tts: &TtsEngine,
    brain: &Brain,
) -> Result<(), Box<dyn Error>> {
    let capture = mic.capture(CaptureOptions::default())?;
    let speaker = Speaker::open_default()?;
    let request = AsrOptions::default()
        .with_end_after_silence(Duration::from_secs(1))
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
            continue;
        }
        speaker.stop();
        let listening = match wake.listen(asr, request.clone()) {
            Ok(listening) => listening,
            Err(SpeechError::Capacity) => continue,
            Err(error) => return Err(error.into()),
        };
        let heard = listening
            .wait(Duration::from_secs(30))
            .unwrap_or_else(|| listening.finish(Duration::from_secs(5)))?;
        handled_until = listening.end().unwrap_or(keyword_end);
        if heard.segments.is_empty() {
            continue;
        }
        let reply = brain.answer(&heard.text());
        speaker.speak(tts, &reply, TtsOptions::default())?;
    }
    Ok(())
}

// ---- 8. Live captions

fn live_captions(asr: &AsrEngine, rtp: &Rtp, overlay: &Overlay) -> Result<(), Box<dyn Error>> {
    let session = asr.start(
        SampleRate::HZ_48000,
        AsrOptions::default(),
        Duration::from_secs(5),
    )?;
    let mut updates = session.updates();
    let mut live = LiveTranscript::new();
    for packet in rtp.audio() {
        session.push(packet, Duration::from_secs(2))?;
        while let Ok(update) = updates.try_recv() {
            live.apply(&update);
        }
        overlay.show(&live.text());
    }
    session.finish(Duration::from_secs(10))?;
    Ok(())
}

// ---- 9. Batch transcription pipeline

fn batch_pipeline(backend: impl speechkit::asr::AsrBackend, files: &[PathBuf]) {
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
}

fn transcribe_file(
    engine: &AsrEngine,
    path: &Path,
) -> Result<Transcript, Box<dyn Error + Send + Sync>> {
    let recording = audio::read(path, DecodeLimits::default())?;
    Ok(engine.transcribe(&recording, AsrOptions::default(), Duration::from_secs(1800))?)
}

// ---- 10. Settings UI

fn settings_ui(model_dir: &Path, ui: &Ui) -> Result<(), Box<dyn Error>> {
    let found = sherpa::inspect(model_dir)?;
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
    Ok(())
}

// ---- 11. Voice agent: an LLM you can interrupt

fn voice_agent(mic: &Microphone, tts: &TtsEngine, llm: &Llm) -> Result<(), Box<dyn Error>> {
    let asr = AsrEngine::new(OpenAiRealtime::new(
        OpenAiRealtimeConfig::new("gpt-4o-transcribe", Secret::from_env("OPENAI_API_KEY")?),
        CloudRuntime::owned(2)?,
    )?);
    let speaker = Speaker::open_default()?;
    let options = AsrOptions::default().with_turn_end(Duration::from_millis(700));
    let listening = mic.listen(&asr, options)?;
    let mut chat = Conversation::default();
    let mut unanswered: Vec<String> = Vec::new();
    let mut newest_speech = Duration::ZERO;
    let mut reply: Option<(Playback, LlmCancel)> = None;
    for update in listening.updates() {
        match update {
            AsrUpdate::SpeechStarted { at } => {
                newest_speech = at;
                retire(&mut reply, &mut chat);
            }
            AsrUpdate::TurnEnded(turn) => {
                let said = turn.text();
                if !said.is_empty() {
                    unanswered.push(said);
                }
                if newest_speech > turn.end || unanswered.is_empty() {
                    continue;
                }
                retire(&mut reply, &mut chat);
                chat.user(&std::mem::take(&mut unanswered).join(" "));
                let (session, playback) = speaker.speak_streaming(tts, TtsOptions::default())?;
                let (tokens, cancel) = llm.reply(&chat);
                std::thread::spawn(move || {
                    for piece in tokens {
                        if session.push_text(&piece).is_err() {
                            break;
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

fn retire(reply: &mut Option<(Playback, LlmCancel)>, chat: &mut Conversation) {
    if let Some((playback, cancel)) = reply.take() {
        playback.stop();
        cancel.cancel();
        chat.assistant(&playback.text_played());
    }
}

#[test]
fn the_scenarios_compile() {
    // Compiling this file is the test.
}
