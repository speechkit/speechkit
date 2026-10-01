//! Latency with real models. Ignored by default; the
//! numbers are printed.
//! The assertions are loose ceilings that only catch something broken.
#![cfg(feature = "sherpa")]

use std::time::{Duration, Instant};

use speechkit::sherpa::{AsrConfig, TtsConfig};
use speechkit::{
    asr::{AsrEngine, AsrOptions, AsrUpdate},
    audio::{self, DecodeLimits},
    tts::{TtsEngine, TtsOptions, TtsUpdate},
};
use speechkit_testkit::{gates::model_dir, secs};

#[test]
#[ignore = "needs models; run cargo xtask fetch-fixtures"]
fn time_to_first_partial() {
    let Some(dir) = model_dir("streaming-en") else {
        return;
    };
    let engine = AsrEngine::new(AsrConfig::streaming(&dir).load().unwrap());
    let audio = audio::read(dir.join("test_wavs/0.wav"), DecodeLimits::default()).unwrap();
    let session = engine
        .start(
            audio.sample_rate,
            AsrOptions::default(),
            Duration::from_secs(10),
        )
        .unwrap();
    let mut observer = session.updates();
    let chunk = usize::try_from(audio.sample_rate.frames_in(Duration::from_millis(100))).unwrap();
    let started = Instant::now();
    let latency = std::thread::scope(|scope| {
        scope.spawn(|| {
            // Real time: one 100 ms chunk every 100 ms.
            for (i, piece) in audio.samples.chunks(chunk).enumerate() {
                let due = started + Duration::from_millis(100 * u64::try_from(i).unwrap());
                std::thread::sleep(due.saturating_duration_since(Instant::now()));
                session.push(piece.to_vec(), secs(10)).unwrap();
            }
            session.close_input();
        });
        loop {
            match observer.recv(secs(30)).unwrap() {
                AsrUpdate::Partial(partial) if !partial.text.trim().is_empty() => {
                    break started.elapsed();
                }
                AsrUpdate::Closed(_) => panic!("no partial result"),
                _ => {}
            }
        }
    });
    let _ = session.finish(secs(30));
    eprintln!("time to first partial: {latency:?}");
    assert!(latency < Duration::from_secs(5), "{latency:?}");
}

#[test]
#[ignore = "needs a TTS model; set SPEECHKIT_MODEL_TTS_PIPER_EN"]
fn time_to_first_audio() {
    let Some(dir) = model_dir("tts-piper-en") else {
        return;
    };
    let engine = TtsEngine::new(TtsConfig::new(dir).load().unwrap());
    // Load and warm up, so the measurement is synthesis only.
    let _ = engine.synthesize("Warm up.", TtsOptions::default(), secs(60));
    let started = Instant::now();
    let (session, mut output) = engine
        .start(TtsOptions::default(), Duration::from_secs(10))
        .unwrap();
    session
        .push_text("The first sentence is short. The second one takes a little longer to say.")
        .unwrap();
    session.close_text();
    let first = output.recv(secs(60)).unwrap();
    let latency = started.elapsed();
    assert!(matches!(first, TtsUpdate::Audio(audio) if !audio.is_empty()));
    eprintln!("time to first audio: {latency:?}");
    assert!(latency < Duration::from_secs(5), "{latency:?}");
}
