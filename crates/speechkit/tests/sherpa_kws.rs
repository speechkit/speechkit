//! Keyword spotting against real models. The tests are ignored by default;
//! run them with `cargo test -p speechkit --features sherpa --test
//! sherpa_kws -- --ignored` after `cargo xtask fetch-fixtures`, which sets
//! `SPEECHKIT_MODEL_*`.
//!
//! The spelling checks against each model's `text2token` output are unit
//! tests in `sherpa/kws/tokenize.rs`.
//!
//! - The spotter finds keywords spoken in the models' test clips, at the
//!   right times, and again when they repeat.
#![cfg(feature = "sherpa")]
#![expect(clippy::unwrap_used, reason = "test helpers fail the calling test")]

use std::{path::Path, time::Duration};

use speechkit::{
    AudioBuffer,
    audio::{self, DecodeLimits},
    sherpa::{Keyword, KeywordSpotter, KeywordSpotterConfig},
    wake::{WakeEvent, WakeWordModel},
};
use speechkit_testkit::gates::model_dir;

/// Loading spells every phrase: English, Chinese, and both.
#[test]
#[ignore = "needs the kws-zh-en model"]
fn common_wake_phrases_load() {
    let Some(zh_en) = model_dir("kws-zh-en") else {
        return;
    };
    let config = KeywordSpotterConfig::new(&zh_en).with_keywords([
        "你好，小爱同学",
        "小艺小艺",
        "Hi Jarvis",
    ]);
    config.load().unwrap();
}

fn clip(path: &Path) -> AudioBuffer {
    audio::read(path, DecodeLimits::default()).unwrap()
}

/// The keywords detected in 16 kHz `samples`, fed in 100 ms chunks.
fn spot(kws: &KeywordSpotter, samples: &[f32]) -> Vec<WakeEvent> {
    let mut detector = kws.create().unwrap();
    let mut events = Vec::new();
    for chunk in samples.chunks(1600) {
        events.extend(detector.accept(chunk));
    }
    events.extend(detector.flush());
    let duration = kws.sample_rate().duration_of(samples.len() as u64);
    for event in &events {
        assert!(
            event.start <= event.end && event.end <= duration,
            "{event:?} in {duration:?} of audio"
        );
    }
    events
}

fn keywords(events: &[WakeEvent]) -> Vec<&str> {
    events.iter().map(|e| e.keyword.as_str()).collect()
}

/// Whether `actual` is within 200 ms of `expected` seconds.
fn near(actual: Duration, expected: f64) -> bool {
    (actual.as_secs_f64() - expected).abs() < 0.2
}

#[test]
#[ignore = "needs the kws-en model"]
fn english_keywords_are_spotted() {
    let Some(dir) = model_dir("kws-en") else {
        return;
    };
    let kws = KeywordSpotterConfig::new(&dir)
        .with_keywords(["light up", "lovely child", "forever", "hey siri"])
        .load()
        .unwrap();
    let light = spot(&kws, &clip(&dir.join("test_wavs/0.wav")).samples);
    assert_eq!(keywords(&light), ["light up"]);
    assert!(near(light[0].start, 3.0), "{light:?}");
    let lovely = spot(&kws, &clip(&dir.join("test_wavs/1.wav")).samples);
    assert_eq!(keywords(&lovely), ["lovely child", "forever"]);
    // Times count from the start of the stream, also after a detection.
    assert!(near(lovely[1].start, 10.9), "{lovely:?}");
}

#[test]
#[ignore = "needs the kws-en model"]
fn a_repeated_keyword_is_spotted_each_time() {
    let Some(dir) = model_dir("kws-en") else {
        return;
    };
    let kws = KeywordSpotterConfig::new(&dir)
        .with_keywords(["light up"])
        .load()
        .unwrap();
    let once = clip(&dir.join("test_wavs/0.wav")).samples;
    let thrice = once.repeat(3);
    let events = spot(&kws, &thrice);
    assert_eq!(keywords(&events), ["light up"; 3]);
    let length = kws.sample_rate().duration_of(once.len() as u64);
    for pair in events.windows(2) {
        let gap = pair[1].start.checked_sub(pair[0].start).unwrap();
        assert!(near(gap, length.as_secs_f64()), "{events:?}");
    }
}

#[test]
#[ignore = "needs the kws-zh-en model"]
fn chinese_and_english_keywords_are_spotted_by_one_model() {
    let Some(dir) = model_dir("kws-zh-en") else {
        return;
    };
    let config = KeywordSpotterConfig::new(&dir).with_keywords([
        Keyword::new("Light up"),
        Keyword::new("文森特卡索"),
        Keyword::new("周望军"),
        Keyword::new("朱丽楠"),
        Keyword::new("蒋友伯"),
        Keyword::new("女儿"),
        Keyword::new("法国"),
        Keyword::new("见面会"),
        Keyword::new("落实"),
    ]);
    let kws = config.load().unwrap();
    let english = spot(&kws, &clip(&dir.join("test_wavs/en_0.wav")).samples);
    assert_eq!(keywords(&english), ["Light up"]);
    let mut found = Vec::new();
    for index in 0..7 {
        let path = dir.join(format!("test_wavs/zh_{index}.wav"));
        found.extend(spot(&kws, &clip(&path).samples));
    }
    assert!(found.len() >= 4, "{found:?}");
    assert!(
        keywords(&found).iter().all(|k| *k != "Light up"),
        "{found:?}"
    );
}

#[test]
#[ignore = "needs the kws-zh model"]
fn keywords_given_as_tokens_are_reported_by_name() {
    let Some(dir) = model_dir("kws-zh") else {
        return;
    };
    // `text2token` output: the tokens, then `@name`.
    let file = std::fs::read_to_string(dir.join("test_wavs/test_keywords.txt")).unwrap();
    let listed: Vec<Keyword> = file
        .lines()
        .filter_map(|line| line.split_once('@'))
        .map(|(tokens, name)| Keyword::new(name.trim()).with_tokens(tokens.trim()))
        .collect();
    let kws = KeywordSpotterConfig::new(&dir)
        .with_keywords(listed)
        .load()
        .unwrap();
    let mut found = Vec::new();
    for index in 0..7 {
        let path = dir.join(format!("test_wavs/{index}.wav"));
        found.extend(spot(&kws, &clip(&path).samples));
    }
    assert!(!found.is_empty());
    for keyword in keywords(&found) {
        assert!(file.contains(&format!("@{keyword}")), "{keyword}");
    }
}
