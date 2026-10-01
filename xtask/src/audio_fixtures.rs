//! `cargo xtask gen-audio-fixtures`: audio files in many formats, made with
//! ffmpeg from one 3 s, 440 Hz sine.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use crate::Result;

const SINE: &str = "sine=frequency=440:duration=3:sample_rate=16000";

fn ffmpeg(args: &[&str]) -> Result {
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .status()
        .map_err(|e| format!("cannot run ffmpeg: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("ffmpeg {} failed: {status}", args.join(" ")))
    }
}

fn path(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn concat(dir: &Path, out: &str, parts: &[&str]) -> Result {
    let mut bytes = Vec::new();
    for part in parts {
        bytes.extend(fs::read(dir.join(part)).map_err(|e| format!("{part}: {e}"))?);
    }
    fs::write(dir.join(out), bytes).map_err(|e| format!("{out}: {e}"))
}

/// Encodes Ogg Vorbis with libvorbis, falling back to ffmpeg's native
/// encoder (stereo only) when libvorbis is missing.
fn vorbis(source: &str, out: &str) -> Result {
    ffmpeg(&["-i", source, "-c:a", "libvorbis", out]).or_else(|_| {
        ffmpeg(&[
            "-i",
            source,
            "-ac",
            "2",
            "-c:a",
            "vorbis",
            "-strict",
            "experimental",
            out,
        ])
    })
}

/// Rewrites every MPEG audio frame's payload with 0xFF, keeping the frame
/// headers, so the file parses but no packet decodes.
fn corrupt_mp3_payloads(bytes: &mut [u8]) -> Result {
    const MPEG1: [u32; 15] = [
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    const MPEG2: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let mut offset = 0;
    let mut frames = 0;
    while offset + 4 <= bytes.len() {
        let header = &bytes[offset..offset + 4];
        if header[0] != 0xFF || header[1] & 0xE0 != 0xE0 {
            return Err(format!("no MPEG frame header at byte {offset}"));
        }
        let mpeg1 = (header[1] >> 3) & 3 == 3;
        let bitrate = if mpeg1 { MPEG1 } else { MPEG2 }[usize::from(header[2] >> 4) % 15] * 1000;
        let rate = if mpeg1 {
            [44_100, 48_000, 32_000]
        } else {
            [22_050, 24_000, 16_000]
        }[usize::from((header[2] >> 2) & 3) % 3];
        let padding = u32::from((header[2] >> 1) & 1);
        let length = (if mpeg1 { 144 } else { 72 }) * bitrate / rate + padding;
        let length = usize::try_from(length).map_err(|e| e.to_string())?;
        if length <= 4 {
            return Err(format!("bad frame length at byte {offset}"));
        }
        let end = (offset + length).min(bytes.len());
        bytes[offset + 4..end].fill(0xFF);
        offset = end;
        frames += 1;
    }
    if frames == 0 {
        return Err("no MPEG frames found".into());
    }
    Ok(())
}

/// Generates every fixture into `dir`.
pub(crate) fn generate(dir: &Path) -> Result {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    encode_formats(dir)?;
    concatenate(dir)?;
    extra_tracks(dir)?;
    damage(dir)
}

/// The source WAV and one file per supported format.
fn encode_formats(dir: &Path) -> Result {
    let p = |name: &str| path(dir, name);
    let source = p("source.wav");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        SINE,
        "-af",
        "volume=0.25",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        &source,
    ])?;
    let encodings: [(&[&str], &str); 7] = [
        (&["-c:a", "libmp3lame", "-b:a", "64k"], "speech.mp3"),
        (&["-c:a", "aac", "-b:a", "64k"], "speech.m4a"),
        (&["-c:a", "aac", "-b:a", "64k", "-f", "adts"], "speech.aac"),
        (&["-c:a", "flac"], "speech.flac"),
        (&["-c:a", "flac", "-f", "matroska"], "speech.mka"),
        (&["-c:a", "flac", "-f", "ogg"], "speech.oga"),
        (
            &[
                "-ac",
                "2",
                "-ar",
                "44100",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "128k",
            ],
            "speech_stereo.mp3",
        ),
    ];
    for (options, name) in encodings {
        let out = p(name);
        let mut args = vec!["-i", source.as_str()];
        args.extend_from_slice(options);
        args.push(&out);
        ffmpeg(&args)?;
    }
    vorbis(&source, &p("speech.ogg"))
}

/// Files made of two streams back to back.
fn concatenate(dir: &Path) -> Result {
    let stereo = dir.join("stereo_16k.tmp.mp3");
    ffmpeg(&[
        "-i",
        &path(dir, "source.wav"),
        "-ac",
        "2",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "64k",
        "-f",
        "mp3",
        &stereo.to_string_lossy(),
    ])?;
    concat(
        dir,
        "concat_channels.mp3",
        &["speech.mp3", "stereo_16k.tmp.mp3"],
    )?;
    fs::remove_file(&stereo).map_err(|e| e.to_string())?;
    concat(dir, "concat_same.mp3", &["speech.mp3", "speech.mp3"])?;
    concat(dir, "chained.ogg", &["speech.ogg", "speech.ogg"])
}

/// A video track next to the audio, in Matroska and MP4, and an Opus
/// stream.
fn extra_tracks(dir: &Path) -> Result {
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        SINE,
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10:duration=3",
        "-af",
        "volume=0.25",
        "-ac",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-c:v",
        "mpeg4",
        "-shortest",
        &path(dir, "av.mp4"),
    ])?;
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        SINE,
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=64x48:rate=10:duration=3",
        "-af",
        "volume=0.25",
        "-ac",
        "1",
        "-c:a",
        "flac",
        "-c:v",
        "mpeg4",
        "-shortest",
        &path(dir, "av.mkv"),
    ])?;
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=3:sample_rate=48000",
        "-af",
        "volume=0.25",
        "-ac",
        "1",
        "-c:a",
        "libopus",
        "-b:a",
        "24k",
        &path(dir, "tone.webm"),
    ])
}

/// Truncated, corrupted, and undecodable MP3s.
fn damage(dir: &Path) -> Result {
    let mp3 = fs::read(dir.join("speech.mp3")).map_err(|e| e.to_string())?;
    fs::write(dir.join("truncated.mp3"), &mp3[..mp3.len() * 2 / 3]).map_err(|e| e.to_string())?;
    let mut flipped = mp3;
    let at = flipped.len() / 3;
    for byte in &mut flipped[at..at + 64] {
        *byte ^= 0xFF;
    }
    fs::write(dir.join("corrupt.mp3"), flipped).map_err(|e| e.to_string())?;
    let zero = dir.join("zero.tmp.mp3");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.05:sample_rate=16000",
        "-ac",
        "1",
        "-c:a",
        "libmp3lame",
        "-write_xing",
        "0",
        "-id3v2_version",
        "0",
        "-f",
        "mp3",
        &zero.to_string_lossy(),
    ])?;
    let mut bytes = fs::read(&zero).map_err(|e| e.to_string())?;
    corrupt_mp3_payloads(&mut bytes)?;
    fs::write(dir.join("zero_samples.mp3"), bytes).map_err(|e| e.to_string())?;
    fs::remove_file(zero).map_err(|e| e.to_string())
}

/// The default output directory: `fixtures/audio` in the workspace.
pub(crate) fn default_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/audio")
}
