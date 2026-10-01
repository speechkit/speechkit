//! Multi-format decoding with symphonia (feature `decode`).
//!
//! Decoding fails loudly when the stream changes under the decoder (a
//! chained Ogg, a codec reconfiguration, a second concatenated file),
//! because silently truncated audio is worse than an error. Local damage,
//! such as one bad packet or a truncated tail, only loses the affected
//! packets.

use symphonia::core::{
    codecs::audio::{AudioDecoder, AudioDecoderOptions},
    errors::Error as SymphoniaError,
    formats::{FormatOptions, FormatReader, TrackType, probe::Hint},
    io::{MediaSource, MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
};

use super::{DecodeLimits, SUPPORTED_FORMATS, invalid, to_session_range};
use crate::{AudioBuffer, SampleRate, SpeechError};

/// Consecutive packets that fail to decode or yield no audio before
/// decoding gives up.
const MAX_UNPRODUCTIVE_PACKETS: usize = 5;

/// An opened audio track, ready to decode.
struct Track {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn AudioDecoder>,
    id: u32,
    rate: SampleRate,
    channels: usize,
}

fn open(source: MediaSourceStream<'static>) -> Result<Track, SpeechError> {
    // symphonia 0.6 ignores the hint, so content sniffing is the only signal.
    let format = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| invalid(format_args!("{e}; supported formats: {SUPPORTED_FORMATS}")))?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| invalid("the file has no audio track"))?;
    let id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| invalid("the audio track has no codec parameters"))?;
    let rate = params
        .sample_rate
        .ok_or_else(|| invalid("the audio track has no sample rate"))?;
    let rate = SampleRate::new(rate)?;
    let channels = params
        .channels
        .as_ref()
        .map(symphonia::core::audio::Channels::count)
        .filter(|&count| count > 0)
        .ok_or_else(|| invalid("the audio track has no channels"))?;
    let decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| {
            invalid(format_args!(
                "unsupported codec ({e}). Opus is not supported; transcode to WAV, FLAC, or MP3 \
                 first, for example `ffmpeg -i input.webm output.wav`. Supported formats: \
                 {SUPPORTED_FORMATS}"
            ))
        })?;
    Ok(Track {
        format,
        decoder,
        id,
        rate,
        channels,
    })
}

/// Counts packets in a row that produced no audio.
struct Unproductive(usize);

impl Unproductive {
    fn note(&mut self, what: &str) -> Result<(), SpeechError> {
        self.0 += 1;
        if self.0 >= MAX_UNPRODUCTIVE_PACKETS {
            return Err(invalid(format_args!(
                "{MAX_UNPRODUCTIVE_PACKETS} consecutive packets {what}; the stream changed \
                 mid-file (for example a second file appended) or is corrupt"
            )));
        }
        Ok(())
    }
}

/// Decodes a whole file.
pub(super) fn decode(
    source: impl MediaSource + 'static,
    limits: DecodeLimits,
) -> Result<AudioBuffer, SpeechError> {
    let source = MediaSourceStream::new(Box::new(source), MediaSourceStreamOptions::default());
    let mut track = open(source)?;
    let max_frames =
        usize::try_from(track.rate.frames_in(limits.max_duration)).unwrap_or(usize::MAX);
    let mut samples = Vec::new();
    let mut interleaved = Vec::new();
    let mut saw_packet = false;
    let mut unproductive = Unproductive(0);
    loop {
        let packet = match track.format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            // A truncated file: keep what was decoded.
            Err(SymphoniaError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(invalid(
                    "the stream changes mid-file (for example a chained Ogg); split it first",
                ));
            }
            Err(e) => return Err(invalid(format_args!("failed to read a packet: {e}"))),
        };
        if packet.track_id != track.id {
            continue;
        }
        saw_packet = true;
        let audio = match track.decoder.decode(&packet) {
            Ok(audio) => audio,
            Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => {
                unproductive.note("failed to decode")?;
                continue;
            }
            Err(SymphoniaError::ResetRequired) => {
                return Err(invalid(
                    "the codec reconfigures mid-file; split the file first",
                ));
            }
            Err(e) => return Err(invalid(e)),
        };
        let spec = audio.spec();
        if spec.rate() != track.rate.hz() || spec.channels().count() != track.channels {
            return Err(invalid("the sample rate or channel count changes mid-file"));
        }
        if audio.frames() == 0 {
            unproductive.note("decoded to no audio")?;
            continue;
        }
        unproductive.0 = 0;
        if samples.len() + audio.frames() > max_frames {
            return Err(invalid(format_args!(
                "the audio is longer than the {} s limit",
                limits.max_duration.as_secs()
            )));
        }
        interleaved.clear();
        interleaved.resize(audio.samples_interleaved(), 0.0_f32);
        audio.copy_to_slice_interleaved(&mut interleaved);
        #[expect(clippy::cast_precision_loss, reason = "channel counts are tiny")]
        let scale = 1.0 / track.channels as f32;
        samples.extend(
            interleaved
                .chunks_exact(track.channels)
                .map(|frame| to_session_range(frame.iter().sum::<f32>() * scale)),
        );
    }
    if samples.is_empty() && saw_packet {
        return Err(invalid("no audio could be decoded: every packet failed"));
    }
    Ok(AudioBuffer::new(track.rate, samples))
}
