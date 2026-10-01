//! What a VAD says about speech activity, for backends that report it.

use std::time::Duration;

use super::{SpeechSegment, Vad};
use crate::SampleRate;

/// One thing a [`Tracker`] found, in order.
#[derive(Debug)]
pub(crate) enum Detected {
    /// Speech started here.
    Started(Duration),
    /// A segment of speech ended at `end`. `ended` says whether the speech
    /// stopped there; otherwise the segment was cut and the speech goes on.
    Segment {
        segment: SpeechSegment,
        end: Duration,
        ended: bool,
    },
    /// Speech the VAD dropped as too short ended by `end`.
    Dropped { start: Duration, end: Duration },
    /// Every start and end before this point was reported.
    Known(Duration),
}

/// Runs a [`Vad`] on one stream and turns what it finds into speech
/// activity: starts, segments that end speech or cut it, dropped speech,
/// and how far activity is known.
pub(crate) struct Tracker {
    vad: Box<dyn Vad>,
    rate: SampleRate,
    start_delay: Duration,
    /// Samples fed to the VAD.
    fed: u64,
    /// The start of the speech in progress, once reported and until it
    /// ends. After a cut, the start of the rest.
    speech: Option<Duration>,
    /// The last point reported as known.
    known: Duration,
}

impl Tracker {
    /// Tracks `vad`, which runs at `rate` and reports starts
    /// `start_delay` late.
    pub(crate) fn new(vad: Box<dyn Vad>, rate: SampleRate, start_delay: Duration) -> Self {
        Self {
            vad,
            rate,
            start_delay,
            fed: 0,
            speech: None,
            known: Duration::ZERO,
        }
    }

    /// Feeds `samples`, at the VAD's rate.
    pub(crate) fn accept(&mut self, samples: &[f32]) -> Vec<Detected> {
        self.fed += samples.len() as u64;
        let segments = self.vad.accept(samples);
        self.report(segments, false)
    }

    /// Ends the stream: everything is known.
    pub(crate) fn finish(&mut self) -> Vec<Detected> {
        let segments = self.vad.flush();
        self.report(segments, true)
    }

    fn report(&mut self, segments: Vec<SpeechSegment>, end_of_input: bool) -> Vec<Detected> {
        let mut out = Vec::new();
        let speaking = self.vad.speaking_since();
        let starts: Vec<Duration> = segments.iter().skip(1).map(|s| s.start).collect();
        for (index, segment) in segments.into_iter().enumerate() {
            let end = segment.start + self.rate.duration_of(segment.samples.len() as u64);
            // What follows the segment: the next one, or the speech in
            // progress. Speech that starts where the segment ends was cut.
            let next = starts.get(index).copied().or(speaking);
            let cut = next.is_some_and(|start| start <= end);
            if self.speech.is_none() {
                out.push(Detected::Started(segment.start));
            }
            self.speech = if cut { next } else { None };
            out.push(Detected::Segment {
                segment,
                end,
                ended: !cut,
            });
        }
        let fed = self.rate.duration_of(self.fed);
        let known = if end_of_input {
            fed
        } else {
            fed.saturating_sub(self.start_delay)
        };
        if let Some(start) = self.speech
            && speaking != Some(start)
        {
            // Speech the VAD dropped as too short: it still ends, somewhere
            // before what is known now.
            let end = known.max(start).min(speaking.unwrap_or(Duration::MAX));
            self.speech = None;
            out.push(Detected::Dropped { start, end });
        }
        if self.speech.is_none()
            && let Some(start) = speaking
        {
            self.speech = Some(start);
            out.push(Detected::Started(start));
        }
        let through = self.speech.map_or(known, |start| start.min(known));
        if through > self.known {
            self.known = through;
            out.push(Detected::Known(through));
        }
        out
    }
}
