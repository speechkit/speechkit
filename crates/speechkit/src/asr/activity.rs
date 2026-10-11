//! How far a backend knows speech activity.

use std::time::Duration;

/// The last [`ActivityKnown`](super::AsrEvent::ActivityKnown) a stream
/// sent. It only moves forward, and never past the start of speech still
/// in progress, since that speech's end is not known yet.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct KnownActivity(Duration);

impl KnownActivity {
    /// The last point sent.
    #[cfg_attr(
        not(feature = "dashscope"),
        expect(dead_code, reason = "only DashScope starts speech where activity is known")
    )]
    pub(crate) fn get(self) -> Duration {
        self.0
    }

    /// Activity is known through `candidate`, held back to the start of
    /// `speech` if speech is in progress. Returns the point to send if it
    /// moved forward.
    pub(crate) fn advance(
        &mut self,
        candidate: Duration,
        speech: Option<Duration>,
    ) -> Option<Duration> {
        let through = speech.map_or(candidate, |start| start.min(candidate));
        (through > self.0).then(|| {
            self.0 = through;
            through
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_forward_and_stops_at_speech() {
        let ms = Duration::from_millis;
        let mut known = KnownActivity::default();
        assert_eq!(known.advance(ms(500), None), Some(ms(500)));
        assert_eq!(known.advance(ms(400), None), None);
        assert_eq!(known.advance(ms(900), Some(ms(700))), Some(ms(700)));
        assert_eq!(known.advance(ms(1_000), Some(ms(700))), None);
        assert_eq!(known.0, ms(700));
    }
}
