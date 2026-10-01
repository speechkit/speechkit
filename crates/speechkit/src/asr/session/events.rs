//! Where a backend stream's results go.

use std::sync::Arc;

use super::Shared;
use crate::{Flow, SpeechError, asr::AsrEvent, speech::sync::lock};

/// Where a stream sends its results: `Clone + Send`, so a stream can send
/// from `accept` or from a thread or task of its own, at any time (A-10).
///
/// The engine adds the session's origin to the times, post-processes
/// segments, and records every event in the session's history for its
/// readers.
#[derive(Clone)]
pub struct AsrEvents {
    target: Target,
}

/// Where events go.
#[derive(Clone)]
enum Target {
    /// A session.
    Session(Arc<Shared>),
    /// Functions of the crate's own, such as a worker process forwarding
    /// events to its parent.
    Forward {
        send: Arc<dyn Fn(AsrEvent) -> Flow + Send + Sync>,
        fail: Arc<dyn Fn(SpeechError) + Send + Sync>,
    },
}

impl AsrEvents {
    pub(super) fn session(shared: Arc<Shared>) -> Self {
        Self {
            target: Target::Session(shared),
        }
    }

    /// Events that go to `send`, and failures to `fail`, as they are: for
    /// the worker process, and for tests that drive a stream without an
    /// engine. Not part of the API, and not covered by semver.
    #[doc(hidden)]
    pub fn forward(
        send: impl Fn(AsrEvent) -> Flow + Send + Sync + 'static,
        fail: impl Fn(SpeechError) + Send + Sync + 'static,
    ) -> Self {
        Self {
            target: Target::Forward {
                send: Arc::new(send),
                fail: Arc::new(fail),
            },
        }
    }

    /// Sends an event. Returns [`Flow::Stop`] once the session has ended or
    /// is failing, and the event is then discarded (A-03): the stream
    /// should stop as soon as it can.
    pub fn send(&self, event: AsrEvent) -> Flow {
        let shared = match &self.target {
            Target::Session(shared) => &**shared,
            Target::Forward { send, .. } => return send(event),
        };
        if shared.terminal.get().is_some() {
            return Flow::Stop;
        }
        let origin = shared.origin;
        let event = match event {
            AsrEvent::Segment(mut segment) => {
                segment.start += origin;
                segment.end += origin;
                // Post-process outside the session lock, since it may run a
                // model, but hold the post-processor until the segment is
                // recorded, so segments from several threads stay in order.
                let post = shared.post.as_ref().map(lock);
                if let Some(mut post) = post {
                    segment.text = post.apply(std::mem::take(&mut segment.text));
                    return record(shared, AsrEvent::Segment(segment));
                }
                AsrEvent::Segment(segment)
            }
            AsrEvent::SpeechStarted { at } => AsrEvent::SpeechStarted { at: at + origin },
            AsrEvent::SpeechEnded { at, utterance } => AsrEvent::SpeechEnded {
                at: at + origin,
                utterance,
            },
            AsrEvent::ActivityKnown { through } => AsrEvent::ActivityKnown {
                through: through + origin,
            },
            partial @ AsrEvent::Partial(_) => partial,
        };
        record(shared, event)
    }

    /// Sends every event in order, stopping at the first [`Flow::Stop`].
    #[cfg_attr(
        not(any(feature = "sherpa", feature = "openai", feature = "dashscope")),
        expect(dead_code, reason = "only the backends that batch events use it")
    )]
    pub(crate) fn send_all(&self, events: impl IntoIterator<Item = AsrEvent>) -> Flow {
        for event in events {
            if self.send(event) == Flow::Stop {
                return Flow::Stop;
            }
        }
        Flow::Continue
    }

    /// Reports that the stream broke between calls, such as a dropped
    /// connection. The session fails with `error` at once, keeping what it
    /// confirmed; its stream is cancelled and dropped.
    pub fn fail(&self, error: SpeechError) {
        let shared = match &self.target {
            Target::Session(shared) => &**shared,
            Target::Forward { fail, .. } => return fail(error),
        };
        let mut core = shared.lock();
        if shared.stopping(&core) {
            return;
        }
        core.failure = Some(error);
        drop(core);
        shared.changed.notify_all();
    }
}

/// Records `event` in the session's history, and publishes the turn ends
/// and the endpoint it confirms.
fn record(shared: &Shared, event: AsrEvent) -> Flow {
    {
        let mut core = shared.lock();
        if shared.stopping(&core) {
            return Flow::Stop;
        }
        let observed = core.endpoints.observe(&event);
        let max = shared.max_history_bytes;
        let mut recorded = core.history.apply(event, max);
        for turn in observed.turns {
            recorded = recorded.and_then(|()| core.history.turn_ended(turn, max));
        }
        if let Err(error) = recorded {
            // The history is full: fail, keeping what was confirmed (A-11).
            core.failure = Some(error);
            drop(core);
            shared.changed.notify_all();
            return Flow::Stop;
        }
        if observed.endpoint {
            // C is how far the input had got when the endpoint was
            // confirmed: with a local model, the end of the block that
            // confirmed it (A-07).
            let at = core.position;
            shared.cut(&mut core, at);
        }
        drop(core);
    }
    shared.changed.notify_all();
    Flow::Continue
}

impl std::fmt::Debug for AsrEvents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("AsrEvents");
        if let Target::Session(shared) = &self.target {
            debug.field("session", &shared.id);
        }
        debug.finish_non_exhaustive()
    }
}
