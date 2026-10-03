//! The toggle-to-talk handlers from the guide, shared by the compile
//! scenarios and the fake-device scenario tests. `Ui` and `Worker` are
//! supplied by the application or test harness.

use std::{error::Error, time::Duration};

use speechkit::{
    Deadline,
    asr::{AsrEngine, AsrOptions, AsrResult, AsrUpdate},
    io::{Listening, Microphone},
};

use super::{Ui, Worker};

/// Takes a round exactly once, stops its input, and finishes off the UI thread.
pub(super) fn finish_dictation(active: &mut Option<(u64, Listening)>, worker: &Worker) {
    if let Some((id, listening)) = active.take() {
        listening.stop();
        let deadline = Deadline::from(Duration::from_secs(5));
        worker.submit(id, move || listening.finish(deadline));
    }
}

/// Handles a non-repeated key-down; key-up does nothing.
pub(super) fn dictation_toggle_to_talk(
    active: &mut Option<(u64, Listening)>,
    mic: &Microphone,
    engine: &AsrEngine,
    ui: &Ui,
    worker: &Worker,
) -> Result<(), Box<dyn Error>> {
    if active.is_some() {
        finish_dictation(active, worker);
    } else {
        let listening = mic.listen(engine, AsrOptions::default().with_hints(ui.contact_names()))?;
        let id = ui.begin_dictation();
        ui.forward_updates(id, listening.updates());
        *active = Some((id, listening));
    }
    Ok(())
}

/// Finishes a self-ended round without clearing a newer round or inserting text.
pub(super) fn dictation_toggle_closed(
    active: &mut Option<(u64, Listening)>,
    id: u64,
    update: &AsrUpdate,
    worker: &Worker,
) {
    if matches!(update, AsrUpdate::Closed(_))
        && active.as_ref().is_some_and(|(current, _)| *current == id)
    {
        finish_dictation(active, worker);
    }
}

/// Routes the worker's result to its original round, without changing `active`.
pub(super) fn dictation_toggle_finished(id: u64, result: AsrResult, ui: &Ui) {
    ui.complete_dictation(id, result);
}
