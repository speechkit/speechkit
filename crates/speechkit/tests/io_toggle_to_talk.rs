//! The guide's toggle-to-talk handlers on a fake microphone and backend.
//! A UI harness receives tagged updates and orders the worker's results.
#![cfg(feature = "devices")]

use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    sync::mpsc,
    thread::JoinHandle,
    time::Duration,
};

use speechkit::{
    SampleRate, SpeechError,
    asr::{AsrEngine, AsrResult, AsrUpdate, AsrUpdates},
    io::Microphone,
};
use speechkit_testkit::{
    Gate,
    asr::{FakeAsr, Script, Step, Trigger},
    eventually,
};

#[path = "support/toggle_to_talk.rs"]
mod toggle_to_talk;

use toggle_to_talk::{
    dictation_toggle_closed, dictation_toggle_finished, dictation_toggle_to_talk,
};

const SETTLE: Duration = Duration::from_secs(10);

/// The application part of the scenario: IDs, update readers, and ordered results.
#[derive(Default)]
struct Ui {
    next: Cell<u64>,
    contacts: RefCell<Vec<String>>,
    updates: RefCell<BTreeMap<u64, AsrUpdates>>,
    pending: RefCell<BTreeMap<u64, Option<AsrResult>>>,
    delivered: RefCell<Vec<(u64, AsrResult)>>,
}

impl Ui {
    fn contact_names(&self) -> Vec<String> {
        self.contacts.borrow().clone()
    }

    fn begin_dictation(&self) -> u64 {
        let id = self.next.get();
        self.next.set(id + 1);
        self.pending.borrow_mut().insert(id, None);
        id
    }

    fn forward_updates(&self, id: u64, updates: AsrUpdates) {
        // The harness reads these as posted UI events, without a real UI loop.
        self.updates.borrow_mut().insert(id, updates);
    }

    fn complete_dictation(&self, id: u64, result: AsrResult) {
        let mut pending = self.pending.borrow_mut();
        let Some(slot) = pending.get_mut(&id) else {
            return;
        };
        if slot.is_some() {
            return;
        }
        *slot = Some(result);
        while pending
            .first_key_value()
            .is_some_and(|(_, result)| result.is_some())
        {
            if let Some((id, Some(result))) = pending.pop_first() {
                self.delivered.borrow_mut().push((id, result));
            }
        }
    }

    fn closed(&self, id: u64) -> AsrUpdate {
        let mut readers = self.updates.borrow_mut();
        let reader = readers
            .get_mut(&id)
            .expect("the round has an update reader");
        loop {
            let update = reader
                .recv(SETTLE)
                .expect("the round ends within the test deadline");
            if matches!(update, AsrUpdate::Closed(_)) {
                return update;
            }
        }
    }
}

/// A worker that executes the scenario's closures on background threads.
struct Worker {
    send: mpsc::Sender<(u64, AsrResult)>,
    results: mpsc::Receiver<(u64, AsrResult)>,
    jobs: RefCell<Vec<JoinHandle<()>>>,
}

impl Worker {
    fn new() -> Self {
        let (send, results) = mpsc::channel();
        Self {
            send,
            results,
            jobs: RefCell::default(),
        }
    }

    fn submit(&self, id: u64, finish: impl FnOnce() -> AsrResult + Send + 'static) {
        let send = self.send.clone();
        self.jobs.borrow_mut().push(std::thread::spawn(move || {
            let _ = send.send((id, finish()));
        }));
    }

    fn result(&self) -> (u64, AsrResult) {
        self.results
            .recv_timeout(SETTLE)
            .expect("the worker returns a result")
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        for job in self.jobs.get_mut().drain(..) {
            job.join().expect("the worker did not panic");
        }
    }
}

/// Always release a blocked backend before joining workers, even on test failure.
struct ReleaseGate(Gate);

impl Drop for ReleaseGate {
    fn drop(&mut self) {
        self.0.release();
    }
}

/// A second round holds its audio while the first finishes on the only
/// engine slot. Late updates cannot stop it, and reversed completion
/// callbacks still deliver each transcript once in start order.
#[test]
fn a_new_round_survives_old_updates_and_results_arriving_out_of_order() {
    let gate = Gate::new();
    let fake = FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1_600), Step::Segment(0, "hello"))
            .then(Trigger::OnFinish, Step::BlockUntilReleased(gate.clone()))
            .then(Trigger::OnFinish, Step::Segment(1, "world")),
    );
    let stats = fake.stats();
    let engine = AsrEngine::new(fake).with_max_sessions(1);
    let (microphone, mic) = Microphone::fake(SampleRate::HZ_16000);
    let ui = Ui::default();
    let worker = Worker::new();
    let _release = ReleaseGate(gate.clone());
    let mut active = None;

    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    let first_audio = vec![0.1; 1_600];
    mic.push(&first_audio);
    assert!(eventually(SETTLE, || stats.heard() == first_audio));

    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    assert!(active.is_none(), "stop returns before the backend finishes");
    assert!(gate.wait_entered(1, SETTLE));
    assert!(matches!(
        worker.results.try_recv(),
        Err(mpsc::TryRecvError::Empty)
    ));

    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    assert_eq!(active.as_ref().map(|(id, _)| *id), Some(1));
    let second_audio = vec![0.2; 3_200];
    mic.push(&second_audio);
    assert_eq!(stats.opened(), 1, "the first round still holds the slot");

    gate.release();
    let closed = ui.closed(0);
    dictation_toggle_closed(&mut active, 0, &closed, &worker);
    dictation_toggle_closed(&mut active, 0, &closed, &worker);
    assert_eq!(active.as_ref().map(|(id, _)| *id), Some(1));
    assert_eq!(
        worker.jobs.borrow().len(),
        1,
        "old Closed does not schedule another finish"
    );
    let (first_id, first_result) = worker.result();
    assert_eq!(first_id, 0);

    let mut heard = first_audio;
    heard.extend(second_audio);
    assert!(eventually(SETTLE, || stats.heard() == heard));
    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    let closed = ui.closed(1);
    dictation_toggle_closed(&mut active, 1, &closed, &worker);
    assert_eq!(
        worker.jobs.borrow().len(),
        2,
        "Closed after stop does not finish twice"
    );
    let (second_id, second_result) = worker.result();
    assert_eq!(second_id, 1);

    // Deliver callbacks in reverse order, then repeat each one.
    dictation_toggle_finished(second_id, second_result.clone(), &ui);
    dictation_toggle_finished(second_id, second_result, &ui);
    assert!(ui.delivered.borrow().is_empty());
    dictation_toggle_finished(first_id, first_result.clone(), &ui);
    dictation_toggle_finished(first_id, first_result, &ui);
    let delivered = ui.delivered.borrow();
    assert_eq!(delivered.len(), 2);
    for (expected, (id, result)) in delivered.iter().enumerate() {
        assert_eq!(*id, expected as u64);
        assert_eq!(result.as_ref().unwrap().text(), "hello world");
    }
    assert!(ui.pending.borrow().is_empty());
}

/// A backend failure clears the matching active round, finishes only
/// once, preserves confirmed text, and does not prevent a later result.
#[test]
fn a_self_ended_failure_uses_the_worker_result_path_once() {
    let engine = AsrEngine::new(FakeAsr::new(
        Script::new()
            .then(Trigger::AfterSamples(1_600), Step::Segment(0, "confirmed"))
            .then(
                Trigger::AfterSamples(1_600),
                Step::Fail(SpeechError::Closed),
            ),
    ));
    let (microphone, mic) = Microphone::fake(SampleRate::HZ_16000);
    let ui = Ui::default();
    let worker = Worker::new();
    let mut active = None;

    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    mic.push(&vec![0.1; 1_600]);
    let closed = ui.closed(0);
    dictation_toggle_closed(&mut active, 0, &closed, &worker);
    dictation_toggle_closed(&mut active, 0, &closed, &worker);
    assert!(active.is_none());
    assert_eq!(worker.jobs.borrow().len(), 1);
    assert!(
        ui.delivered.borrow().is_empty(),
        "Closed never inserts text"
    );
    let (id, result) = worker.result();
    dictation_toggle_finished(id, result, &ui);
    let delivered = ui.delivered.borrow();
    let failure = delivered[0].1.as_ref().unwrap_err();
    assert!(matches!(failure.error, SpeechError::Closed));
    assert_eq!(failure.confirmed.text(), "confirmed");
    drop(delivered);

    let recovered = AsrEngine::new(FakeAsr::hello_world());
    dictation_toggle_to_talk(&mut active, &microphone, &recovered, &ui, &worker).unwrap();
    dictation_toggle_to_talk(&mut active, &microphone, &recovered, &ui, &worker).unwrap();
    let (id, result) = worker.result();
    dictation_toggle_finished(id, result, &ui);
    assert_eq!(
        ui.delivered.borrow().len(),
        2,
        "a failed round does not block later results"
    );
    assert!(ui.delivered.borrow()[1].1.is_ok());
}

/// An immediate start failure leaves no active handle or hole in result
/// order; the next valid key-down starts a fresh round.
#[test]
fn a_failed_start_does_not_reserve_a_result() {
    let engine = AsrEngine::new(FakeAsr::hello_world());
    let (microphone, _mic) = Microphone::fake(SampleRate::HZ_16000);
    let ui = Ui::default();
    ui.contacts.borrow_mut().push("unsupported hint".into());
    let worker = Worker::new();
    let mut active = None;

    let error = dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker)
        .expect_err("the fake does not accept hints");
    assert!(matches!(
        error.downcast_ref::<SpeechError>(),
        Some(SpeechError::Unsupported(_))
    ));
    assert!(active.is_none());
    assert!(ui.pending.borrow().is_empty());
    assert!(worker.jobs.borrow().is_empty());

    ui.contacts.borrow_mut().clear();
    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    assert_eq!(active.as_ref().map(|(id, _)| *id), Some(0));
    dictation_toggle_to_talk(&mut active, &microphone, &engine, &ui, &worker).unwrap();
    let (id, result) = worker.result();
    dictation_toggle_finished(id, result, &ui);
    assert_eq!(ui.delivered.borrow().len(), 1);
    assert!(ui.delivered.borrow()[0].1.is_ok());
}
