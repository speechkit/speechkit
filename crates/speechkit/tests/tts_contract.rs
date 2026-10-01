//! The TTS contract against the fake backend.

use speechkit::tts::TtsEngine;
use speechkit_testkit::{contract::tts as contract, tts::FakeTts};

fn make() -> TtsEngine {
    TtsEngine::new(FakeTts::plain())
}

macro_rules! generic {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                contract::$name(&make);
            }
        )*
    };
}

macro_rules! faults {
    ($($name:ident),* $(,)?) => {
        mod faults {
            use super::contract::faults;
            $(
                #[test]
                fn $name() {
                    faults::$name();
                }
            )*
        }
    };
}

generic!(
    t01_output,
    t02_drop_rules,
    t03_marks,
    t04_validation,
    t06_finish_idempotent,
    t07_cancel_releases_after_backend,
    t08_incremental_text,
    t09_failure_keeps_progress,
    t10_defaults,
    t11_slot_free_after_finish,
);

faults!(
    t01_audio_follows_text_order,
    t01_full_queue_pauses_synthesis,
    t02_finish_waits_for_the_reader,
    t07_cancel_waits_for_native_call,
    t07_start_waits_for_a_slot,
    t04_empty_text_never_reaches_the_backend,
    t09_failure_after_first_chunk,
    panic_is_isolated,
);

#[test]
fn full_suite_runs_in_one_go() {
    contract::run_tts_contract(make);
}
