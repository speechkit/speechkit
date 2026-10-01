//! The ASR contract against the fake backend.

use speechkit::asr::AsrEngine;
use speechkit_testkit::{asr::FakeAsr, contract::asr as contract};

fn make() -> AsrEngine {
    AsrEngine::new(FakeAsr::hello_world())
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
    a01_refused_chunk_returned,
    a02_backpressure,
    a03_result_never_changes,
    a04_failure_keeps_confirmed,
    a05_readers,
    a06_no_retention,
    a09_slots_and_panics,
    a13_cancel_and_drop,
    a14_deadline_prompt,
    a15_defaults,
    a16_clones_share_slots,
    a17_empty_chunk_noop,
    a18_empty_hints_ignored,
    a19_expired_deadline_no_slot,
);

faults!(
    a02_backpressure_full,
    a02_push_wait_wakes_on_close,
    a03_late_events_discarded,
    a04_failure_keeps_confirmed,
    a05_readers,
    a05_slow_reader,
    a07_endpointing,
    a08_turns,
    a09_slot_held_while_blocked,
    a09_backend_panic_isolated,
    a09_single_slot,
    a10_events_while_idle,
    a11_bounded,
    a14_deadline_while_opening,
    a14_start_waits_for_a_slot,
    a15_feed_blocks,
    a19_expired_deadline_no_slot,
);

#[test]
fn full_suite_runs_in_one_go() {
    contract::run_asr_contract(make);
}
