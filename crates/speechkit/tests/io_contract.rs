//! The device contract (rules `D-01` to `D-06`) on the fake devices.
#![cfg(feature = "devices")]

use speechkit_testkit::contract::devices as contract;

macro_rules! rules {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                contract::$name();
            }
        )*
    };
}

rules!(
    d01_held_position_complete,
    d02_behind_fails,
    d03_wake_reservation,
    d04_stop_returns_at_once,
    d05_device_lost,
    d06_played_bounded,
);
