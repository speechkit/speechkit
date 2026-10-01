//! Behavior of the real binary that needs no backend: help and usage errors.

use assert_cmd::Command;

#[test]
fn help_lists_the_commands() {
    let output = Command::cargo_bin("speechkit")
        .unwrap()
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for command in ["transcribe", "stream", "serve"] {
        assert!(help.contains(command), "{help}");
    }
}

#[test]
fn usage_errors_exit_with_2() {
    Command::cargo_bin("speechkit")
        .unwrap()
        .args(["transcribe", "--no-such-flag"])
        .assert()
        .code(2);
    Command::cargo_bin("speechkit").unwrap().assert().code(2);
}
