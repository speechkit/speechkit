//! `speechkit devices`: list the audio devices `mic` and `speak --play` can use.

use std::io::Write;

use speechkit::io::{DeviceInfo, Microphone, Speaker};

use crate::CliError;

/// Prints both lists. A list that cannot be read does not hide the other:
/// the command fails with the first such error, and a second one is
/// printed on stderr.
pub(crate) fn devices(stdout: &mut dyn Write, stderr: &mut dyn Write) -> Result<(), CliError> {
    let lists = [("input", Microphone::list()), ("output", Speaker::list())];
    let mut failure: Option<CliError> = None;
    for (index, (kind, devices)) in lists.into_iter().enumerate() {
        if index > 0 {
            writeln!(stdout).map_err(|e| CliError::input(e.to_string()))?;
        }
        match devices {
            Ok(devices) => {
                print(stdout, kind, &devices).map_err(|e| CliError::input(e.to_string()))?;
            }
            Err(error) => {
                let mut error = CliError::from(error);
                error.message = format!("cannot list {kind} devices: {}", error.message);
                if failure.is_some() {
                    let _ = writeln!(stderr, "error: {error}");
                } else {
                    failure = Some(error);
                }
            }
        }
    }
    failure.map_or(Ok(()), Err)
}

fn print(stdout: &mut dyn Write, kind: &str, devices: &[DeviceInfo]) -> std::io::Result<()> {
    writeln!(stdout, "{kind} devices:")?;
    if devices.is_empty() {
        writeln!(stdout, "  (none)")?;
    }
    for device in devices {
        let marker = if device.is_default { " (default)" } else { "" };
        writeln!(stdout, "  {}{marker}", device.name)?;
    }
    Ok(())
}
