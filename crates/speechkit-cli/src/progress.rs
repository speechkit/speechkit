//! Showing progress on stderr while a session runs.

use std::io::Write;

use std::time::Instant;

use speechkit::{
    RecvError,
    asr::{AsrSession, AsrUpdate, AsrUpdates},
};

use crate::output::short_time;

/// Prints updates on stderr: committed segments with their times, and on
/// an interactive terminal, partial results rewriting one line.
struct Display<'a> {
    stderr: &'a mut dyn Write,
    interactive: bool,
    partial_shown: bool,
}

impl Display<'_> {
    /// Shows `update`, and says whether the session closed.
    fn show(&mut self, update: &AsrUpdate) -> bool {
        let clear = if self.partial_shown { "\r\x1b[K" } else { "" };
        let stderr = &mut *self.stderr;
        let _ = match update {
            AsrUpdate::Segment(segment) => {
                self.partial_shown = false;
                match segment.text.trim() {
                    "" => write!(stderr, "{clear}"),
                    text => writeln!(
                        stderr,
                        "{clear}[{} → {}] {text}",
                        short_time(segment.start),
                        short_time(segment.end),
                    ),
                }
            }
            AsrUpdate::Partial(partial) if self.interactive => {
                self.partial_shown = true;
                write!(stderr, "{clear}… {}", partial.text)
            }
            AsrUpdate::Closed(_) => write!(stderr, "{clear}"),
            _ => Ok(()),
        };
        let _ = stderr.flush();
        matches!(update, AsrUpdate::Closed(_))
    }
}

/// Shows updates until the session closes or `deadline` passes (then the
/// session is cancelled).
pub(crate) fn show_until_done(
    mut updates: AsrUpdates,
    stderr: &mut dyn Write,
    interactive: bool,
    session: &AsrSession,
    deadline: Instant,
) {
    let mut display = Display {
        stderr,
        interactive,
        partial_shown: false,
    };
    loop {
        let update = match updates.recv(deadline) {
            Ok(update) => update,
            Err(RecvError::Timeout | RecvError::Empty) => {
                session.cancel();
                continue;
            }
            Err(RecvError::Closed) => break,
        };
        if display.show(&update) {
            break;
        }
    }
}

/// Shows updates until the session closes, for a live session that ends
/// when its input does.
pub(crate) fn show_all(updates: AsrUpdates, stderr: &mut dyn Write, interactive: bool) {
    let mut display = Display {
        stderr,
        interactive,
        partial_shown: false,
    };
    for update in updates {
        if display.show(&update) {
            break;
        }
    }
}
