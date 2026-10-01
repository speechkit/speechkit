## What and why

## Checklist (see AGENTS.md)

- [ ] Every new behavior has a test; a change to a contract rule (`A-xx`, `T-xx`, `D-xx`) updates the check of the same name in `speechkit-testkit`.
- [ ] No `unwrap`, `panic!`, or unexplained `expect` in library code, and lint exceptions use `#[expect(..., reason = "...")]`.
- [ ] Public items have doc comments.
- [ ] Audio, transcripts, and synthesized text are never logged above `trace`, and secrets never.
