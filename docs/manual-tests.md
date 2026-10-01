# Manual tests

Checklists for behavior that needs hardware. Run each on Linux, macOS, and Windows before a release that changes the code involved, and record the result in the table.

## Microphone capture (`speechkit mic`)

Prepare a model, for example streaming Zipformer:

```sh
speechkit mic --backend sherpa-streaming --model sherpa-onnx-streaming-zipformer-en-2023-06-26
```

1. The command prints "listening; press Enter to stop".
2. Speaking shows partial results rewriting one line on the terminal.
3. Pausing commits a segment, printed on its own line with timestamps.
4. Pressing Enter stops capture; the transcript is printed to stdout and the command exits with code 0.
5. Redirecting stdout (`speechkit mic ... > out.txt`) leaves only the transcript in the file.
6. A 44.1 kHz or 48 kHz default device works (the session resamples).
7. A stereo device works (channels are averaged).
8. Unplugging a USB microphone mid-session logs a warning that the microphone stopped, ends the session, and prints the transcript so far.
9. With no input device, the command fails with exit code 4 and "no input device".
10. On macOS, denying microphone permission fails cleanly instead of hanging.
11. `speechkit devices` lists the input devices and marks the default one.
12. `--device` with part of a non-default device's name (for example a USB microphone or a BlackHole loopback) captures from that device; a name matching nothing fails with exit code 3 and lists the devices.
13. With a cloud backend (`--backend openai-realtime` or `dashscope`), words spoken as soon as "listening" appears, while the connection is still opening, are in the transcript: the listening holds the audio until its session opens.

| Date | OS | Device | Result | Notes |
|---|---|---|---|---|
| | Linux | | not run | |
| 2026-09-24 | macOS 15.7 (Intel x86_64) | MacBook Pro built-in microphone (mono, 48 kHz) | partial: 1–6 pass | Streaming Zipformer EN. 7 not run (no stereo input), 8 not run (no USB microphone), 9 not run (no way to remove the built-in input), 10 not run. |
| | Windows | | not run | |

## Speaker playback (`speechkit speak --play`)

Prepare a TTS model, for example a Piper voice:

```sh
speechkit speak "The quick brown fox jumps over the lazy dog." --backend sherpa --model vits-piper-en_US-amy-low --play
```

1. The sentence plays once, at normal pitch and speed, with no clicks at the start or the end.
2. A 44.1 kHz or 48 kHz default device works (audio is resampled from the model's rate).
3. A stereo device plays the voice on both channels.
4. A paragraph of several sentences plays without gaps between chunks; no underrun warning is printed on an idle machine.
5. Under heavy CPU load, underruns are reported as warnings and playback continues.
6. Unplugging USB headphones mid-sentence fails with "the output device stopped", cancels synthesis, and exits with a non-zero code.
7. With no output device, the command fails with "no output device".
8. Pressing Ctrl-C stops playback promptly.
9. `--play --device` with part of a non-default output device's name plays through that device.

| Date | OS | Device | Result | Notes |
|---|---|---|---|---|
| | Linux | | not run | |
| 2026-09-24 | macOS 15.7 (Intel x86_64) | MacBook Pro built-in speakers (stereo, 44.1 kHz) | partial: 1–5, 8 pass | Piper amy-low (16 kHz). 5: no underrun could be triggered with 16 busy loops on 8 cores; playback stayed smooth. 6 and 7 not run (no USB output device, no way to remove the built-in output). |
| | Windows | | not run | |
