# FunASR Nano INT8 on x86 CPUs

FunASR Nano is supported by `AsrConfig::offline` with `AsrFamily::FunAsrNano`.
It needs an encoder adaptor, an LLM, an embedding model, a tokenizer directory,
and a separate Silero VAD model. This guide covers a CPU compatibility problem
in the `sherpa-onnx-funasr-nano-int8-2025-12-30` encoder and an offline workaround.

## Symptoms and affected machines

On some x86 CPUs, speech produces empty text or short, unrelated phrases such
as “嗯。” or “嗯，好。” even for the model's own test audio. Check a known WAV
file before changing microphone gain or VAD settings.

[ONNX Runtime documents U8S8 saturation on AVX2 and AVX512 CPUs without VNNI](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#when-and-why-do-i-need-to-try-u8u8).
The optimized integer kernel can clamp intermediate sums to 16 bits.
U8U8 avoids that saturation path. The documented problem does not affect Arm
or x86 CPUs with VNNI; this workaround is for CPU inference and is not required
on every machine.

The problem was reproduced on an Intel Core i7-8559U Mac with sherpa-onnx
1.13.8 and ONNX Runtime 1.28.2. The original encoder returned empty text for
`rag_math.wav` and `rag_medical.wav` even when calling sherpa-onnx directly,
without speechkit or VAD. Converting only the encoder to U8U8 recovered the
reference transcripts. No microphone recordings are needed to reproduce this.

## Convert an existing model offline

Use the model you already have. Extract its original archive into a model
directory, then run the command below from the speechkit repository root.
The [conversion script](../scripts/fix_funasr_encoder_u8u8.py) requires Python
3.9 or newer and only uses the standard library. It makes no network requests.

```sh
python3 scripts/fix_funasr_encoder_u8u8.py \
  /path/to/model/encoder_adaptor.int8.onnx \
  /path/to/model/encoder_adaptor.u8u8.onnx
```

On Windows, use `python` or `py -3` instead of `python3` if needed. Paths may
point anywhere; the cache location is optional.

The script checks the original encoder's SHA256 before converting. It only
accepts the tested export, refuses an already converted encoder, and checks
the output against the tested conversion's SHA256. It never overwrites an
existing output or the source. A failed write or checksum check removes the
partial output.

Successful output is:

```text
Converted 292 MatMulInteger operations / 584 initializers to U8U8
```

Once conversion succeeds, stop applications using the model. On macOS or Linux,
back up and replace the encoder with this sequence. The backup must not already
exist; `mkdir` stops the sequence if it does.

```sh
mkdir /path/to/model/.speechkit-backup &&
  cp /path/to/model/encoder_adaptor.int8.onnx \
     /path/to/model/.speechkit-backup/encoder_adaptor.int8.onnx &&
  mv /path/to/model/encoder_adaptor.u8u8.onnx \
     /path/to/model/encoder_adaptor.int8.onnx
```

On Windows, create the backup folder, copy the original encoder into it, then
replace the original with the generated file using Explorer. Keep the installed
filename `encoder_adaptor.int8.onnx` so the model directory has one preferred
encoder. The LLM, embedding, tokenizer, and application binaries stay the same.
Reload the model after replacing the file.

The conversion changes INT8 weights and their zero points to UINT8 by adding
128 to both. Their represented values are preserved:

```text
scale * ((weight + 128) - (zero_point + 128))
    = scale * (weight - zero_point)
```

It changes the integer kernel selected by the runtime, without retraining or
recovering the original floating-point weights. The script is restricted to
this tested encoder; an unknown checksum requires diagnosing that export
separately.

| Encoder | SHA256 |
| --- | --- |
| Original U8S8 | `d0246c823f2c34133ae0efee395d8a189c8f92643e3432f866939ee34d34492c` |
| Converted U8U8 | `29e0b6796fc7fed4440561aa31a70f7087915bf23dafbea59a382ca9bb3bd623` |

## Verify and move to another computer

Run the installed model with its existing test audio and VAD file:

```sh
speechkit transcribe /path/to/model/test_wavs/rag_math.wav \
  --backend sherpa-offline --family funasr-nano --provider cpu \
  --model /path/to/model --vad /path/to/silero_vad.onnx
```

The expected text is “对微分形式的积分是微分几何中的基本概念。”.
Also try `rag_medical.wav`, whose expected text is
“肾脏中肾小球囊上的细胞膜孔隙很小。”.
This addresses the severe empty-text and hallucination problem; it does not
guarantee accurate recognition for every recording. Retest on each target machine.

To move the fixed model, copy the whole model directory, including its three
ONNX files and `Qwen3-0.6B` tokenizer directory. Copy `silero_vad.onnx` separately
if the target machine lacks it, then select both in the application. Conversion
only needs to be done once. Alternatively, take the original archive and this
script to the other computer and convert the extracted encoder there.

The original archive still contains the original encoder. Extracting it again
does not carry over the fix. Also, `cargo xtask fetch-fixtures` checks original
model hashes: it will reject a modified encoder in a fixture directory. Keep a
separate converted model directory if you also need unmodified fixtures.

## Restore the original encoder

Stop applications using the model, then copy the backup over the installed
encoder and reload:

```sh
cp /path/to/model/.speechkit-backup/encoder_adaptor.int8.onnx \
   /path/to/model/encoder_adaptor.int8.onnx
```
