# nemotron-asr (Go)

A Go implementation of streaming inference for
[Nemotron 3.5 ASR Streaming Multilingual 0.6B](https://huggingface.co/tonythethompson/Nemotron-3.5-ASR-Streaming-0.6B-ONNX)
(cache-aware FastConformer encoder + stateful RNN-T decoder/joint), using
[ONNX Runtime](https://onnxruntime.ai/) via
[`yalue/onnxruntime_go`](https://github.com/yalue/onnxruntime_go).

## How it works

```
raw audio (16kHz mono)
   │  chunked into 8960-sample (560ms) steps, with an 80ms rolling
   │  overlap carried from the previous chunk
   ▼
log-mel spectrogram (128 mels)              internal/melspec
   ▼
encoder.onnx                                internal/asr
   │  cache-aware: carries cache_last_channel / cache_last_time /
   │  cache_last_channel_len across chunks; includes the 2
   │  "extra pre-encoded" frames as input context
   ▼
7 new encoder frames per chunk
   ▼
decoder_joint.onnx  (greedy RNN-T label loop, one call per candidate token)
   │  carries LSTM prediction-network state across the *entire* stream;
   │  state only advances on non-blank predictions
   ▼
token ids → SentencePiece detokenize        internal/tokenizer
   ▼
text
```

The chunk-framing constants (8960 new samples / 1280-sample overlap / 65
mel-frame encoder input) are derived at startup from `config.json`
(`subsampling_factor`, `chunk_size_output_frames`, `drop_extra_pre_encoded`),
not hardcoded — see `internal/asr/framing_test.go`, which checks the
derivation against the exact `test_input`/`test_output` vector the model repo
ships in its own `config.json`.

## Setup

### 1. Get ONNX Runtime

Download a prebuilt release for your OS/arch from
https://github.com/microsoft/onnxruntime/releases (e.g.
`onnxruntime-linux-x64-*.tgz`, `onnxruntime-osx-*.tgz`, or the Windows zip),
and note the path to `libonnxruntime.so` / `libonnxruntime.dylib` /
`onnxruntime.dll` inside it. This is passed via `-ort-lib` below — no
compile-time linking is required (the Go binding loads it dynamically).

### 2. Build

```bash
go mod tidy   # fetches github.com/yalue/onnxruntime_go
go build -o nemotron-asr .
```

### 3. Run

```bash
./nemotron-asr \
  -model-dir /home/alex/.cache/huggingface/hub/models--tonythethompson--Nemotron-3.5-ASR-Streaming-0.6B-ONNX/snapshots/3a2c8c48cf093c890740577d9467f835ca95fdbd \
  -audio ./sample.wav \
  -ort-lib /path/to/libonnxruntime.so \
  -language en-US
```

`sample.wav` can be any PCM16/PCM32/float32 WAV; it's auto-downmixed to mono
and resampled to 16kHz if needed. `-language` must be a key from
`config.json`'s `prompt_dictionary` (or `auto` to let the model pick).

Add `-offline` when the entire local file is available and a single
transcription result is desired. The model export is still a streaming
encoder, so offline mode feeds its required fixed-size windows internally;
it does not send the whole file to the encoder as one tensor (which would
only produce the first seven encoder frames).

Run `go test ./...` to run the framing/feature-extraction unit tests (no
ONNX Runtime or model files required for those).

## Browser microphone server

Start the real-time browser server:

```bash
go run ./cmd/server
```

Open `http://localhost:8081`, select **Auto-detect language** (the default) or
any language supported by the model, and allow microphone access.
The page downsamples microphone audio to 16 kHz mono PCM16 and sends it over
`/ws/transcribe`. The server returns JSON `partial` messages after each 560 ms
model window, followed by a `final` message when **Stop** is selected. The
WebSocket protocol is also suitable for another client: send `{"type":"start",
"language":"es"}`, raw little-endian PCM16 binary messages, then
`{"type":"end"}`.

The server uses the same local model snapshot and ONNX Runtime library as the
CLI defaults; no model-related command-line flags are required.

## Project layout

- `main.go` — CLI: loads `config.json`, wires everything together, feeds a
  WAV file through the model in fixed-size streaming chunks.
- `internal/asr` — ONNX Runtime sessions, cache-aware encoder stepping,
  greedy RNN-T decode loop, all persistent streaming state.
- `internal/melspec` — log-mel feature extraction (own FFT/mel-filterbank,
  no external deps).
- `internal/tokenizer` — minimal SentencePiece `.model` (protobuf) reader;
  only extracts the id→piece vocabulary, since RNN-T inference only ever
  needs to detokenize ids, never encode new text.
- `internal/wav` — minimal WAV reader/downmixer/resampler.

## Known limitations / what to check first if transcripts look wrong

I could not run these ONNX graphs myself to numerically validate output
(no GPU/ONNX Runtime in the environment I built this in) — the project
**does compile and its framing math checks out against the model's own
`config.json` test vector**, but the following areas are reasoned from the
NeMo/FastConformer architecture and config rather than verified byte-for-byte
against a Python reference, and are the first places to check with a
Python/NeMo or `parakeet-rs` reference if accuracy is off:

1. **Mel filterbank scale** — implemented as HTK (`2595*log10(1+f/700)`).
   NeMo's preprocessor can also be configured for the Slaney scale.
2. **Log-guard epsilon** (`log(mel + 1e-5)`) — NeMo's default
   `log_zero_guard_value` varies by config; this is a common source of a
   constant offset rather than gross errors.
3. **STFT windowing/padding** — Hann window, `center=True` with reflect
   padding (matching `torch.stft` defaults). If NeMo's preprocessor was
   exported with `center=False` or a different pad mode, alignment will
   drift by a few samples per frame.
4. **`drop_extra_pre_encoded` is input context** — its two frames are
   included in the 65-frame encoder input. The exported graph's seven-frame
   output already excludes that context; do not discard output frames again.
5. **RNN-T max symbols per step** is capped at 10 (`maxSymbolsPerStep` in
   `internal/asr/asr.go`) as a safety bound against runaway loops; NeMo's
   default is also commonly 10 but check your model's training config if
   you see truncated words.

If you can run the encoder against the `test_input`/`test_output` vector in
`config.json` from Python once, that's the fastest way to confirm/deny #1-3
(feed the 65×128 test mel array in, diff `encoded` against `test_output`).
