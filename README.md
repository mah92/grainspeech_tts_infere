# grainspeech_tts_infere

CPU-only C++ inference for the **GrainSpeech** 16 kHz Persian/English TTS model, built the same
way as `match_tts_infer`: NormalizeText is a git submodule in this repository, ONNX Runtime does
the neural work, and no GPU is used at any point.

```
text ──► NormalizeText (C++ G2P, submodule) ──► model-format phone tokens (lexicon lookup)
     ──► GrainSpeech acoustic model (ONNX, phones → mel)
     ──► Vocos vocoder (ONNX, mel → mag/x/y) ──► ISTFT ──► WAV
```

## Why the phones come from a lexicon, not straight from the IPA

The GrainSpeech model is trained on the phone stream of its own alignments (phoneme-level tokens,
stress and length attached to the vowel, every token tagged `fa:` / `en:`). Feeding it the plain
espeak IPA stream gives unintelligible speech. `NormalizeText/grain_phones.cpp` (the `--grain`
flag of `NormalizeCSV`) reproduces the training convention by looking each word up in
`infer_lex_<lang>.txt` — lexicons derived from the model's alignments — with the word's IPA as an
OOV fallback. This program calls the same module in-process, so the C++ path emits exactly the
tokens the Python path does (verified: identical token sequences, sample correlation 1.0 and
log-mel correlation 0.9999 against the Python implementation).

## Build

```bash
# ONNX Runtime (CPU) at /usr/local, ICU, espeak-ng, and the NormalizeText submodule:
git submodule update --init --recursive
mkdir -p build && cd build && cmake .. && make -j
```

`NormalizeText` must contain the `--grain` support (`grain_phones.cpp` / `grain_phones.h`).
For a local checkout, point `NormalizeText` at it instead of a submodule clone.

## Assets

The NormalizeText models are expected next to the working directory (`./assets/`): `ezafe_model.onnx`,
`ezafe_spiece.model`, `homograph_data.json`, `shakkelha.onnx`, and the optional hazm `.dat` files.
They live in the NormalizeText repository (`assets/`); symlinks are fine.

## Models

| File | What it is | Where it comes from |
|------|------------|---------------------|
| `grainspeech-16k.onnx` | acoustic model (phones → mel), 1.14 MB | `export_grainspeech_onnx.py` in the GrainSpeech training fork (input `phoneme` int64 [1,N], outputs `mel` float32 [1,T,80] and `mel_len`) |
| `vocos-matcha-16khz-80mels.onnx` | 16 kHz Vocos (mel → mag/x/y) | `~/Basir/Vocoder/Trained/vocos-matcha-16KHz-80mels-v1/` |
| `symbols-16k.txt` | symbol table, one token per line, **line number = id** | generated next to the ONNX by `export_grainspeech_onnx.py` |
| `infer_lex_fa.txt`, `infer_lex_en.txt` | word → model phone tokens | `extract_lexicon_from_alignments.py` (from the training alignments) |

## Run

```bash
bash test.sh "سلام. حال شما چطور است؟" out.wav
```

or directly:

```bash
./build/GrainspeechTTSInfer \
  --text "سلام. حال شما چطور است؟" \
  --model  <grainspeech-16k.onnx> \
  --vocoder-model <vocos-matcha-16khz-80mels.onnx> \
  --symbols <symbols-16k.txt> \
  --lexicon-dir <dir with infer_lex_fa.txt / infer_lex_en.txt> \
  --sample-rate 16000 \
  --espeak-data <espeak-ng-data> \
  --output out.wav --debug
```

Other flags: `--main-lang FA|EN`, `--speed`, `--play`, `--daemon`, `--stop`, `--help`.

## Daemon mode (for latency)

Timings measured on this machine for a 1.7 s utterance (CPU only):

| Stage | one-shot | daemon |
|-------|----------|--------|
| NormalizeText (asset loading included) | 2356 ms | ~130 ms (models stay resident) |
| GrainSpeech acoustic model | 1.8 ms | 1.8 ms |
| Vocos + ISTFT | 12.1 ms | 12.1 ms |

The neural work is ~14 ms; the one-shot cost is almost entirely loading the NormalizeText assets.
Run `--daemon` once and subsequent `--text` calls reuse it over a private socket
(`/tmp/grainspeech_infer.sock` — deliberately NOT the Matcha daemon's socket, so neither program
hijacks the other).

## Reproduce the ONNX export

```bash
python tools/export_grainspeech_onnx.py \
  --ckpt <training checkpoint> --config <preprocess.yaml> \
  --out grainspeech-16k.onnx --symbols-out symbols-16k.txt
```
It prints the ONNX-vs-PyTorch maximum difference (≈9e-06) and checks several sequence lengths so
the dynamic axis is verified, not assumed.
