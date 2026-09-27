#!/usr/bin/env bash
# Starts the GrainSpeech daemon (CPU only): models and NormalizeText assets stay resident, so
# every later synthesis call skips the ~2.4 s one-shot asset loading.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
GS=/home/oem/Basir/TTS/GrainSpeech
FORK=$GS/grainspeech-16khz-fa-en
ESPEAK=/home/oem/Basir/TTS/Piper/piper_linux_x86_64/piper/espeak-ng-data

exec ./build/GrainspeechTTSInfer --daemon \
  --model  "$FORK/exports/grainspeech-16k.onnx" \
  --vocoder-model /home/oem/Basir/Vocoder/Trained/vocos-matcha-16KHz-80mels-v1/vocos-matcha-16khz-80mels.onnx \
  --symbols "$FORK/exports/symbols-16k.txt" \
  --lexicon-dir "$GS/lex_current" \
  --sample-rate 16000 \
  --espeak-data "$ESPEAK" \
  --ezafe-onnx ./assets/ezafe_model.onnx \
  --ezafe-spiece ./assets/ezafe_spiece.model \
  --homograph ./assets/homograph_data.json \
  --shakkelha ./assets/shakkelha.onnx
