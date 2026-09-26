#!/usr/bin/env bash
# End-to-end test of the GrainSpeech C++ inference program (CPU only).
#
# Usage: bash test.sh [TEXT] [OUTPUT_WAV]
#
# The hazm word/verb lists are picked up from ./assets/ defaults inside the binary,
# so they are not repeated here.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

TEXT="${1:-سلام. حال شما چطور است؟}"
OUT="${2:-/tmp/gs_cpp_test.wav}"
GS=/home/oem/Basir/TTS/GrainSpeech
FORK=$GS/grainspeech-16khz-fa-en
ESPEAK=/home/oem/Basir/TTS/Piper/piper_linux_x86_64/piper/espeak-ng-data

./build/GrainspeechTTSInfer \
  --text "$TEXT" \
  --model  "$FORK/exports/grainspeech-16k.onnx" \
  --vocoder-model /home/oem/Basir/Vocoder/Trained/vocos-matcha-16KHz-80mels-v1/vocos-matcha-16khz-80mels.onnx \
  --symbols "$FORK/exports/symbols-16k.txt" \
  --lexicon-dir "$GS/lex_current" \
  --sample-rate 16000 \
  --espeak-data "$ESPEAK" \
  --ezafe-onnx ./assets/ezafe_model.onnx \
  --ezafe-spiece ./assets/ezafe_spiece.model \
  --homograph ./assets/homograph_data.json \
  --shakkelha ./assets/shakkelha.onnx \
  --output "$OUT" --debug
