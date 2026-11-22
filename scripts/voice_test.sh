#!/usr/bin/env bash
# Genera la frase de prueba con las voces Piper es_ES que haya en models/piper/
set -euo pipefail
cd "$(dirname "$0")/.."
P=./bin/piper/piper
F=out/voice-test/frase.txt
M=models/piper
gen(){ # nombre modelo [speaker]
  local out="out/voice-test/piper_$1.wav"; shift
  if [ $# -ge 2 ]; then $P --model "$1" --speaker "$2" --output_file "$out" < $F 2>/dev/null
  else $P --model "$1" --output_file "$out" < $F 2>/dev/null; fi
  echo "ok -> $out"
}
if [ -f $M/es_ES-davefx-medium.onnx ]; then gen davefx $M/es_ES-davefx-medium.onnx; fi
if [ -f $M/es_ES-miro-high.onnx ]; then gen miro_high $M/es_ES-miro-high.onnx; fi
if [ -f $M/es_ES-sharvard-medium.onnx ]; then
  gen sharvard_M $M/es_ES-sharvard-medium.onnx 0
  gen sharvard_F $M/es_ES-sharvard-medium.onnx 1
fi
