"""Genera la frase de prueba con voces Piper es_ES (paquetes sherpa-onnx)."""
import sys, soundfile as sf, sherpa_onnx
txt = open("out/voice-test/frase.txt", encoding="utf-8").read().strip()
S = "models/sherpa/vits-piper-es_ES-"
jobs = [("davefx", "davefx-medium", 0), ("sharvard_M", "sharvard-medium", 0),
        ("sharvard_F", "sharvard-medium", 1), ("miro_high", "miro-high", 0)]
cache = {}
for name, pkg, sid in jobs:
    d = S + pkg
    if pkg not in cache:
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{d}/es_ES-{pkg}.onnx", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
            num_threads=4))
        cache[pkg] = sherpa_onnx.OfflineTts(cfg)
    a = cache[pkg].generate(txt, sid=sid, speed=1.0)
    out = f"out/voice-test/piper_{name}.wav"
    import numpy as np; sf.write(out, np.asarray(a.samples, dtype="float32"), a.sample_rate); print(out, round(len(a.samples)/a.sample_rate,1), "s")
