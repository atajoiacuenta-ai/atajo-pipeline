"""Transcribe audio con Whisper (sherpa-onnx) y compara con el texto esperado. Uso: asr_check.py <carpeta_variante>"""
import sys, glob, json, re, difflib, unicodedata, soundfile as sf, numpy as np, sherpa_onnx
D = "models/asr/sherpa-onnx-whisper-small"
rec = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=f"{D}/small-encoder.int8.onnx", decoder=f"{D}/small-decoder.int8.onnx",
        tokens=f"{D}/small-tokens.txt", language="es", task="transcribe", num_threads=4)
def norm(s):
    s = unicodedata.normalize("NFKD", s.lower()); s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("i.a.", "ia").replace("i. a.", "ia"); return re.findall(r"[a-z0-9ñ]+", s)
def transcribe(a, sr):
    st = rec.create_stream(); st.accept_waveform(sr, a.astype("float32")); rec.decode_stream(st); return st.result.text.strip()
def check(folder, texts):
    tot_err = tot_w = 0; bad = []
    for i, exp in enumerate(texts):
        a, sr = sf.read(f"{folder}/c{i}.wav"); a = np.concatenate([np.zeros(int(.3 * sr)), a, np.zeros(int(.3 * sr))])
        got = transcribe(a, sr); e, g = norm(exp), norm(got)
        sm = difflib.SequenceMatcher(None, e, g); err = sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")
        tot_err += err; tot_w += len(e)
        if err: bad.append((i, exp, got))
    return tot_err, tot_w, bad
if __name__ == "__main__":
    folder = sys.argv[1]; meta = json.load(open(f"{folder}/meta.json")); er, w, bad = check(folder, meta["texts"])
    print(f"{folder}: errores {er}/{w} palabras ({100 * er / w:.1f} %)")
    for i, exp, got in bad: print(f"  c{i}\n    esperado: {exp}\n    whisper:  {got}")
