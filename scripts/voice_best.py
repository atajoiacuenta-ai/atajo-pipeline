"""Voz por cláusulas, N tomas por cláusula, verificación con Whisper y selección por claridad + variación tonal.
Uso: python scripts/voice_best.py <nombre> <length_scale> <noise_scale> <noise_w> <N> <escala_pausas>"""
import subprocess, sys, os, json, time, numpy as np, soundfile as sf
sys.path.insert(0, "scripts"); import asr_check as A
NAME, LS, NS, NW, N, PS = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6])
MODEL = "models/piper/es_ES-sharvard-medium.onnx"; SR = 22050; T0 = time.time(); BUDGET = 140
CLAUSES = [
 ("Si tienes un negocio,", 0.16),
 ("estás perdiendo horas cada semana en tareas que la inteligencia artificial puede hacer por ti.", 0.50),
 ("Primera: contestar siempre las mismas preguntas.", 0.30),
 ("Horarios, precios, cómo llegar.", 0.40),
 ("Un asistente con inteligencia artificial responde al instante,", 0.14),
 ("y tú solo entras cuando de verdad hace falta.", 0.50),
 ("Segunda: los recordatorios.", 0.28),
 ("Un mensaje automático el día antes evita que os dejen plantados.", 0.50),
 ("Y tercera: los presupuestos.", 0.28),
 ("El asistente lee el correo del cliente y te deja el borrador listo para revisar.", 0.50),
 ("Nada de esto exige saber programar.", 0.35),
 ("Guarda este vídeo y sígueme para más atajos con inteligencia artificial.", 0.0),
]
OUT = f"out/voice-v3/{NAME}"; CAND = f"{OUT}/cand"; os.makedirs(CAND, exist_ok=True)
SC = f"{OUT}/scores.json"; scores = json.load(open(SC)) if os.path.exists(SC) else {}

def trim(a):
    idx = np.where(np.abs(a) > 0.008 * np.abs(a).max())[0]; a = a[max(0, idx[0] - int(.04 * SR)): idx[-1] + int(.07 * SR)].copy()
    f = int(.02 * SR); a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f); return a
def f0_std(a):
    fr, hop, lo, hi = int(.04 * SR), int(.01 * SR), int(SR / 260), int(SR / 70); f = []
    for s in range(0, len(a) - fr, hop):
        w = a[s:s + fr] - a[s:s + fr].mean()
        if np.sqrt(np.mean(w ** 2)) < 0.02: continue
        ac = np.correlate(w, w, "full")[fr - 1:]; ac = ac / (ac[0] + 1e-9); seg = ac[lo:hi]; k = int(np.argmax(seg))
        if seg[k] > 0.45: f.append(SR / (k + lo))
    if len(f) < 10: return 0.0
    st = 12 * np.log2(np.array(f) / np.median(f)); st = st[np.abs(st) < 8]; return float(np.std(st))
def errors(exp, got):
    import difflib
    e, g = A.norm(exp), A.norm(got); sm = difflib.SequenceMatcher(None, e, g)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal")

for i, (text, _) in enumerate(CLAUSES):
    for k in range(N):
        key = f"{i}_{k}"
        if key in scores: continue
        if time.time() - T0 > BUDGET: json.dump(scores, open(SC, "w")); print("tiempo agotado, vuelve a ejecutar"); sys.exit(3)
        wav = f"{CAND}/c{i}_{k}.wav"
        subprocess.run(["./bin/piper/piper", "--model", MODEL, "--speaker", "0", "--length_scale", LS, "--noise_scale", NS, "--noise_w", NW, "--output_file", wav],
                       input=text.encode(), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        a, _ = sf.read(wav); a = trim(a.astype("float32")); pad = np.zeros(int(.3 * SR), "float32")
        got = A.transcribe(np.concatenate([pad, a, pad]), SR); er = errors(text, got)
        scores[key] = {"err": er, "f0": round(f0_std(a), 2), "dur": round(len(a) / SR, 2), "asr": got}
json.dump(scores, open(SC, "w"))
# --- selección y montaje ---
LEAD, TAIL = 0.15, 0.9; parts = [np.zeros(int(LEAD * SR), "float32")]; t = LEAD; times = []; report = []
for i, (text, pause) in enumerate(CLAUSES):
    best = max(range(N), key=lambda k: (1000 if scores[f"{i}_{k}"]["err"] == 0 else -100 * scores[f"{i}_{k}"]["err"]) + min(scores[f"{i}_{k}"]["f0"], 4.5))
    a, _ = sf.read(f"{CAND}/c{i}_{best}.wav"); a = trim(a.astype("float32")); d = len(a) / SR; ps = pause * PS
    times.append((t, t + d)); parts += [a, np.zeros(int(ps * SR), "float32")]; t += d + ps
    ok = sum(scores[f"{i}_{k}"]["err"] == 0 for k in range(N)); report.append((i, best, ok, scores[f"{i}_{best}"]))
parts.append(np.zeros(int(TAIL * SR), "float32")); full = np.concatenate(parts); DUR = len(full) / SR
sf.write(f"{OUT}/voice.wav", full, SR)
json.dump({"variant": NAME, "duration": DUR, "times": times, "texts": [c[0] for c in CLAUSES], "params": [LS, NS, NW, N, PS]}, open(f"{OUT}/meta.json", "w"), ensure_ascii=False)
words = sum(len(c[0].split()) for c in CLAUSES); speech = sum(b - a for a, b in times)
print(f"{NAME}: {DUR:.1f} s totales, habla {speech:.1f} s, {words / speech:.2f} palabras/s")
for i, best, ok, s in report: print(f"  c{i}: elegida toma {best} | tomas sin error {ok}/{N} | f0_std {s['f0']} | err {s['err']}")
