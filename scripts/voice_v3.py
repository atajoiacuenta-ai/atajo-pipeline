"""Voz por cláusulas con pausas controladas. Uso: python scripts/voice_v3.py <variante>"""
import subprocess, sys, os, json, numpy as np, soundfile as sf
VAR = sys.argv[1]
PARAMS = {  # length_scale, noise_scale, noise_w
 "estable":   (1.06, 0.50, 0.60),
 "expresiva": (1.00, 0.80, 0.90),
 "base":      (1.00, 0.667, 0.80),
}[VAR]
MODEL = "models/piper/es_ES-sharvard-medium.onnx"; SPK = 0; SR = 22050
CLAUSES = [  # (texto para el motor, texto en pantalla, pausa después)
 ("Si tienes un negocio,", "Si tienes un negocio,", 0.16),
 ("estás perdiendo horas cada semana en tareas que una I.A. puede hacer por ti.", "estás perdiendo horas cada semana en tareas que una IA puede hacer por ti.", 0.55),
 ("Primera: contestar siempre las mismas preguntas.", "Primera: contestar siempre las mismas preguntas.", 0.32),
 ("Horarios, precios, cómo llegar.", "Horarios, precios, cómo llegar.", 0.45),
 ("Un asistente con inteligencia artificial responde al instante,", "Un asistente con inteligencia artificial responde al instante,", 0.16),
 ("y tú solo entras cuando de verdad hace falta.", "y tú solo entras cuando de verdad hace falta.", 0.55),
 ("Segunda: los recordatorios.", "Segunda: los recordatorios.", 0.30),
 ("Un mensaje automático el día antes evita que os dejen plantados.", "Un mensaje automático el día antes evita que os dejen plantados.", 0.55),
 ("Y tercera: los presupuestos.", "Y tercera: los presupuestos.", 0.30),
 ("La I.A. lee el correo del cliente y te deja el borrador listo para revisar.", "La IA lee el correo del cliente y te deja el borrador listo para revisar.", 0.55),
 ("Nada de esto exige saber programar.", "Nada de esto exige saber programar.", 0.38),
 ("Guarda este vídeo y sígueme para más atajos con I.A.", "Guarda este vídeo y sígueme para más atajos con IA.", 0.0),
]
OUT = f"out/voice-v3/{VAR}"; os.makedirs(OUT, exist_ok=True)
def synth(i, text):
    wav = f"{OUT}/c{i}.wav"
    cmd = ["./bin/piper/piper", "--model", MODEL, "--speaker", str(SPK), "--length_scale", str(PARAMS[0]),
           "--noise_scale", str(PARAMS[1]), "--noise_w", str(PARAMS[2]), "--output_file", wav]
    subprocess.run(cmd, input=text.encode(), check=True, stderr=subprocess.DEVNULL)
    a, sr = sf.read(wav); assert sr == SR; a = a.astype("float32")
    idx = np.where(np.abs(a) > 0.012 * np.abs(a).max())[0]; m = int(0.03 * SR)
    a = a[max(0, idx[0] - m): idx[-1] + m]
    f = int(0.012 * SR); a[:f] *= np.linspace(0, 1, f); a[-f:] *= np.linspace(1, 0, f); return a
LEAD, TAIL = 0.15, 0.9
parts = [np.zeros(int(LEAD * SR), "float32")]; t = LEAD; times = []
for i, (tts, shown, pause) in enumerate(CLAUSES):
    a = synth(i, tts); d = len(a) / SR; times.append((t, t + d)); parts += [a, np.zeros(int(pause * SR), "float32")]; t += d + pause
parts.append(np.zeros(int(TAIL * SR), "float32")); full = np.concatenate(parts); DUR = len(full) / SR
sf.write(f"{OUT}/voice.wav", full, SR)
json.dump({"variant": VAR, "duration": DUR, "times": times, "texts": [c[1] for c in CLAUSES]}, open(f"{OUT}/meta.json", "w"), ensure_ascii=False)
print(VAR, round(DUR, 1), "s", [round(b - a, 1) for a, b in times])
