#!/usr/bin/env python3
"""Voz del Reel con Chatterbox es-ES (referencia sharvard). Genera out/voice-v3/<nombre>/{voice.wav,meta.json}
con el mismo formato que usa reel_v3.py (times por clausula, duration, texts).
Uso: python scripts/voice_cb.py content/001.json <nombre>      (FAKE=1 -> tonos de prueba, sin modelo)
Env: REF (wav de referencia), TEMPO (1.0 = natural; 1.1 = 10% mas rapido)"""
import sys, os, json, subprocess, time
import numpy as np, soundfile as sf

spec = json.load(open(sys.argv[1])); name = sys.argv[2]
tempo = float(os.environ.get('TEMPO', spec.get('tempo', 1.0)))
SR = 24000; LEAD = 0.15; TAIL = 0.35
out = f"out/voice-v3/{name}"; os.makedirs(out + "/clips", exist_ok=True)
clauses = spec['clauses']

def trim_tempo(path_in, path_out):
    af = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
          "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
    if abs(tempo - 1.0) > 1e-3: af += f",atempo={tempo}"
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path_in, '-af', af, '-ar', str(SR), '-ac', '1', path_out], check=True)

if os.environ.get('FAKE'):
    def gen(i, text):
        n = int(SR * (0.35 + 0.055 * len(text)))
        t = np.arange(n) / SR
        w = 0.3 * np.sin(2 * np.pi * 150 * t) * np.minimum(1, np.minimum(t, t[::-1]) * 20)
        sf.write(f"{out}/clips/raw{i:02d}.wav", w.astype('float32'), SR)
else:
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    ref = os.environ.get('REF', 'refs/ref_sharvard.wav')
    sys.path.insert(0, os.environ.get('CB_SRC', 'space/chatterbox/src'))
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained('cpu')
    def gen(i, text):
        t0 = time.time()
        w = model.generate(text[:300], audio_prompt_path=ref, exaggeration=0.5, temperature=0.8, cfg_weight=0.5, language_id='es')
        w = w.cpu().numpy().squeeze()
        if model.sr != SR:
            import torchaudio
            w = torchaudio.functional.resample(torch.from_numpy(w), model.sr, SR).numpy()
        sf.write(f"{out}/clips/raw{i:02d}.wav", w.astype('float32'), SR)
        print(i, f"{time.time()-t0:.1f}s", text, flush=True)

pieces = [np.zeros(int(LEAD * SR), dtype='float32')]; times = []; cur = LEAD
for i, c in enumerate(clauses):
    f = f"{out}/clips/{i:02d}.wav"
    if not os.path.exists(f):
        gen(i, c['text']); trim_tempo(f"{out}/clips/raw{i:02d}.wav", f)
    w, _ = sf.read(f); w = w.astype('float32')
    times.append([round(cur, 3), round(cur + len(w) / SR, 3)])
    pieces.append(w); cur += len(w) / SR
    gap = c['pause'] if i < len(clauses) - 1 else TAIL
    pieces.append(np.zeros(int(gap * SR), dtype='float32')); cur += gap
voice = np.concatenate(pieces)
sf.write(f"{out}/voice.wav", voice, SR)
json.dump({"variant": name, "duration": len(voice) / SR, "times": times, "texts": [c['text'] for c in clauses],
           "params": {"engine": "chatterbox_es", "tempo": tempo}}, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
print(f"voz lista: {len(voice)/SR:.1f}s, {len(clauses)} cláusulas")
