#!/usr/bin/env python3
"""Voz de los Reels con Chatterbox es-ES (referencia sharvard). Para cada guion JSON genera
out/voice-v3/<id>/{voice.wav,meta.json} (times por cláusula, duration, texts) para scripts/reel.py.
Uso: python scripts/voice_cb.py content/002.json content/003.json ...   (FAKE=1 -> tonos de prueba, sin modelo)
Carga el modelo una sola vez. Env: REF (wav de referencia), TEMPO (1.0 natural; 1.1 = 10% más rápido)"""
import sys, os, json, subprocess, time
import numpy as np, soundfile as sf

SR = 24000; LEAD = 0.15; TAIL = 0.35
EXAG = float(os.environ.get('EXAG', 0.5)); CFG = float(os.environ.get('CFG', 0.5)); TEMP = float(os.environ.get('TEMP', 0.8))

def esdrujulas(text):
    """Palabras con tilde en la antepenúltima sílaba (esdrújulas): el modelo las suele leer como llanas."""
    import re
    out = []
    for w in re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑüÜ]+", text):
        m = re.search(r"[áéíóú]", w.lower())
        if not m: continue
        rest = w.lower()[m.end():]
        nuclei = re.findall(r"(?:[iuü][aeo]|[aeo][iu]|[aeoiuü])(?=[^aeoiuüáéíóú]|$)|[aeo](?=[aeo])", rest)
        n = len(re.findall(r"[iuü]?[aeo][iu]?|[iuü]", rest))
        if n >= 2: out.append(w)
    return out
FAKE = bool(os.environ.get('FAKE'))
if not FAKE:
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    ref = os.environ.get('REF', 'refs/ref_sharvard.wav')
    sys.path.insert(0, os.environ.get('CB_SRC', 'space/chatterbox/src'))
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained('cpu')

def synth(text, path):
    if FAKE:
        n = int(SR * (0.35 + 0.055 * len(text))); t = np.arange(n) / SR
        w = 0.3 * np.sin(2 * np.pi * 150 * t) * np.minimum(1, np.minimum(t, t[::-1]) * 20); sr = SR
    else:
        t0 = time.time()
        w = model.generate(text[:300], audio_prompt_path=ref, exaggeration=EXAG, temperature=TEMP, cfg_weight=CFG, language_id='es')
        w = w.cpu().numpy().squeeze(); sr = model.sr
        print(f"  {time.time()-t0:.1f}s  {text}", flush=True)
    if sr != SR:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, SR).numpy()
    sf.write(path, w.astype('float32'), SR)

def trim_tempo(path_in, path_out, tempo):
    af = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
          "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
    if abs(tempo - 1.0) > 1e-3: af += f",atempo={tempo}"
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', path_in, '-af', af, '-ar', str(SR), '-ac', '1', path_out], check=True)

for spec_path in sys.argv[1:]:
    spec = json.load(open(spec_path)); name = spec['id']; clauses = spec['clauses']
    tempo = float(os.environ.get('TEMPO', spec.get('tempo', 1.0)))
    out = f"out/voice-v3/{name}"; os.makedirs(out + "/clips", exist_ok=True)
    print(f"== {name} ({len(clauses)} cláusulas)", flush=True)
    for c in clauses:
        bad = esdrujulas(c['text'])
        if bad: print("  AVISO esdrújulas (suelen leerse mal):", bad, "en:", c['text'], flush=True)
    pieces = [np.zeros(int(LEAD * SR), dtype='float32')]; times = []; cur = LEAD
    for i, c in enumerate(clauses):
        f = f"{out}/clips/{i:02d}.wav"
        if not os.path.exists(f):
            synth(c['text'], f"{out}/clips/raw{i:02d}.wav"); trim_tempo(f"{out}/clips/raw{i:02d}.wav", f, tempo)
        w, _ = sf.read(f); w = w.astype('float32')
        times.append([round(cur, 3), round(cur + len(w) / SR, 3)])
        pieces.append(w); cur += len(w) / SR
        gap = c['pause'] if i < len(clauses) - 1 else TAIL
        pieces.append(np.zeros(int(gap * SR), dtype='float32')); cur += gap
    voice = np.concatenate(pieces)
    sf.write(f"{out}/voice.wav", voice, SR)
    json.dump({"variant": name, "duration": len(voice) / SR, "times": times, "texts": [c['text'] for c in clauses],
               "params": {"engine": "chatterbox_es", "tempo": tempo}}, open(f"{out}/meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"voz lista {name}: {len(voice)/SR:.1f}s")
