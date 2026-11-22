#!/usr/bin/env python3
"""Banco de pruebas de voces (GitHub Actions, CPU). No publica nada.
Uso: python scripts/voice_lab.py <engine> <outdir>
engines:
  edge[:+15%]      voz es-ES-AlvaroNeural con velocidad relativa (linea base, servicio no oficial)
  chatterbox_es    finetune oficial es-ES; voz de referencia = REF_F (muestra oficial es-ES)
  chatterbox_es_m  idem con muestra masculina oficial REF_M
Env: REF_F, REF_M (rutas wav), GAP (segundos entre frases en full.wav)"""
import sys, os, json, asyncio, subprocess, traceback, time
import numpy as np, soundfile as sf

engine, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
clauses = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'samples', 'voice_clauses.json')))
sr_out = 24000
GAP = float(os.environ.get('GAP', '0.12'))
chunks = []

def save(i, wav, sr):
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if sr != sr_out:
        import torch, torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, sr_out).numpy()
    sf.write(f"{out}/{i:02d}.wav", wav, sr_out)
    chunks.append(wav)

def run_chatterbox(ref):
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    if not os.path.exists(ref):
        raise SystemExit(f"falta la muestra de referencia {ref}")
    # El código del Space oficial tiene que ir primero en sys.path: asi no lo tapa el paquete pip 'chatterbox'
    sys.path.insert(0, os.environ.get('CB_SRC', 'space/chatterbox/src'))
    from chatterbox.tts import ChatterboxTTS
    import chatterbox
    print('chatterbox desde', chatterbox.__file__)
    model = ChatterboxTTS.from_pretrained('cpu')
    for i, t in enumerate(clauses):
        t0 = time.time()
        w = model.generate(t[:300], audio_prompt_path=ref, exaggeration=0.5,
                           temperature=0.8, cfg_weight=0.5, language_id='es')
        save(i, w.cpu().numpy(), model.sr)
        print(i, f"{time.time()-t0:.1f}s", t, flush=True)

def run_edge(rate):
    import edge_tts
    async def go():
        for i, t in enumerate(clauses):
            f = f"{out}/{i:02d}.mp3"
            await edge_tts.Communicate(t, 'es-ES-AlvaroNeural', rate=rate).save(f)
            trim = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                    "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f, '-af', trim,
                            '-ar', str(sr_out), '-ac', '1', f"{out}/{i:02d}.wav"], check=True)
            w, sr = sf.read(f"{out}/{i:02d}.wav"); chunks.append(w.astype(np.float32))
    asyncio.run(go())

try:
    name, _, arg = engine.partition(':')
    if name == 'edge':
        run_edge(arg or '+0%')
    elif name == 'chatterbox_es':
        run_chatterbox(os.environ.get('REF_F', 'refs/es_es_f1.wav'))
    elif name == 'chatterbox_es_m':
        run_chatterbox(os.environ.get('REF_M', 'refs/es_m1.flac'))
    else:
        raise SystemExit('motor desconocido: ' + engine)
finally:
    if chunks:
        gap = np.zeros(int(sr_out * GAP), dtype=np.float32)
        full = np.concatenate([x for c in chunks for x in (c, gap)])
        sf.write(f"{out}/full.wav", full, sr_out)
        print('full.wav', round(len(full) / sr_out, 1), 's')
