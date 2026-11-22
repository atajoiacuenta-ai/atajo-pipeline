#!/usr/bin/env python3
"""Banco de pruebas de voces (se ejecuta en GitHub Actions, CPU).
Uso: python scripts/voice_lab.py <engine> <outdir> [ref_wav]
engines: chatterbox_es (finetune es-ES), chatterbox_mtl (modelo multilingue general), edge (linea base)
Genera una frase por cláusula + full.wav. No publica nada."""
import sys, os, json, asyncio, subprocess, traceback, time
import numpy as np, soundfile as sf

engine, out = sys.argv[1], sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None
os.makedirs(out, exist_ok=True)
clauses = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'samples', 'voice_clauses.json')))
sr_out = 24000
chunks = []

def save(i, wav, sr):
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    if sr != sr_out:
        import torch, torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, sr_out).numpy()
    sf.write(f"{out}/{i:02d}.wav", wav, sr_out)
    chunks.append(wav)

def run_chatterbox(kind):
    import torch
    torch.set_num_threads(os.cpu_count() or 2)
    if kind == 'es':
        from huggingface_hub import snapshot_download
        sd = snapshot_download('ResembleAI/Chatterbox-Multilingual-es-es')
        print('es-es files:', os.listdir(sd))
        # Código vendorizado del Space oficial (clonado por el workflow)
        sys.path.insert(0, os.environ.get('CB_SPACE', 'space'))
        from chatterbox.src.chatterbox.tts import ChatterboxTTS
        try:
            model = ChatterboxTTS.from_local(sd, 'cpu')
        except Exception:
            traceback.print_exc()
            model = ChatterboxTTS.from_pretrained('cpu')
    else:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
    for i, t in enumerate(clauses):
        t0 = time.time()
        kw = dict(language_id='es', exaggeration=0.5, temperature=0.8, cfg_weight=0.5)
        if ref: kw['audio_prompt_path'] = ref
        w = model.generate(t[:300], **kw)
        save(i, w.cpu().numpy() if hasattr(w, 'cpu') else w, model.sr)
        print(i, f"{time.time()-t0:.1f}s", t, flush=True)

def run_edge():
    import edge_tts
    async def go():
        for i, t in enumerate(clauses):
            f = f"{out}/{i:02d}.mp3"
            await edge_tts.Communicate(t, 'es-ES-AlvaroNeural', rate='+0%').save(f)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f, '-ar', str(sr_out), '-ac', '1', f"{out}/{i:02d}.wav"], check=True)
            w, sr = sf.read(f"{out}/{i:02d}.wav"); chunks.append(w.astype(np.float32))
    asyncio.run(go())

try:
    {'chatterbox_es': lambda: run_chatterbox('es'),
     'chatterbox_mtl': lambda: run_chatterbox('mtl'),
     'edge': run_edge}[engine]()
finally:
    if chunks:
        gap = np.zeros(int(sr_out * 0.30), dtype=np.float32)
        full = np.concatenate([x for c in chunks for x in (c, gap)])
        sf.write(f"{out}/full.wav", full, sr_out)
        print('full.wav', len(full) / sr_out, 's')
