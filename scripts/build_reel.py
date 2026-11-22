"""Reel de prueba: voz Piper por frase -> tiempos exactos -> subtítulos ASS karaoke -> MP4 9:16 con ffmpeg.
Uso: python scripts/build_reel.py <nombre_voz> <modelo.onnx> <speaker|-1>"""
import subprocess, sys, os, wave, json, random
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFilter

NAME, MODEL, SPK = sys.argv[1], sys.argv[2], int(sys.argv[3])
OUT = "out/reel-test"; os.makedirs(OUT, exist_ok=True)
TMP = f"{OUT}/tmp_{NAME}"; os.makedirs(TMP, exist_ok=True)
SR = 22050
W, H = 1080, 1920

# (etiqueta, texto)
SEGS = [
 ("",                  "Si tienes un negocio, estás perdiendo horas cada semana en tareas que una IA puede hacer por ti."),
 ("1 · WHATSAPP",      "Primera: contestar siempre las mismas preguntas. Horarios, precios, cómo llegar."),
 ("1 · WHATSAPP",      "Un asistente con inteligencia artificial responde al instante, y tú solo entras cuando de verdad hace falta."),
 ("2 · CITAS",         "Segunda: los recordatorios. Un mensaje automático el día antes evita que os dejen plantados."),
 ("3 · PRESUPUESTOS",  "Y tercera: los presupuestos. La IA lee el correo del cliente y te deja el borrador listo para revisar."),
 ("",                  "Nada de esto exige saber programar. Guarda este vídeo y sígueme para más atajos con IA."),
]
PAUSE = [0.30, 0.20, 0.35, 0.35, 0.35, 0.0]
LEAD, TAIL = 0.15, 0.9

def synth(i, text):
    wav = f"{TMP}/s{i}.wav"
    cmd = ["./bin/piper/piper", "--model", MODEL, "--output_file", wav]
    if SPK >= 0: cmd += ["--speaker", str(SPK)]
    subprocess.run(cmd, input=text.encode(), check=True, stderr=subprocess.DEVNULL)
    a, sr = sf.read(wav); assert sr == SR, sr
    return a.astype("float32")

audio = [np.zeros(int(LEAD * SR), dtype="float32")]
times = []; t = LEAD
for i, (lab, txt) in enumerate(SEGS):
    a = synth(i, txt); d = len(a) / SR
    times.append((t, t + d)); audio += [a, np.zeros(int(PAUSE[i] * SR), dtype="float32")]
    t += d + PAUSE[i]
audio.append(np.zeros(int(TAIL * SR), dtype="float32"))
full = np.concatenate(audio); DUR = len(full) / SR
sf.write(f"{TMP}/voice.wav", full, SR)
print(NAME, "duración total", round(DUR, 1), "s")

# --- ASS ---
def ts(x):
    h = int(x // 3600); m = int(x % 3600 // 60); s = x % 60
    return f"{h}:{m:02d}:{s:05.2f}"
ass = [f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Main,Poppins,84,&H0000D7FF,&H00FFFFFF,&H00201008,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,90,90,0,1
Style: Tag,Poppins,46,&H00201008,&H00201008,&H0000D7FF,&H00000000,-1,0,0,0,100,100,2,0,3,18,0,8,60,60,330,1
Style: Brand,Poppins,40,&H00FFFFFF,&H00FFFFFF,&H00201008,&H00000000,-1,0,0,0,100,100,1,0,1,3,0,2,60,60,430,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text"""]
for (lab, txt), (a, b) in zip(SEGS, times):
    words = txt.split(); wts = [len(w) + 2 for w in words]; tot = sum(wts)
    span = (b - a) * 100
    parts = [f"{{\\kf{max(1, round(span * w / tot))}}}{wd}" for wd, w in zip(words, wts)]
    pop = "{\\fscx90\\fscy90\\t(0,140,\\fscx100\\fscy100)}"
    ass.append(f"Dialogue: 0,{ts(a)},{ts(b + 0.05)},Main,,0,0,0,,{pop}{' '.join(parts)}")
    if lab: ass.append(f"Dialogue: 1,{ts(a)},{ts(b + 0.05)},Tag,,0,0,0,,{lab}")
ass.append(f"Dialogue: 2,{ts(0)},{ts(DUR)},Brand,,0,0,0,,Atajo | IA para negocios")
open(f"{TMP}/subs.ass", "w", encoding="utf-8").write("\n".join(ass))

# --- fondo (degradado vertical con formas suaves; se desplaza despacio) ---
BH = 2700
bg = Image.new("RGB", (W, BH)); px = ImageDraw.Draw(bg)
c0, c1 = (12, 20, 48), (22, 78, 160)
for y in range(BH):
    k = y / BH; px.line([(0, y), (W, y)], fill=tuple(int(c0[j] + (c1[j] - c0[j]) * k) for j in range(3)))
ov = Image.new("RGBA", (W, BH), (0, 0, 0, 0)); od = ImageDraw.Draw(ov); random.seed(7)
for _ in range(14):
    r = random.randint(120, 380); x = random.randint(-100, W); y = random.randint(0, BH)
    od.ellipse([x - r, y - r, x + r, y + r], fill=(255, 215, 0, random.randint(10, 26)))
ov = ov.filter(ImageFilter.GaussianBlur(40)); bg = Image.alpha_composite(bg.convert("RGBA"), ov).convert("RGB")
bg.save(f"{TMP}/bg.png")

# --- ffmpeg ---
fc = (f"[0:v]crop={W}:{H}:0:'(ih-{H})*t/{DUR:.3f}',format=yuv420p[a];"
      f"[1:v][a]overlay=x='-w+w*t/{DUR:.3f}':y={H-70}:shortest=0[b];"
      f"[b]ass={TMP}/subs.ass:fontsdir=assets/fonts[v]")
# la barra de progreso es el input 1 (color)
cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
       "-loop", "1", "-framerate", "30", "-t", f"{DUR:.3f}", "-i", f"{TMP}/bg.png",
       "-f", "lavfi", "-t", f"{DUR:.3f}", "-i", f"color=c=0xFFD700:s={W}x12:r=30",
       "-i", f"{TMP}/voice.wav",
       "-filter_complex", fc.replace("[1:v][a]overlay", "[a][1:v]overlay"),
       "-map", "[v]", "-map", "2:a",
       "-af", "loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
       "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "1", "-movflags", "+faststart",
       f"{OUT}/reel_{NAME}.mp4"]
subprocess.run(cmd, check=True)
json.dump({"name": NAME, "duration": DUR, "times": times}, open(f"{TMP}/meta.json", "w"))
print("ok", f"{OUT}/reel_{NAME}.mp4")
