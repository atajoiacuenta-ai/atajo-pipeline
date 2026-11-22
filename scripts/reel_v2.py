"""Reel v2: escenas animadas dibujadas con PIL, un MP4 por escena, concat -c copy, sonido al final.
Uso: python scripts/reel_v2.py preview 1 4 9 ...   |   python scripts/reel_v2.py render"""
import json, os, sys, math, random, subprocess
from functools import lru_cache
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

W, H, FPS = 1080, 1920, 30
SRC = "out/reel-test/tmp_sharvard_M"
OUTD = "out/reel-v2"; os.makedirs(OUTD + "/scenes", exist_ok=True)
meta = json.load(open(f"{SRC}/meta.json")); TIMES = meta["times"]; DUR = meta["duration"]
SEGS = [
 "Si tienes un negocio, estás perdiendo horas cada semana en tareas que una IA puede hacer por ti.",
 "Primera: contestar siempre las mismas preguntas. Horarios, precios, cómo llegar.",
 "Un asistente con inteligencia artificial responde al instante, y tú solo entras cuando de verdad hace falta.",
 "Segunda: los recordatorios. Un mensaje automático el día antes evita que os dejen plantados.",
 "Y tercera: los presupuestos. La IA lee el correo del cliente y te deja el borrador listo para revisar.",
 "Nada de esto exige saber programar. Guarda este vídeo y sígueme para más atajos con IA.",
]
LABELS = ["TAREAS QUE TE ROBAN TIEMPO", "1 · WHATSAPP", "2 · CITAS", "3 · PRESUPUESTOS", ""]
mid = lambda i: (TIMES[i][1] + TIMES[i + 1][0]) / 2
BOUNDS = [0, mid(0), mid(2), mid(3), mid(4), DUR]
FR = [round(b * FPS) for b in BOUNDS]; NF = FR[-1]

YEL = (255, 215, 0); INK = (16, 24, 48); WHITE = (255, 255, 255); RED = (239, 68, 68)
GREEN = (22, 163, 120); GRAY = (203, 211, 225); MUTED = (100, 112, 140)

# ---------- utilidades ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def e_out(x): return 1 - (1 - x) ** 3
def e_back(x): c1 = 1.70158; c3 = c1 + 1; return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def ease(t, t0, d, f=e_out): return f(clamp((t - t0) / d))
def F(sz, w="Bold"): return ImageFont.truetype(f"assets/fonts/Poppins-{w}.ttf", sz)

@lru_cache(None)
def tspr(s, size, weight, color, anchor, stroke):
    f = F(size, weight); d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = d0.textbbox((0, 0), s, font=f, anchor=anchor, stroke_width=stroke)
    pad = stroke + 4; im = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((-l + pad, -t + pad), s, font=f, fill=color + (255,), anchor=anchor,
                            stroke_width=stroke, stroke_fill=(10, 14, 30, 255))
    return im, -l + pad, -t + pad

def twidth(s, size, weight="Medium"): return int(F(size, weight).getlength(s))

def paste(cv, im, X, Y, a=1.0):
    if a <= 0.01: return
    if a < 1: im = im.copy(); im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
    cv.paste(im, (int(X), int(Y)), im)

def blit(cv, im, cx, cy, a=1.0, s=1.0):
    if s != 1.0: im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    paste(cv, im, cx - im.width / 2, cy - im.height / 2, a)

def text(cv, s, size, weight, color, x, y, anchor="mm", a=1.0, stroke=0):
    im, ox, oy = tspr(s, size, weight, color, anchor, stroke); paste(cv, im, x - ox, y - oy, a)

@lru_cache(None)
def card(w, h, r, fill, shadow=True):
    S, pad = 2, 28 if shadow else 2
    im = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    if shadow:
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([pad * S, (pad + 8) * S, (pad + w) * S, (pad + h + 8) * S], r * S, fill=(0, 0, 0, 95))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(11 * S)))
    ImageDraw.Draw(im).rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], r * S, fill=fill)
    return im.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)

@lru_cache(None)
def shape(kind, size, color):
    S = 4; n = size * S; im = Image.new("RGBA", (n, n), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = color + (255,)
    if kind == "circle": d.ellipse([0, 0, n - 1, n - 1], fill=c)
    elif kind == "check":
        d.line([(n * .2, n * .55), (n * .42, n * .75), (n * .8, n * .28)], fill=c, width=int(n * .14), joint="curve")
        for p in [(n * .2, n * .55), (n * .8, n * .28)]: d.ellipse([p[0] - n * .07, p[1] - n * .07, p[0] + n * .07, p[1] + n * .07], fill=c)
    elif kind == "star":
        pts = []
        for i in range(8):
            r = n / 2 if i % 2 == 0 else n * .12; a = math.pi / 4 * i - math.pi / 2
            pts.append((n / 2 + r * math.cos(a), n / 2 + r * math.sin(a)))
        d.polygon(pts, fill=c)
    elif kind == "bell":
        d.ellipse([n * .2, n * .12, n * .8, n * .72], fill=c); d.polygon([(n * .2, n * .45), (n * .8, n * .45), (n * .9, n * .78), (n * .1, n * .78)], fill=c)
        d.ellipse([n * .4, n * .8, n * .6, n * .96], fill=c); d.ellipse([n * .44, n * .02, n * .56, n * .16], fill=c)
    elif kind == "pin":
        d.ellipse([n * .2, n * .05, n * .8, n * .65], fill=c); d.polygon([(n * .27, n * .52), (n * .73, n * .52), (n * .5, n * .97)], fill=c)
        d.ellipse([n * .38, n * .23, n * .62, n * .47], fill=(255, 255, 255, 255))
    elif kind == "bookmark":
        d.polygon([(n * .22, n * .05), (n * .78, n * .05), (n * .78, n * .95), (n * .5, n * .72), (n * .22, n * .95)], fill=c)
    elif kind == "bolt":
        d.polygon([(n * .58, n * .02), (n * .2, n * .56), (n * .46, n * .56), (n * .38, n * .98), (n * .8, n * .4), (n * .54, n * .4)], fill=c)
    return im.resize((size, size), Image.LANCZOS)

@lru_cache(None)
def pill(label, size, fill, fg, padx=34, pady=14, weight="Bold"):
    f = F(size, weight); w = int(f.getlength(label)) + 2 * padx; h = size + 2 * pady + 6
    base = card(w, h, h // 2, fill + (255,) if len(fill) == 3 else fill, shadow=True)
    im = base.copy(); t, ox, oy = tspr(label, size, weight, fg, "mm", 0)
    im.paste(t, (int(im.width / 2 - ox), int(im.height / 2 - oy)), t); return im

def chip(cv, t, label):
    if not label: return
    p = ease(t, 0.05, 0.4); blit(cv, pill(label, 38, YEL, INK), 540, 300 - (1 - p) * 34, p)

@lru_cache(None)
def clock_face(R):
    S = 2; n = (2 * R + 40) * S; im = Image.new("RGBA", (n, n), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = n // 2
    d.ellipse([c - R * S - 14, c - R * S - 14, c + R * S + 14, c + R * S + 14], fill=YEL + (255,))
    d.ellipse([c - R * S, c - R * S, c + R * S, c + R * S], fill=(255, 255, 255, 255))
    for i in range(12):
        a = math.pi / 6 * i; r0 = R * S * (0.82 if i % 3 else 0.74)
        d.line([(c + r0 * math.sin(a), c - r0 * math.cos(a)), (c + R * S * .92 * math.sin(a), c - R * S * .92 * math.cos(a))], fill=INK + (255,), width=6 * S // 2)
    return im.resize((n // S, n // S), Image.LANCZOS)

def hand(cv, cx, cy, ang, L, wd, color):
    n = L * 2 + 20; im = Image.new("RGBA", (n * 2, n * 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([n - wd, n - L * 2, n + wd, n + wd], wd, fill=color + (255,))
    im = im.rotate(-ang, Image.BICUBIC).resize((n, n), Image.LANCZOS); paste(cv, im, cx - n / 2, cy - n / 2)

@lru_cache(None)
def bg_full():
    BH = 2800; yy = np.linspace(0, 1, BH)[:, None, None]
    c0 = np.array([10, 16, 40]); c1 = np.array([20, 82, 170])
    arr = (c0 + (c1 - c0) * yy).repeat(W, 1).astype("uint8"); im = Image.fromarray(arr).convert("RGBA")
    ov = Image.new("RGBA", (W, BH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); random.seed(11)
    for _ in range(16):
        r = random.randint(140, 400); x = random.randint(-100, W); y = random.randint(0, BH)
        d.ellipse([x - r, y - r, x + r, y + r], fill=((255, 215, 0) if random.random() < .5 else (90, 170, 255)) + (random.randint(12, 30),))
    return Image.alpha_composite(im, ov.filter(ImageFilter.GaussianBlur(45))).convert("RGB")

def bg_at(T):
    y = int((2800 - H) * T / DUR); return bg_full().crop((0, y, W, y + H))

# ---------- subtítulos ----------
SUBSZ, SUBW = 68, 920
@lru_cache(None)
def sub_chunks(i):
    words = SEGS[i].split(); a, b = TIMES[i]; wts = [len(w) + 2 for w in words]; tot = sum(wts); acc = a; ws = []
    for w, k in zip(words, wts): ws.append((w, acc)); acc += (b - a) * k / tot
    chunks, lines, cur = [], [], []
    def fits(c): return twidth(" ".join(x[0] for x in c), SUBSZ, "Bold") <= SUBW
    for w in ws:
        if fits(cur + [w]): cur = cur + [w]
        elif len(lines) == 0: lines.append(cur); cur = [w]
        else: lines.append(cur); chunks.append(lines); lines, cur = [], [w]
    lines.append(cur); chunks.append(lines)
    out = []
    for n, ch in enumerate(chunks):
        st = ch[0][0][1]; en = chunks[n + 1][0][0][1] if n + 1 < len(chunks) else b + 0.12
        out.append((st, en, ch))
    return out

def subs(cv, T):
    for i in range(len(SEGS)):
        for st, en, lines in sub_chunks(i):
            if st - 0.02 <= T < en:
                p = ease(T, st - 0.02, 0.16); y0 = 1290 - (len(lines) - 1) * 46
                for li, line in enumerate(lines):
                    widths = [twidth(w, SUBSZ, "Bold") for w, _ in line]; sp = 20; total = sum(widths) + sp * (len(line) - 1); x = 540 - total / 2
                    for (w, wst), wd in zip(line, widths):
                        im, ox, oy = tspr(w, SUBSZ, "Bold", YEL if T >= wst else WHITE, "lm", 6)
                        paste(cv, im, x - ox, y0 + li * 92 + (1 - p) * 22 - oy, p); x += wd + sp
                return

def overlay_ui(cv, T):
    text(cv, "Atajo | IA para negocios", 32, "SemiBold", WHITE, 540, 1490, a=0.75)
    d = ImageDraw.Draw(cv); d.rectangle([0, 1884, W, 1892], fill=(255, 255, 255)) if False else None
    d.rounded_rectangle([60, 1884, W - 60, 1892], 4, fill=(60, 80, 130)); d.rounded_rectangle([60, 1884, 60 + (W - 120) * T / DUR, 1892], 4, fill=YEL)

# ---------- ESCENA A: hook ----------
def tl_A(d): return {"cards": [0.9 + 1.05 * k for k in range(4)]}
def scene_A(cv, t, d):
    tl = tl_A(d); chip(cv, t, LABELS[0])
    blit(cv, clock_face(130), 540, 530); ang = t * 540
    hand(cv, 540, 530, ang, 105, 8, INK); hand(cv, 540, 530, ang / 12, 70, 11, INK); blit(cv, shape("circle", 30, INK), 540, 530)
    for k, (ti, title) in enumerate(zip(tl["cards"], ["Contestar mensajes", "Confirmar citas", "Hacer presupuestos", "Reclamar facturas"])):
        p = ease(t, ti, 0.5, e_back); a = clamp((t - ti) / 0.2); cx = 540 + (1 - p) * 700; cy = 770 + k * 112
        blit(cv, card(880, 100, 30, (255, 255, 255, 238)), cx, cy, a)
        blit(cv, card(60, 60, 18, RED + (255,), shadow=False), cx - 440 + 64, cy, a); text(cv, "!", 40, "Bold", WHITE, cx - 440 + 64, cy - 1, a=a)
        text(cv, title, 38, "SemiBold", INK, cx - 440 + 120, cy, "lm", a)
        blit(cv, pill("se repite", 26, (254, 226, 226), RED, 22, 8), cx + 440 - 110, cy, a)

# ---------- ESCENA B: WhatsApp ----------
MSGS = [("c", "¿Qué horario tenéis?"), ("a", "Abrimos de 9:00 a 20:00"), ("c", "¿Cuánto cuesta?"),
        ("a", "Te paso la tarifa ahora mismo"), ("c", "¿Cómo llego?"), ("a", "Aquí tienes la ubicación")]
def tl_B(d): return {"msg": [0.7, 1.6, 2.5, 3.4, 4.3, 5.2], "typ": [(1.1, 1.6), (2.9, 3.4), (4.7, 5.2)], "note": 8.4, "win": 0.0}
@lru_cache(None)
def chat_window():
    S, w, h, r, pad = 2, 920, 660, 36, 28
    im = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([pad * S, (pad + 10) * S, (pad + w) * S, (pad + h + 10) * S], r * S, fill=(0, 0, 0, 100))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(12 * S)))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    d.rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], fill=(232, 238, 246, 255)); d.rectangle([pad * S, pad * S, (pad + w) * S, (pad + 100) * S], fill=GREEN + (255,))
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], r * S, fill=255)
    im.paste(layer, (0, 0), m); return im.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
@lru_cache(None)
def bubble(w, kind): return card(w, 74, 26, (255, 255, 255, 255) if kind == "c" else (205, 245, 224, 255), shadow=False)
def scene_B(cv, t, d):
    tl = tl_B(d); chip(cv, t, LABELS[1]); p = ease(t, 0.0, 0.5, e_back); cx, cy = 540, 730 + (1 - p) * 120
    blit(cv, chat_window(), cx, cy, clamp(t / 0.25), 0.94 + 0.06 * p)
    if t < 0.3: return
    top = 730 - 330; blit(cv, shape("circle", 58, (255, 255, 255)), 130, top + 52)
    text(cv, "T", 32, "Bold", GREEN, 130, top + 52); text(cv, "Tu negocio", 34, "Bold", WHITE, 180, top + 38, "lm"); text(cv, "asistente con IA · en línea", 24, "Medium", (214, 247, 233), 180, top + 74, "lm")
    blit(cv, pill("IA", 22, YEL, INK, 18, 6), 940, top + 52)
    for k, (kind, s) in enumerate(MSGS):
        ts = tl["msg"][k]
        if t < ts: continue
        pp = ease(t, ts, 0.3, e_back); a = clamp((t - ts) / 0.15); y = top + 150 + k * 82 + (1 - pp) * 18
        bw = twidth(s, 32, "Medium") + 56 + (44 if k == 5 else 0)
        if kind == "c": bx = 118 + bw / 2
        else: bx = 962 - bw / 2
        blit(cv, bubble(bw, kind), bx, y, a)
        if k == 5: blit(cv, shape("pin", 34, RED), bx + bw / 2 - 40, y - 2, a); text(cv, s, 32, "Medium", INK, bx - bw / 2 + 28, y, "lm", a)
        else: text(cv, s, 32, "Medium", INK, bx - bw / 2 + 28, y, "lm", a)
    for k, (t0, t1) in enumerate(tl["typ"]):
        if t0 <= t < t1:
            y = top + 150 + (2 * k + 1) * 82; blit(cv, bubble(110, "a"), 962 - 55, y)
            for j in range(3): blit(cv, shape("circle", 14, (90, 130, 120)), 962 - 80 + j * 25, y - 5 * math.sin((t - t0) * 9 - j * 0.9) ** 2 * 2)
    if t > tl["note"]:
        pp = ease(t, tl["note"], 0.45, e_back); blit(cv, card(760, 84, 42, (255, 255, 255, 245)), 540, 1108 + (1 - pp) * 80, clamp((t - tl["note"]) / 0.2))
        blit(cv, shape("bell", 40, GREEN), 214, 1108 + (1 - pp) * 80); text(cv, "Solo te avisa si hace falta", 32, "SemiBold", INK, 262, 1108 + (1 - pp) * 80, "lm")

# ---------- ESCENA C: citas ----------
def tl_C(d): return {"banner": 0.5, "cards": [0.9, 1.15, 1.4], "conf": [2.5, 3.3, 4.1]}
def scene_C(cv, t, d):
    tl = tl_C(d); chip(cv, t, LABELS[2])
    pb = ease(t, tl["banner"], 0.5, e_back)
    if t >= tl["banner"]:
        y = 425 - (1 - pb) * 150; blit(cv, card(900, 110, 34, (255, 255, 255, 245)), 540, y, clamp((t - tl["banner"]) / 0.2))
        blit(cv, shape("bell", 52, GREEN), 140, y); text(cv, "Recordatorio enviado", 34, "Bold", INK, 190, y - 18, "lm"); text(cv, "Mañana a las 10:30 · ¿Confirmas?", 28, "Medium", MUTED, 190, y + 22, "lm")
    for k, (ti, who, hour) in enumerate(zip(tl["cards"], ["Ana · Revisión", "Luis · Primera visita", "Marta · Consulta"], ["10:30", "12:00", "17:15"])):
        if t < ti: continue
        p = ease(t, ti, 0.45, e_back); a = clamp((t - ti) / 0.2); cy = 630 + k * 170 + (1 - p) * 60
        blit(cv, card(900, 150, 36, (255, 255, 255, 245)), 540, cy, a)
        text(cv, hour, 52, "Bold", INK, 150, cy - 14, "lm", a); text(cv, who, 32, "Medium", MUTED, 150, cy + 34, "lm", a)
        c = tl["conf"][k]; done = t >= c; pp = ease(t, c, 0.35, e_back)
        if done: blit(cv, pill("Confirmada", 30, GREEN, WHITE, 26, 10), 800, cy, a, 0.85 + 0.15 * min(pp, 1.2)); blit(cv, shape("check", 40, WHITE), 800 - twidth("Confirmada", 30, "Bold") / 2 - 8, cy, a)
        else: blit(cv, pill("Pendiente", 30, (226, 232, 242), MUTED, 26, 10, "SemiBold"), 800, cy, a)

# ---------- ESCENA D: presupuestos ----------
EMAIL = ["Hola, ¿me podéis pasar un presupuesto", "para reformar la cocina? Gracias."]
def tl_D(d): return {"email": 0.3, "type": (0.7, 2.3), "spark": 2.5, "rows": [2.9, 3.3, 3.7, 4.1], "tot": 4.6, "stamp": 5.3}
def scene_D(cv, t, d):
    tl = tl_D(d); chip(cv, t, LABELS[3]); p = ease(t, tl["email"], 0.5, e_back)
    if t >= tl["email"]:
        cy = 505 + (1 - p) * -80; blit(cv, card(900, 280, 34, (255, 255, 255, 245)), 540, cy, clamp((t - tl["email"]) / 0.2))
        blit(cv, shape("circle", 52, (210, 220, 240)), 130, cy - 90); text(cv, "Cliente", 32, "Bold", INK, 175, cy - 100, "lm"); text(cv, "Solicitud de presupuesto", 26, "Medium", MUTED, 175, cy - 66, "lm")
        n = int(len("".join(EMAIL)) * clamp((t - tl["type"][0]) / (tl["type"][1] - tl["type"][0]))); s1 = EMAIL[0][:n]; s2 = EMAIL[1][:max(0, n - len(EMAIL[0]))]
        if s1: text(cv, s1, 32, "Medium", INK, 110, cy - 5, "lm")
        if s2: text(cv, s2, 32, "Medium", INK, 110, cy + 45, "lm")
        if (t * 2) % 1 < .5 and t < tl["type"][1] + .5: ImageDraw.Draw(cv).rectangle([110 + twidth(s2 or s1, 32), cy + (35 if s2 else -17), 114 + twidth(s2 or s1, 32), cy + (65 if s2 else 13)], fill=INK)
    if t >= tl["spark"]:
        ps = ease(t, tl["spark"], 0.5, e_back); blit(cv, shape("star", 90, YEL), 540, 695, clamp((t - tl["spark"]) / .2), 0.9 * ps + 0.08 * math.sin(t * 8))
        for j in range(3): text(cv, "v", 38, "Bold", WHITE, 540 + (j - 1) * 0, 715 + j * 0, a=0) if False else None
    if t >= tl["spark"] + 0.2:
        pd = ease(t, tl["spark"] + 0.2, 0.5, e_back); cy = 925; blit(cv, card(900, 340, 34, (255, 255, 255, 248)), 540, cy + (1 - pd) * 90, clamp((t - tl["spark"] - .2) / .2))
        oy = (1 - pd) * 90; text(cv, "PRESUPUESTO", 36, "Bold", INK, 110, cy - 125 + oy, "lm"); text(cv, "Nº 0042", 26, "Medium", MUTED, 970, cy - 125 + oy, "rm")
        widths = [420, 520, 360, 470]
        for k, tr in enumerate(tl["rows"]):
            g = ease(t, tr, 0.4); y = cy - 70 + k * 44 + oy
            if g > 0:
                ImageDraw.Draw(cv).rounded_rectangle([110, y - 8, 110 + widths[k] * g, y + 8], 8, fill=GRAY)
                ImageDraw.Draw(cv).rounded_rectangle([970 - 120 * g, y - 8, 970, y + 8], 8, fill=(170, 182, 205))
        if t >= tl["tot"]:
            g = ease(t, tl["tot"], 0.4); y = cy + 120 + oy; ImageDraw.Draw(cv).line([110, y - 28, 970, y - 28], fill=GRAY, width=3)
            text(cv, "TOTAL", 28, "Bold", INK, 110, y, "lm", g); ImageDraw.Draw(cv).rounded_rectangle([970 - 170 * g, y - 12, 970, y + 12], 10, fill=YEL)
    if t >= tl["stamp"]:
        ps = ease(t, tl["stamp"], 0.45, e_back); blit(cv, pill("Borrador listo · revisar", 30, GREEN, WHITE, 30, 12), 540, 1112, clamp((t - tl["stamp"]) / .2), 0.7 + 0.3 * ps)

# ---------- ESCENA E: cierre ----------
def tl_E(d): return {"logo": 0.15, "name": 0.5, "sub": 0.8, "handle": 1.1, "btn": 1.8}
def scene_E(cv, t, d):
    tl = tl_E(d); p = ease(t, tl["logo"], 0.6, e_back)
    blit(cv, shape("circle", 250, YEL), 540, 520, clamp((t - tl["logo"]) / .2), p); blit(cv, shape("bolt", 140, INK), 540, 520, clamp((t - tl["logo"]) / .2), p)
    for key, s, size, wt, col, y in [("name", "Atajo", 112, "Bold", WHITE, 730), ("sub", "IA para negocios", 50, "SemiBold", YEL, 825), ("handle", "@atajos.ia.es", 40, "Medium", WHITE, 895)]:
        if t >= tl[key]: q = ease(t, tl[key], 0.4); text(cv, s, size, wt, col, 540, y + (1 - q) * 30, a=q)
    if t >= tl["btn"]:
        q = ease(t, tl["btn"], 0.5, e_back); a = clamp((t - tl["btn"]) / .2); pulse = 1 + 0.04 * math.sin(t * 6)
        pg = pill("Guardar", 36, (255, 255, 255), INK, 36, 16); gx = 390; pw = pg.width + 54
        blit(cv, card(pw, pg.height - 56, (pg.height - 56) // 2, (255, 255, 255, 255)), gx, 1040, a, q); text(cv, "Guardar", 36, "Bold", INK, gx + 27, 1040, a=a)
        blit(cv, shape("bookmark", 40, INK), gx - pw / 2 + 40, 1040, a, q)
        blit(cv, pill("Seguir", 36, YEL, INK, 52, 16), 710, 1040, a, q * pulse)

SCENES = [scene_A, scene_B, scene_C, scene_D, scene_E]

def frame(T):
    k = max(i for i in range(5) if T >= BOUNDS[i] - 1e-9) if T < DUR else 4
    s0 = BOUNDS[k]; d = BOUNDS[k + 1] - s0; t = T - s0; base = bg_at(T); cv = base.copy()
    SCENES[k](cv, t, d)
    if k < 4 and t > d - 0.18: cv = Image.blend(cv, base, clamp((t - (d - 0.18)) / 0.18))
    subs(cv, T); overlay_ui(cv, T); return cv

# ---------- sonido ----------
def sfx_events():
    ev = []; off = BOUNDS
    ev += [(off[0] + x, "pop") for x in tl_A(0)["cards"]]
    b = tl_B(0); ev += [(off[1] + x, "pop") for x in b["msg"]] + [(off[1] + b["note"], "ding")]
    c = tl_C(0); ev += [(off[2] + c["banner"], "ding")] + [(off[2] + x, "pop") for x in c["cards"]] + [(off[2] + x, "tick") for x in c["conf"]]
    dd = tl_D(0); ev += [(off[3] + dd["spark"], "ding"), (off[3] + dd["stamp"], "tick")] + [(off[3] + x, "pop") for x in dd["rows"][:1]]
    e = tl_E(0); ev += [(off[4] + e["logo"], "pop"), (off[4] + e["btn"], "ding")]
    return ev
def make_sfx(sr=44100):
    out = np.zeros(int((DUR + 1) * sr), dtype="float32")
    def tone(f, dur, decay, amp, f2=None):
        x = np.arange(int(dur * sr)) / sr; y = np.sin(2 * np.pi * f * x) + (0.5 * np.sin(2 * np.pi * f2 * x) if f2 else 0); return (y * np.exp(-decay * x) * amp).astype("float32")
    snd = {"pop": tone(560, .14, 38, .30), "tick": tone(1000, .09, 50, .25), "ding": tone(1318, .6, 6, .22, 1760)}
    for tm, k in sfx_events():
        i = int(tm * sr); s = snd[k]; out[i:i + len(s)] += s[:len(out) - i]
    return out

def render_scene(k):
    final = f"{OUTD}/scenes/s{k}.mp4"; tmp = f"{OUTD}/scenes/s{k}.tmp.mp4"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-g", "30", "-an", tmp]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(FR[k], FR[k + 1]): pr.stdin.write(frame(n / FPS).tobytes())
    pr.stdin.close(); rc = pr.wait()
    if rc == 0: os.replace(tmp, final)
    return k, rc

if __name__ == "__main__":
    if sys.argv[1] == "preview":
        os.makedirs(OUTD + "/prev", exist_ok=True)
        for x in sys.argv[2:]: frame(float(x)).save(f"{OUTD}/prev/f_{float(x):05.2f}.png")
        print("preview ok")
    else:
        with Pool(4) as p: res = p.map(render_scene, range(5), chunksize=1)
        print(res); assert all(r == 0 for _, r in res)
        open(f"{OUTD}/list.txt", "w").write("".join(f"file 'scenes/s{k}.mp4'\n" for k in range(5)))
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{OUTD}/list.txt", "-c", "copy", f"{OUTD}/video.tmp.mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"{SRC}/voice.wav", "-ar", "44100", f"{OUTD}/voice44.wav"], check=True)
        v, sr = sf.read(f"{OUTD}/voice44.wav"); v = v.astype("float32"); s = make_sfx(sr)
        n = max(len(v), len(s)); mix = np.zeros(n, dtype="float32"); mix[:len(v)] += v; mix[:len(s)] += s; sf.write(f"{OUTD}/mix.wav", mix, sr)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"{OUTD}/video.tmp.mp4", "-i", f"{OUTD}/mix.wav", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-af", "loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100", "-c:a", "aac", "-b:a", "160k", "-ac", "1", "-shortest", "-movflags", "+faststart",
                        f"{OUTD}/reel_v2_M.tmp.mp4"], check=True)
        os.replace(f"{OUTD}/reel_v2_M.tmp.mp4", f"{OUTD}/reel_v2_M.mp4"); print("done")
