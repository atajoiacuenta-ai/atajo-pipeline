"""Reel dirigido por datos: plantillas de escena animadas (PIL) configuradas desde un guion JSON.
Uso: python scripts/reel.py render content/002.json | preview content/002.json 1.0 5.5 ...
Requiere out/voice-v3/<id>/{voice.wav,meta.json} (los crea scripts/voice_cb.py)."""
import json, os, sys, math, random, subprocess
from functools import lru_cache
import numpy as np, soundfile as sf
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from multiprocessing import Pool

W, H, FPS = 1080, 1920, 30
SPEC = json.load(open(sys.argv[2])); NAME = SPEC["id"]
SRC = f"out/voice-v3/{NAME}"; OUTD = f"out/reel/{NAME}"; os.makedirs(OUTD + "/scenes", exist_ok=True)
meta = json.load(open(f"{SRC}/meta.json")); TIMES = meta["times"]; DUR = meta["duration"]; SEGS = meta["texts"]
SC = SPEC["scenes"]; NS = len(SC)
mid = lambda i: (TIMES[i][1] + TIMES[i + 1][0]) / 2
BOUNDS = [0] + [mid(s["first"] - 1) for s in SC[1:]] + [DUR]
FR = [round(b * FPS) for b in BOUNDS]

IVORY = (250, 247, 240); INK = (28, 31, 40); ACC = (190, 84, 52); WHITE = (255, 255, 255); RED = ACC
GREEN = (31, 122, 94); GRAY = (224, 217, 202); MUTED = (112, 108, 100); SAND = (236, 229, 214); YEL = ACC

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
        ImageDraw.Draw(sh).rounded_rectangle([pad * S, (pad + 8) * S, (pad + w) * S, (pad + h + 8) * S], r * S, fill=(70, 55, 30, 40))
        im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(12 * S)))
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
    p = ease(t, 0.05, 0.4); blit(cv, pill(label, 36, INK, IVORY), 540, 300 - (1 - p) * 34, p)

@lru_cache(None)
def clock_face(R):
    S = 2; n = (2 * R + 40) * S; im = Image.new("RGBA", (n, n), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = n // 2
    d.ellipse([c - R * S - 14, c - R * S - 14, c + R * S + 14, c + R * S + 14], fill=INK + (255,))
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
    c0 = np.array([252, 250, 244]); c1 = np.array([241, 235, 221])
    arr = (c0 + (c1 - c0) * yy).repeat(W, 1).astype("uint8"); im = Image.fromarray(arr).convert("RGBA")
    ov = Image.new("RGBA", (W, BH), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); random.seed(11)
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
                cur = max([wst for ln in lines for _, wst in ln if wst <= T] or [-1])
                for li, line in enumerate(lines):
                    widths = [twidth(w, SUBSZ, "Bold") for w, _ in line]; sp = 20; total = sum(widths) + sp * (len(line) - 1); x = 540 - total / 2
                    for (w, wst), wd in zip(line, widths):
                        im, ox, oy = tspr(w, SUBSZ, "Bold", ACC if wst == cur else INK, "lm", 0)
                        paste(cv, im, x - ox, y0 + li * 92 + (1 - p) * 22 - oy, p); x += wd + sp
                return

def overlay_ui(cv, T):
    text(cv, "Atajo | IA para negocios", 32, "SemiBold", MUTED, 540, 1490)
    d = ImageDraw.Draw(cv); d.rectangle([0, 1884, W, 1892], fill=(255, 255, 255)) if False else None
    d.rounded_rectangle([60, 1884, W - 60, 1892], 4, fill=(226, 219, 203)); d.rounded_rectangle([60, 1884, 60 + (W - 120) * T / DUR, 1892], 4, fill=INK)


# =================== PLANTILLAS ===================
def lab(s): return s.get("label", "")

# ---- hook: reloj + tarjetas que se repiten
def tl_hook(s, d):
    n = len(s["items"]); st = max(0.5, min(1.05, (d - 1.4) / n)); return {"cards": [0.9 + st * k for k in range(n)]}
def ev_hook(s, tl): return [(x, "pop") for x in tl["cards"]]
def draw_hook(cv, t, d, s, tl):
    chip(cv, t, lab(s)); blit(cv, clock_face(130), 540, 530); ang = t * 540
    hand(cv, 540, 530, ang, 105, 8, INK); hand(cv, 540, 530, ang / 12, 70, 11, INK); blit(cv, shape("circle", 30, ACC), 540, 530)
    for k, (ti, title) in enumerate(zip(tl["cards"], s["items"])):
        p = ease(t, ti, 0.5, e_back); a = clamp((t - ti) / 0.2); cx = 540 + (1 - p) * 700; cy = 770 + k * 112
        blit(cv, card(880, 100, 30, (255, 255, 255, 238)), cx, cy, a)
        blit(cv, card(60, 60, 18, ACC + (255,), shadow=False), cx - 440 + 64, cy, a); text(cv, "!", 40, "Bold", WHITE, cx - 440 + 64, cy - 1, a=a)
        text(cv, title, 38, "SemiBold", INK, cx - 440 + 120, cy, "lm", a)
        blit(cv, pill(s.get("tag", "se repite"), 26, (247, 228, 220), ACC, 22, 8), cx + 440 - 110, cy, a)

# ---- chat: conversación tipo mensajería con indicador de escritura
@lru_cache(None)
def chat_window():
    S, w, h, r, pad = 2, 920, 660, 36, 28
    im = Image.new("RGBA", ((w + 2 * pad) * S, (h + 2 * pad) * S), (0, 0, 0, 0))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([pad * S, (pad + 10) * S, (pad + w) * S, (pad + h + 10) * S], r * S, fill=(0, 0, 0, 100))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(12 * S)))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0)); dd = ImageDraw.Draw(layer)
    dd.rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], fill=(244, 240, 230, 255)); dd.rectangle([pad * S, pad * S, (pad + w) * S, (pad + 100) * S], fill=INK + (255,))
    m = Image.new("L", im.size, 0); ImageDraw.Draw(m).rounded_rectangle([pad * S, pad * S, (pad + w) * S, (pad + h) * S], r * S, fill=255)
    im.paste(layer, (0, 0), m); return im.resize((w + 2 * pad, h + 2 * pad), Image.LANCZOS)
@lru_cache(None)
def bubble(w, kind): return card(w, 74, 26, (255, 255, 255, 255) if kind == "c" else (224, 235, 226, 255), shadow=False)
def tl_chat(s, d):
    n = len(s["msgs"]); st = max(0.6, min(0.9, (d - 2.4) / n)); ms = [0.7 + st * k for k in range(n)]
    typ = [(m - 0.45, m) for m, (kind, _) in zip(ms, s["msgs"]) if kind == "a"]
    return {"msg": ms, "typ": typ, "note": ms[-1] + 0.9}
def ev_chat(s, tl): return [(x, "pop") for x in tl["msg"]] + ([(tl["note"], "ding")] if s.get("note") else [])
def draw_chat(cv, t, d, s, tl):
    chip(cv, t, lab(s)); p = ease(t, 0.0, 0.5, e_back); cx, cy = 540, 730 + (1 - p) * 120
    blit(cv, chat_window(), cx, cy, clamp(t / 0.25), 0.94 + 0.06 * p)
    if t < 0.3: return
    top = 730 - 330; blit(cv, shape("circle", 58, (255, 255, 255)), 130, top + 52)
    text(cv, s.get("name", "Tu negocio")[:1].upper(), 32, "Bold", INK, 130, top + 52); text(cv, s.get("name", "Tu negocio"), 34, "Bold", IVORY, 180, top + 38, "lm")
    text(cv, s.get("status", "asistente con IA · en línea"), 24, "Medium", (190, 186, 176), 180, top + 74, "lm"); bd = s.get("badge", "IA")
    if bd: blit(cv, pill(bd, 22, ACC, WHITE, 18, 6), 940, top + 52)
    n = len(s["msgs"])
    for k, (kind, msg) in enumerate(s["msgs"]):
        ts = tl["msg"][k]
        if t < ts: continue
        pp = ease(t, ts, 0.3, e_back); a = clamp((t - ts) / 0.15); y = top + 150 + k * 82 + (1 - pp) * 18
        pin = s.get("pin_last") and k == n - 1; bw = twidth(msg, 32, "Medium") + 56 + (44 if pin else 0)
        bx = 118 + bw / 2 if kind == "c" else 962 - bw / 2
        blit(cv, bubble(bw, kind), bx, y, a)
        if pin: blit(cv, shape("pin", 34, ACC), bx + bw / 2 - 40, y - 2, a)
        text(cv, msg, 32, "Medium", INK, bx - bw / 2 + 28, y, "lm", a)
    for (t0, t1), k in zip(tl["typ"], [i for i, (kd, _) in enumerate(s["msgs"]) if kd == "a"]):
        if t0 <= t < t1:
            y = top + 150 + k * 82; blit(cv, bubble(110, "a"), 962 - 55, y)
            for j in range(3): blit(cv, shape("circle", 14, (140, 136, 126)), 962 - 80 + j * 25, y - 5 * math.sin((t - t0) * 9 - j * 0.9) ** 2 * 2)
    if s.get("note") and t > tl["note"]:
        pp = ease(t, tl["note"], 0.45, e_back); y = 1108 + (1 - pp) * 80; blit(cv, card(760, 84, 42, (255, 255, 255, 245)), 540, y, clamp((t - tl["note"]) / 0.2))
        blit(cv, shape("bell", 40, ACC), 214, y); text(cv, s["note"], 32, "SemiBold", INK, 262, y, "lm")

# ---- status: tarjetas que pasan de un estado a otro (citas, pedidos, facturas...)
def tl_status(s, d):
    n = len(s["items"]); st = max(0.5, min(0.8, (d - 2.8) / n))
    return {"banner": 0.5, "cards": [0.9 + 0.25 * k for k in range(n)], "conf": [2.3 + st * k for k in range(n)]}
def ev_status(s, tl): return [(tl["banner"], "ding")] + [(x, "pop") for x in tl["cards"]] + [(x, "tick") for x in tl["conf"]]
def draw_status(cv, t, d, s, tl):
    chip(cv, t, lab(s)); pb = ease(t, tl["banner"], 0.5, e_back); b = s["banner"]
    if t >= tl["banner"]:
        y = 425 - (1 - pb) * 150; blit(cv, card(900, 110, 34, (255, 255, 255, 245)), 540, y, clamp((t - tl["banner"]) / 0.2))
        blit(cv, shape("bell", 52, ACC), 140, y); text(cv, b[0], 34, "Bold", INK, 190, y - 18, "lm"); text(cv, b[1], 28, "Medium", MUTED, 190, y + 22, "lm")
    f0, f1 = s.get("from", "Pendiente"), s.get("to", "Confirmada")
    for k, (ti, it) in enumerate(zip(tl["cards"], s["items"])):
        if t < ti: continue
        p = ease(t, ti, 0.45, e_back); a = clamp((t - ti) / 0.2); cy = 630 + k * 170 + (1 - p) * 60
        blit(cv, card(900, 150, 36, (255, 255, 255, 245)), 540, cy, a)
        text(cv, it[0], 52, "Bold", INK, 150, cy - 14, "lm", a); text(cv, it[1], 32, "Medium", MUTED, 150, cy + 34, "lm", a)
        c = tl["conf"][k]; pp = ease(t, c, 0.35, e_back)
        if t >= c:
            blit(cv, pill(f1, 30, GREEN, WHITE, 26, 10), 800, cy, a, 0.85 + 0.15 * min(pp, 1.2)); blit(cv, shape("check", 40, WHITE), 800 - twidth(f1, 30, "Bold") / 2 - 8, cy, a)
        else: blit(cv, pill(f0, 30, SAND, MUTED, 26, 10, "SemiBold"), 800, cy, a)

# ---- doc: un mensaje se convierte en un documento listo
def _sc(s, d): return clamp(d / 6.4, 0.6, 1.5)
def tl_doc(s, d):
    k = _sc(s, d); return {"email": 0.3, "type": (0.7 * k, 2.3 * k), "spark": 2.5 * k, "rows": [2.9 * k, 3.3 * k, 3.7 * k, 4.1 * k], "tot": 4.6 * k, "stamp": 5.3 * k}
def ev_doc(s, tl): return [(tl["spark"], "ding"), (tl["stamp"], "tick"), (tl["rows"][0], "pop")]
def draw_doc(cv, t, d, s, tl):
    chip(cv, t, lab(s)); p = ease(t, tl["email"], 0.5, e_back); EM = s["email"]
    if t >= tl["email"]:
        cy = 505 + (1 - p) * -80; blit(cv, card(900, 280, 34, (255, 255, 255, 245)), 540, cy, clamp((t - tl["email"]) / 0.2))
        blit(cv, shape("circle", 52, SAND), 130, cy - 90); text(cv, s.get("from", "Cliente"), 32, "Bold", INK, 175, cy - 100, "lm"); text(cv, s.get("subject", "Solicitud"), 26, "Medium", MUTED, 175, cy - 66, "lm")
        n = int(len("".join(EM)) * clamp((t - tl["type"][0]) / (tl["type"][1] - tl["type"][0]))); s1 = EM[0][:n]; s2 = EM[1][:max(0, n - len(EM[0]))]
        if s1: text(cv, s1, 32, "Medium", INK, 110, cy - 5, "lm")
        if s2: text(cv, s2, 32, "Medium", INK, 110, cy + 45, "lm")
        if (t * 2) % 1 < .5 and t < tl["type"][1] + .5: ImageDraw.Draw(cv).rectangle([110 + twidth(s2 or s1, 32), cy + (35 if s2 else -17), 114 + twidth(s2 or s1, 32), cy + (65 if s2 else 13)], fill=INK)
    if t >= tl["spark"]:
        ps = ease(t, tl["spark"], 0.5, e_back); blit(cv, shape("star", 90, ACC), 540, 695, clamp((t - tl["spark"]) / .2), 0.9 * ps + 0.08 * math.sin(t * 8))
    if t >= tl["spark"] + 0.2:
        pd = ease(t, tl["spark"] + 0.2, 0.5, e_back); cy = 925; blit(cv, card(900, 340, 34, (255, 255, 255, 248)), 540, cy + (1 - pd) * 90, clamp((t - tl["spark"] - .2) / .2))
        oy = (1 - pd) * 90; text(cv, s.get("doc", "PRESUPUESTO"), 36, "Bold", INK, 110, cy - 125 + oy, "lm"); text(cv, s.get("num", "Nº 0042"), 26, "Medium", MUTED, 970, cy - 125 + oy, "rm")
        widths = [420, 520, 360, 470]
        for k, tr in enumerate(tl["rows"]):
            g = ease(t, tr, 0.4); y = cy - 70 + k * 44 + oy
            if g > 0:
                ImageDraw.Draw(cv).rounded_rectangle([110, y - 8, 110 + widths[k] * g, y + 8], 8, fill=GRAY)
                ImageDraw.Draw(cv).rounded_rectangle([970 - 120 * g, y - 8, 970, y + 8], 8, fill=(204, 195, 178))
        if t >= tl["tot"]:
            g = ease(t, tl["tot"], 0.4); y = cy + 120 + oy; ImageDraw.Draw(cv).line([110, y - 28, 970, y - 28], fill=GRAY, width=3)
            text(cv, s.get("total", "TOTAL"), 28, "Bold", INK, 110, y, "lm", g); ImageDraw.Draw(cv).rounded_rectangle([970 - 170 * g, y - 12, 970, y + 12], 10, fill=ACC)
    if t >= tl["stamp"]:
        ps = ease(t, tl["stamp"], 0.45, e_back); blit(cv, pill(s.get("stamp", "Borrador listo · revisar"), 30, GREEN, WHITE, 30, 12), 540, 1112, clamp((t - tl["stamp"]) / .2), 0.7 + 0.3 * ps)

# ---- checklist: elementos que se van marcando
def tl_checklist(s, d):
    n = len(s["items"]); st = max(0.5, min(1.0, (d - 1.6) / n)); return {"title": 0.3, "rows": [0.9 + st * k for k in range(n)]}
def ev_checklist(s, tl): return [(x, "tick") for x in tl["rows"]]
def draw_checklist(cv, t, d, s, tl):
    chip(cv, t, lab(s))
    if s.get("title"): text(cv, s["title"], 50, "Bold", INK, 540, 440, a=ease(t, tl["title"], 0.4))
    for k, ti in enumerate(tl["rows"]):
        if t < ti: continue
        p = ease(t, ti, 0.45, e_back); a = clamp((t - ti) / 0.2); cy = 580 + k * 125 + (1 - p) * 50
        blit(cv, card(880, 104, 30, (255, 255, 255, 245)), 540, cy, a)
        blit(cv, card(64, 64, 32, GREEN + (255,), shadow=False), 140, cy, a); blit(cv, shape("check", 40, WHITE), 140, cy, a)
        text(cv, s["items"][k], 36, "SemiBold", INK, 200, cy, "lm", a)

# ---- compare: antes / con IA
def tl_compare(s, d): return {"a": 0.6, "b": max(2.4, d * 0.5)}
def ev_compare(s, tl): return [(tl["a"], "pop"), (tl["b"], "ding")]
def draw_compare(cv, t, d, s, tl):
    chip(cv, t, lab(s))
    for key, (y0, col, fg) in zip(("a", "b"), ((620, MUTED, WHITE), (990, GREEN, WHITE))):
        ti = tl[key]
        if t < ti: continue
        side = s["before"] if key == "a" else s["after"]; p = ease(t, ti, 0.5, e_back); a = clamp((t - ti) / 0.2); cy = y0 + (1 - p) * 70
        blit(cv, card(900, 330, 36, (255, 255, 255, 245)), 540, cy, a)
        blit(cv, pill(side["title"], 30, col, fg, 28, 10), 150 + twidth(side["title"], 30, "Bold") / 2, cy - 120, a)
        for j, ln in enumerate(side["lines"][:3]):
            blit(cv, shape("circle", 22, col), 130, cy - 40 + j * 66, a); text(cv, ln, 34, "Medium", INK, 170, cy - 40 + j * 66, "lm", a)

# ---- stat: una cifra grande
def tl_stat(s, d): return {"v": 0.5, "c": 1.3, "src": 2.0}
def ev_stat(s, tl): return [(tl["v"], "pop"), (tl["c"], "tick")]
def draw_stat(cv, t, d, s, tl):
    chip(cv, t, lab(s)); p = ease(t, tl["v"], 0.55, e_back); a = clamp((t - tl["v"]) / 0.2)
    blit(cv, card(900, 360, 40, (255, 255, 255, 245)), 540, 760, a, 0.9 + 0.1 * p); text(cv, s["value"], s.get("size", 180), "Bold", ACC, 540, 760 + (1 - p) * 30, a=a)
    for j, ln in enumerate(s["caption"]):
        q = ease(t, tl["c"] + 0.25 * j, 0.4); text(cv, ln, 46, "SemiBold", INK, 540, 1010 + j * 62 + (1 - q) * 20, a=q)
    if s.get("source"): text(cv, s["source"], 26, "Medium", MUTED, 540, 1010 + len(s["caption"]) * 62 + 30, a=ease(t, tl["src"], 0.4))

# ---- outro: cierre de marca
def tl_outro(s, d): return {"logo": 0.15, "name": 0.5, "sub": 0.8, "handle": 1.1, "btn": 1.8}
def ev_outro(s, tl): return [(tl["logo"], "pop"), (tl["btn"], "ding")]
def draw_outro(cv, t, d, s, tl):
    p = ease(t, tl["logo"], 0.6, e_back)
    blit(cv, shape("circle", 250, ACC), 540, 520, clamp((t - tl["logo"]) / .2), p); blit(cv, shape("bolt", 140, IVORY), 540, 520, clamp((t - tl["logo"]) / .2), p)
    for key, st, size, wt, col, y in [("name", "Atajo", 112, "Bold", INK, 730), ("sub", "IA para negocios", 50, "SemiBold", ACC, 825), ("handle", "@atajos.ia.es", 40, "Medium", MUTED, 895)]:
        if t >= tl[key]: q = ease(t, tl[key], 0.4); text(cv, st, size, wt, col, 540, y + (1 - q) * 30, a=q)
    if t >= tl["btn"]:
        q = ease(t, tl["btn"], 0.5, e_back); a = clamp((t - tl["btn"]) / .2); pulse = 1 + 0.04 * math.sin(t * 6)
        pg = pill("Guardar", 36, (255, 255, 255), INK, 36, 16); gx = 390; pw = pg.width + 54
        blit(cv, card(pw, pg.height - 56, (pg.height - 56) // 2, (255, 255, 255, 255)), gx, 1040, a, q); text(cv, "Guardar", 36, "Bold", INK, gx + 27, 1040, a=a)
        blit(cv, shape("bookmark", 40, INK), gx - pw / 2 + 40, 1040, a, q); blit(cv, pill("Seguir", 36, ACC, WHITE, 52, 16), 710, 1040, a, q * pulse)

# ---- steps: pasos numerados con la ruta exacta (cada paso hecho pasa a verde)
def tl_steps(s, d):
    n = len(s["steps"]); st = max(0.6, min(1.3, (d - 1.2) / n)); return {"title": 0.3, "rows": [0.6 + st * k for k in range(n)]}
def ev_steps(s, tl): return [(x, "tick") for x in tl["rows"]]
def draw_steps(cv, t, d, s, tl):
    chip(cv, t, lab(s))
    if s.get("title"): text(cv, s["title"], 46, "Bold", INK, 540, 440, a=ease(t, tl["title"], 0.4))
    cur = sum(1 for x in tl["rows"] if t >= x)
    for k, ti in enumerate(tl["rows"]):
        if t < ti: continue
        p = ease(t, ti, 0.45, e_back); a = clamp((t - ti) / 0.2); cy = 560 + k * 150 + (1 - p) * 50
        blit(cv, card(900, 128, 32, (255, 255, 255, 245)), 540, cy, a)
        done = k < cur - 1
        blit(cv, card(70, 70, 35, (GREEN if done else ACC) + (255,), shadow=False), 135, cy, a)
        if done: blit(cv, shape("check", 44, WHITE), 135, cy, a)
        else: text(cv, str(k + 1), 38, "Bold", WHITE, 135, cy - 1, a=a)
        text(cv, s["steps"][k][0], 36, "SemiBold", INK, 200, cy - 18, "lm", a); text(cv, s["steps"][k][1], 27, "Medium", MUTED, 200, cy + 24, "lm", a)

# ---- prompt: el prompt se escribe y aparece el resultado
def tl_prompt(s, d):
    P = s["prompt"]; R = s["result"]; n = sum(len(x) for x in P); te = min(0.5 + n * 0.03, 0.5 + d * 0.35); rs = te + 0.8
    step = max(0.35, min(0.9, (d * 0.92 - rs) / len(R)))
    return {"p0": 0.5, "p1": te, "spark": te + 0.15, "r0": rs, "rows": [rs + step * k for k in range(len(R))]}
def ev_prompt(s, tl): return [(tl["p1"], "tick"), (tl["spark"], "ding")] + [(x, "pop") for x in tl["rows"]]
def draw_prompt(cv, t, d, s, tl):
    chip(cv, t, lab(s)); P = s["prompt"]; R = s["result"]
    h1 = 110 + 50 * len(P); cy1 = 420 + h1 / 2; a1 = clamp(t / 0.25); p1 = ease(t, 0.0, 0.5, e_back)
    blit(cv, card(900, h1, 34, (255, 255, 255, 245)), 540, cy1 + (1 - p1) * 60, a1)
    blit(cv, pill(s.get("prompt_tag", "Tu prompt"), 26, ACC, WHITE, 24, 8), 100 + twidth(s.get("prompt_tag", "Tu prompt"), 26, "Bold") / 2 + 12, cy1 - h1 / 2 + 36, a1)
    total = sum(len(x) for x in P); typed = int(total * clamp((t - tl["p0"]) / max(0.1, tl["p1"] - tl["p0"])))
    for j, line in enumerate(P):
        k = max(0, min(len(line), typed)); typed -= len(line)
        if k: text(cv, line[:k], 32, "Medium", INK, 100, cy1 - h1 / 2 + 92 + j * 50, "lm")
    if t >= tl["spark"]:
        ps = ease(t, tl["spark"], 0.5, e_back); blit(cv, shape("star", 64, ACC), 540, cy1 + h1 / 2 + 36, clamp((t - tl["spark"]) / .2), 0.9 * ps + 0.08 * math.sin(t * 8))
    if t >= tl["r0"] - 0.2:
        h2 = 110 + 50 * len(R); cy2 = 780 + h2 / 2 + 20; pr = ease(t, tl["r0"] - 0.2, 0.5, e_back)
        blit(cv, card(900, h2, 34, (255, 255, 255, 248)), 540, cy2 + (1 - pr) * 70, clamp((t - tl["r0"] + 0.2) / 0.2))
        tg = s.get("result_tag", "Resultado"); blit(cv, pill(tg, 26, GREEN, WHITE, 24, 8), 100 + twidth(tg, 26, "Bold") / 2 + 12, cy2 - h2 / 2 + 36 + (1 - pr) * 70)
        for j, (line, ti) in enumerate(zip(R, tl["rows"])):
            q = ease(t, ti, 0.35)
            if q > 0: text(cv, line, 32, "Medium", INK, 100, cy2 - h2 / 2 + 92 + j * 50 + (1 - q) * 14, "lm", q)

TPL = {n: (globals()["tl_" + n], globals()["draw_" + n], globals()["ev_" + n]) for n in ["hook", "chat", "status", "doc", "checklist", "compare", "stat", "steps", "prompt", "outro"]}
TL = []
for k, s in enumerate(SC): TL.append(TPL[s["type"]][0](s, BOUNDS[k + 1] - BOUNDS[k]))

def validate():
    err = []
    f = [s["first"] for s in SC]
    if f[0] != 0 or f != sorted(set(f)) or f[-1] >= len(SEGS): err.append("'first' de las escenas debe empezar en 0, ser creciente y < nº de cláusulas")
    if SC[-1]["type"] != "outro": err.append("la última escena debe ser 'outro'")
    for s in SC:
        if s["type"] == "chat":
            if len(s["msgs"]) > 6: err.append("chat: máx. 6 mensajes")
            for kd, m in s["msgs"]:
                if twidth(m, 32) + 100 > 880: err.append(f"chat: mensaje demasiado largo: {m}")
        if s["type"] == "hook" and len(s["items"]) > 4: err.append("hook: máx. 4 elementos")
        if s["type"] == "status" and len(s["items"]) > 3: err.append("status: máx. 3 elementos")
        if s["type"] == "checklist":
            if len(s["items"]) > 5: err.append("checklist: máx. 5 elementos")
            for m in s["items"]:
                if twidth(m, 36, "SemiBold") > 640: err.append(f"checklist: texto largo: {m}")
        if s["type"] == "hook":
            for m in s["items"]:
                if twidth(m, 38, "SemiBold") > 500: err.append(f"hook: texto largo: {m}")
        if s["type"] == "compare":
            for side in (s["before"], s["after"]):
                for ln in side["lines"]:
                    if twidth(ln, 34) > 680: err.append(f"compare: línea larga: {ln}")
        if s["type"] == "stat" and twidth(s["value"], s.get("size", 180), "Bold") > 820: err.append("stat: cifra demasiado ancha")
        if s["type"] == "steps":
            if len(s["steps"]) > 4: err.append("steps: máx. 4 pasos")
            for a_, b_ in s["steps"]:
                if twidth(a_, 36, "SemiBold") > 640: err.append(f"steps: título largo: {a_}")
                if twidth(b_, 27) > 640: err.append(f"steps: detalle largo: {b_}")
        if s["type"] == "prompt":
            if len(s["prompt"]) > 4 or len(s["result"]) > 5: err.append("prompt: máx. 4 líneas de prompt y 5 de resultado")
            for ln in s["prompt"] + s["result"]:
                if twidth(ln, 32) > 780: err.append(f"prompt: línea larga: {ln}")
        lbl = s.get("label", "")
        if lbl and twidth(lbl, 36, "Bold") > 900: err.append(f"label largo: {lbl}")
    if err: print("\n".join("ERROR: " + e for e in err)); sys.exit(2)

def frame(T):
    k = max(i for i in range(NS) if T >= BOUNDS[i] - 1e-9) if T < DUR else NS - 1
    s0 = BOUNDS[k]; d = BOUNDS[k + 1] - s0; t = T - s0; base = bg_at(T); cv = base.copy()
    TPL[SC[k]["type"]][1](cv, t, d, SC[k], TL[k])
    if k < NS - 1 and t > d - 0.18: cv = Image.blend(cv, base, clamp((t - (d - 0.18)) / 0.18))
    subs(cv, T); overlay_ui(cv, T); return cv

def make_sfx(sr=44100):
    out = np.zeros(int((DUR + 1) * sr), dtype="float32")
    def tone(f, dur, decay, amp, f2=None):
        x = np.arange(int(dur * sr)) / sr; y = np.sin(2 * np.pi * f * x) + (0.5 * np.sin(2 * np.pi * f2 * x) if f2 else 0); return (y * np.exp(-decay * x) * amp).astype("float32")
    snd = {"pop": tone(560, .14, 38, .30), "tick": tone(1000, .09, 50, .25), "ding": tone(1318, .6, 6, .22, 1760)}
    for k, s in enumerate(SC):
        for tm, kind in TPL[s["type"]][2](s, TL[k]):
            i = int((BOUNDS[k] + tm) * sr)
            if i < len(out): sn = snd[kind]; out[i:i + len(sn)] += sn[:len(out) - i]
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
    validate()
    if sys.argv[1] == "preview":
        os.makedirs(OUTD + "/prev", exist_ok=True)
        for x in sys.argv[3:]: frame(float(x)).save(f"{OUTD}/prev/f_{float(x):05.2f}.png")
        print("preview ok", [round(b, 2) for b in BOUNDS])
    else:
        with Pool(4) as p: res = p.map(render_scene, range(NS), chunksize=1)
        print(res); assert all(r == 0 for _, r in res)
        open(f"{OUTD}/list.txt", "w").write("".join(f"file 'scenes/s{k}.mp4'\n" for k in range(NS)))
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{OUTD}/list.txt", "-c", "copy", f"{OUTD}/video.tmp.mp4"], check=True)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"{SRC}/voice.wav", "-ar", "44100", f"{OUTD}/voice44.wav"], check=True)
        v, sr = sf.read(f"{OUTD}/voice44.wav"); v = v.astype("float32"); s = make_sfx(sr)
        n = max(len(v), len(s)); mix = np.zeros(n, dtype="float32"); mix[:len(v)] += v; mix[:len(s)] += s; sf.write(f"{OUTD}/mix.wav", mix, sr)
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", f"{OUTD}/video.tmp.mp4", "-i", f"{OUTD}/mix.wav", "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-t", f"{DUR:.3f}", "-af", f"atrim=0:{DUR:.3f},loudnorm=I=-16:TP=-1.5:LRA=9,aresample=44100", "-c:a", "aac", "-b:a", "160k", "-ac", "1", "-movflags", "+faststart",
                        f"{OUTD}/reel.tmp.mp4"], check=True)
        os.replace(f"{OUTD}/reel.tmp.mp4", f"{OUTD}/reel.mp4"); print("done")
