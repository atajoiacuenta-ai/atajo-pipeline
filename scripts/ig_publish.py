#!/usr/bin/env python3
"""Publica un Reel con la API oficial de Instagram (Graph API) usando subida reanudable (sin hosting público).
Variables de entorno: IG_ACCESS_TOKEN (secreto), IG_USER_ID, GRAPH_VERSION (opc.), GRAPH_BASE (opc., para pruebas).
Uso: python scripts/ig_publish.py --video v.mp4 --caption-file c.txt [--cover c.jpg] [--dry-run] [--no-ai-label]
--dry-run: crea el contenedor y sube el vídeo, espera a que esté procesado, pero NO publica."""
import argparse, json, os, sys, time, urllib.request, urllib.parse, urllib.error

TOKEN = os.environ.get("IG_ACCESS_TOKEN", ""); UID = os.environ.get("IG_USER_ID", "")
VER = os.environ.get("GRAPH_VERSION", "v23.0"); BASE = os.environ.get("GRAPH_BASE", "https://graph.facebook.com").rstrip("/")

def fail(msg, code=1): print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)
def redact(s): return s.replace(TOKEN, "***") if TOKEN else s

def call(method, url, data=None, headers=None, body=None, retries=3):
    """data: dict -> form-urlencoded (el token va en el cuerpo, nunca en la URL). body: bytes crudos."""
    h = dict(headers or {})
    if data is not None: body = urllib.parse.urlencode(data).encode(); h.setdefault("Content-Type", "application/x-www-form-urlencoded")
    for n in range(retries):
        req = urllib.request.Request(url, data=body, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as r: return json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")
            if e.code >= 500 and n < retries - 1: time.sleep(5 * (n + 1)); continue
            try: err = json.loads(txt).get("error", {})
            except Exception: err = {"message": txt[:300]}
            fail(redact(f"HTTP {e.code} en {method} {url.split('?')[0]} -> tipo={err.get('type')} código={err.get('code')} sub={err.get('error_subcode')} mensaje={err.get('message')}"))
        except urllib.error.URLError as e:
            if n < retries - 1: time.sleep(5 * (n + 1)); continue
            fail(redact(f"red: {e.reason}"))

def get(path, **params):
    q = urllib.parse.urlencode({**params, "access_token": TOKEN}); return call("GET", f"{BASE}/{VER}/{path}?{q}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--caption-file", required=True)
    ap.add_argument("--cover"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--no-ai-label", action="store_true")
    ap.add_argument("--max-wait", type=int, default=300); a = ap.parse_args()
    if not TOKEN or not UID: fail("faltan IG_ACCESS_TOKEN o IG_USER_ID")
    size = os.path.getsize(a.video)
    if size > 300 * 1024 * 1024: fail("el vídeo supera 300 MB")
    caption = open(a.caption_file, encoding="utf-8").read().strip()
    if len(caption) > 2200: fail("el pie supera 2200 caracteres")

    lim = get(f"{UID}/content_publishing_limit", fields="quota_usage,config")["data"][0]
    print(f"Cupo 24 h: {lim['quota_usage']}/{lim['config']['quota_total']}")
    if lim["quota_usage"] >= lim["config"]["quota_total"]: fail("cupo de publicación agotado")

    form = {"media_type": "REELS", "upload_type": "resumable", "caption": caption, "share_to_feed": "true", "access_token": TOKEN}
    if not a.no_ai_label: form["is_ai_generated"] = "true"
    cont = call("POST", f"{BASE}/{VER}/{UID}/media", data=form)
    cid, uri = cont.get("id"), cont.get("uri")
    if not cid or not uri: fail(f"respuesta inesperada al crear el contenedor: {redact(json.dumps(cont))}")
    print(f"Contenedor creado: {cid}")

    with open(a.video, "rb") as f: payload = f.read()
    up = call("POST", uri, headers={"Authorization": f"OAuth {TOKEN}", "offset": "0", "file_size": str(size), "Content-Type": "application/octet-stream"}, body=payload)
    if not up.get("success", True): fail(f"la subida no confirmó éxito: {redact(json.dumps(up))}")
    print(f"Vídeo subido ({size / 1e6:.1f} MB)")

    t0 = time.time(); status = ""
    while time.time() - t0 < a.max_wait:
        st = get(cid, fields="status_code,status"); status = st.get("status_code", "")
        print(f"Estado: {status}")
        if status == "FINISHED": break
        if status in ("ERROR", "EXPIRED"): fail(f"el procesamiento terminó en {status}: {st.get('status')}")
        time.sleep(float(os.environ.get("POLL_SECS", "15")))
    if status != "FINISHED": fail("tiempo de espera agotado esperando al procesamiento")

    if a.dry_run: print("DRY-RUN: contenedor listo, NO se ha publicado."); return
    pub = call("POST", f"{BASE}/{VER}/{UID}/media_publish", data={"creation_id": cid, "access_token": TOKEN})
    mid = pub.get("id"); print(f"Publicado: id={mid}")
    try: print("Enlace:", get(mid, fields="permalink").get("permalink"))
    except SystemExit: pass

if __name__ == "__main__": main()
