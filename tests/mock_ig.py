"""Servidor simulado de la Graph API para probar scripts/ig_publish.py sin tocar Instagram."""
import json, threading, http.server, urllib.parse, sys
LOG = []; STATE = {"polls": 0}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj, code=200): b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); LOG.append(("GET", u.path, dict(q)))
        if "/9999/" in u.path: return self._send({"error": {"type": "OAuthException", "code": 190, "message": "token inválido: " + q.get("access_token", [""])[0]}}, 400)
        if u.path.endswith("/content_publishing_limit"): return self._send({"data": [{"quota_usage": 3, "config": {"quota_total": 100, "quota_duration": 86400}}]})
        if u.path.endswith("/C1"):
            STATE["polls"] += 1; return self._send({"status_code": "IN_PROGRESS" if STATE["polls"] < 3 else "FINISHED", "id": "C1"})
        if u.path.endswith("/M1"): return self._send({"permalink": "https://www.instagram.com/reel/TEST/"})
        self._send({"error": {"type": "OAuthException", "code": 100, "message": "campo inexistente"}}, 400)
    def do_POST(self):
        u = urllib.parse.urlparse(self.path); n = int(self.headers.get("Content-Length", 0)); raw = self.rfile.read(n)
        if u.path.startswith("/upload/"):
            LOG.append(("UPLOAD", u.path, {"auth": self.headers.get("Authorization"), "offset": self.headers.get("offset"), "file_size": self.headers.get("file_size"), "len": len(raw), "query": u.query})); return self._send({"success": True})
        form = urllib.parse.parse_qs(raw.decode()); LOG.append(("POST", u.path, {k: v[0] for k, v in form.items()}, u.query))
        if u.path.endswith("/media"): return self._send({"id": "C1", "uri": f"http://127.0.0.1:{PORT}/upload/C1"})
        if u.path.endswith("/media_publish"): return self._send({"id": "M1"})
        self._send({"error": {"message": "no"}}, 404)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H); PORT = srv.server_address[1]
def start(): threading.Thread(target=srv.serve_forever, daemon=True).start(); return PORT
