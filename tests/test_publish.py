import os, subprocess, sys, json
sys.path.insert(0, "tests"); import mock_ig
port = mock_ig.start(); SECRET = "EAAtokenSECRETOdePrueba123"
env = {**os.environ, "IG_ACCESS_TOKEN": SECRET, "IG_USER_ID": "1784", "GRAPH_BASE": f"http://127.0.0.1:{port}", "POLL_SECS": "0.1"}
open("/tmp/v.mp4", "wb").write(b"\x00" * 5000); open("/tmp/c.txt", "w").write("Pie de prueba #x")
def run(*extra):
    mock_ig.LOG.clear(); mock_ig.STATE["polls"] = 0
    p = subprocess.run([sys.executable, "scripts/ig_publish.py", "--video", "/tmp/v.mp4", "--caption-file", "/tmp/c.txt", *extra], env=env, capture_output=True, text=True); return p
# 1) dry-run: NO debe llamar a media_publish
p = run("--dry-run"); print(p.stdout, p.stderr)
assert p.returncode == 0 and "NO se ha publicado" in p.stdout
assert not any(l[1].endswith("media_publish") for l in mock_ig.LOG), "dry-run publicó"
post = [l for l in mock_ig.LOG if l[0] == "POST"][0]; assert post[2]["media_type"] == "REELS" and post[2]["upload_type"] == "resumable" and post[2]["is_ai_generated"] == "true"
assert SECRET not in post[3], "el token viajó en la URL del POST"
up = [l for l in mock_ig.LOG if l[0] == "UPLOAD"][0][2]; assert up["auth"] == f"OAuth {SECRET}" and up["offset"] == "0" and up["file_size"] == "5000" and up["len"] == 5000 and SECRET not in up["query"]
assert SECRET not in p.stdout + p.stderr, "el token aparece en la salida"
# 2) publicación completa
p = run(); print(p.stdout, p.stderr); assert p.returncode == 0 and "Publicado: id=M1" in p.stdout and "reel/TEST" in p.stdout
assert any(l[1].endswith("media_publish") for l in mock_ig.LOG); assert SECRET not in p.stdout + p.stderr
# 3) sin etiqueta IA
run("--dry-run", "--no-ai-label"); post = [l for l in mock_ig.LOG if l[0] == "POST"][0]; assert "is_ai_generated" not in post[2]
# 4) error de la API: el token no debe filtrarse en el mensaje
env["GRAPH_VERSION"] = "v23.0"; env2 = {**env, "IG_USER_ID": "9999"}
p = subprocess.run([sys.executable, "scripts/ig_publish.py", "--video", "/tmp/v.mp4", "--caption-file", "/tmp/c.txt", "--dry-run"], env=env2, capture_output=True, text=True)
print("salida de error esperada:", p.stderr.strip()[:200]); assert p.returncode != 0 and SECRET not in p.stdout + p.stderr
print("TODAS LAS PRUEBAS OK")
