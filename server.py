#!/usr/bin/env python3
"""aimghost server: serves the game and trains ghosts.

    python3 server.py            # http://localhost:8000
    PORT=3000 python3 server.py

Static files come from this directory. POST /api/train takes a session JSON
(the recorder's schema) and returns a trained brain JSON, which the page
plugs straight into the bot — play, press "Train Ghost", then Duel yourself.

Stdlib + NumPy only, matching the rest of the project.
"""

import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "ml"))

from train import train_model  # noqa: E402

MAX_BODY = 50 * 1024 * 1024
# One training at a time keeps a small host responsive; NumPy releases the
# GIL poorly across threads anyway and a run only takes a few seconds.
train_lock = threading.Lock()

# Ghost gallery: named brains stored as JSON files. NOTE: on free-tier hosts
# the disk is ephemeral — ghosts survive while the instance is up and are
# lost on restart/redeploy. Newest 50 are kept.
GHOST_DIR = os.path.join(ROOT, "ghosts")
os.makedirs(GHOST_DIR, exist_ok=True)
MAX_GHOSTS = 50
ghost_lock = threading.Lock()


def ghost_files():
    files = [os.path.join(GHOST_DIR, f) for f in os.listdir(GHOST_DIR)
             if f.endswith(".json")]
    return sorted(files, key=os.path.getmtime, reverse=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/ghosts":
            out = []
            for f in ghost_files()[:MAX_GHOSTS]:
                try:
                    g = json.loads(open(f).read())
                    out.append({"id": os.path.basename(f)[:-5], "name": g["name"]})
                except (OSError, ValueError, KeyError):
                    continue
            self.send_json(200, {"ghosts": out})
        elif self.path.startswith("/api/ghosts/"):
            gid = self.path.rsplit("/", 1)[1]
            if not gid.replace("-", "").isalnum():
                self.send_json(400, {"error": "bad ghost id"})
                return
            f = os.path.join(GHOST_DIR, gid + ".json")
            if not os.path.isfile(f):
                self.send_json(404, {"error": "ghost not found (the gallery resets when the server restarts)"})
                return
            self.send_json(200, json.loads(open(f).read()))
        else:
            super().do_GET()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "request too large"})
                return
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/train":
                if "trajectory" not in body or "targets" not in body:
                    self.send_json(400, {"error": "not a session JSON"})
                    return
                with train_lock:
                    brain = train_model([body], verbose=False)
                self.send_json(200, brain)
            elif self.path == "/api/ghosts":
                name = str(body.get("name", "")).strip()[:24]
                brain = body.get("brain")
                if not name:
                    self.send_json(400, {"error": "give your ghost a name"})
                    return
                if not isinstance(brain, dict) or brain.get("kind") != "aimghost-mlp":
                    self.send_json(400, {"error": "not a trained ghost"})
                    return
                import uuid
                gid = uuid.uuid4().hex[:12]
                with ghost_lock:
                    with open(os.path.join(GHOST_DIR, gid + ".json"), "w") as f:
                        json.dump({"name": name, "brain": brain}, f)
                    for old in ghost_files()[MAX_GHOSTS:]:
                        os.remove(old)
                self.send_json(200, {"id": gid, "name": name})
            else:
                self.send_json(404, {"error": "unknown endpoint"})
        except ValueError as e:
            self.send_json(400, {"error": str(e)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_json(500, {"error": "server error"})


def main():
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"aimghost serving on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
