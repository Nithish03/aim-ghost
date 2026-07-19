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

    def do_POST(self):
        if self.path != "/api/train":
            self.send_json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "session too large"})
                return
            session = json.loads(self.rfile.read(length))
            if "trajectory" not in session or "targets" not in session:
                self.send_json(400, {"error": "not a session JSON"})
                return
            with train_lock:
                brain = train_model([session], verbose=False)
            self.send_json(200, brain)
        except ValueError as e:
            self.send_json(400, {"error": str(e)})
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_json(500, {"error": "training failed on the server"})


def main():
    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"aimghost serving on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
