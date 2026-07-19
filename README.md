# aim-ghost

Browser-based aim trainer that learns *your* aim. Practice for a minute, press
**Train Ghost**, and a hand-rolled neural network (NumPy, no frameworks) is
trained on your mouse trajectories on the server. Then press **Duel** and
fight a ghost that moves, reacts, and aims like you.

## Run it

```bash
pip install numpy
python3 server.py          # -> http://localhost:8000
```

That's the whole stack: vanilla JS + canvas frontend, one stdlib Python
server, NumPy for training. No build step.

## Play

1. Click targets. Hit at least ~10.
2. **Train Ghost** — your session is sent to the server, trained (~10 s),
   and the resulting brain is loaded into the bot automatically.
3. **Duel: ON** — you and your ghost race for the same target; first click
   wins the point. Scoreboard is in the HUD.

The ghost gets your movement style, your reaction time (measured from your
own play), and your cursor speed — recalibrated on every retrain.

## Hosting for friends

Any host that runs Python works — the server binds `0.0.0.0` and honors
`$PORT`:

- **Render / Railway / Fly**: point it at this repo; start command
  `python3 server.py`, dependency `numpy` (requirements.txt is here).
- **Your own box + a tunnel** (quickest): `python3 server.py` then
  `ngrok http 8000` (or Tailscale/Cloudflare Tunnel) and share the URL.

Each visitor trains on their own play; nothing is stored server-side — the
brain comes back in the HTTP response.

## Other buttons

- **End Session** — download your raw session JSON (the training-data schema;
  see CLAUDE.md). Bank these and batch-train a stronger ghost:
  `python3 tools/merge_sessions.py sessions/ -o dataset.json &&
  python3 ml/train.py dataset.json -o ml/brain.json`
- **Load Brain** — load a `brain.json` trained offline.
- **Load Replay** — watch a recorded session as a ghost-trail replay.
- **Bot** — the bot plays solo (straight-line baseline brain unless a trained
  one is loaded).
