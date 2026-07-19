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

Deploy configs for the common free tiers are checked in — the server binds
`0.0.0.0` and honors `$PORT`.

**Render (easiest, free):**
1. https://dashboard.render.com -> **New** -> **Blueprint**.
2. Connect this GitHub repo. `render.yaml` does the rest.
3. Share the `https://aimghost-<something>.onrender.com` URL.

Free tier sleeps after ~15 min idle; the first visit after that takes ~30 s
to wake. Fine for friends.

**Railway** (`Procfile`): New Project -> Deploy from GitHub repo.

**Fly.io / Cloud Run / anything Docker** (`Dockerfile`):
`fly launch` or `gcloud run deploy --source .`.

**Your own box + a tunnel** (no signup, ephemeral): `python3 server.py` then
`ngrok http 8000` and share the URL.

Each visitor trains on their own play; sessions are never stored — the brain
comes back in the HTTP response. The one exception is the **ghost gallery**:
Upload Ghost stores your named brain on the server so friends can fight it
via the Ghosts button. On free tiers the disk is ephemeral, so the gallery
resets when the instance restarts — re-upload after a quiet spell.

## Other buttons

- **End Session** — download your raw session JSON (the training-data schema;
  see CLAUDE.md). Bank these and batch-train a stronger ghost:
  `python3 tools/merge_sessions.py sessions/ -o dataset.json &&
  python3 ml/train.py dataset.json -o ml/brain.json`
- **Load Brain** — load a `brain.json` trained offline.
- **Load Replay** — watch a recorded session as a ghost-trail replay.
- **Bot** — the bot plays solo (straight-line baseline brain unless a trained
  one is loaded).
