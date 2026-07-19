# aimghost — project rules (persist across sessions)

aimghost is a browser aim trainer that trains a neural network on the user's
own mouse trajectories, so the user can fight a bot that aims like them.

> **2026-07-18:** This began as a learning project where `/ml` was
> user-written-only territory. The user explicitly and repeatedly rescinded
> that rule and asked Claude to build `/ml` and ship. The old rule is gone;
> the user plans a separate hand-written ML learning project later.

## Hard rules
1. This CLAUDE.md exists so these rules persist across sessions. Keep it current.
2. Claude builds everything, including `/ml` (Python 3 + NumPy only there —
   no torch/tf/sklearn; the net, backprop, and training loop are hand-rolled).
3. Work in **rungs**. One rung = one small, testable, working increment. Commit
   after each rung with message `Rung N: <description>`. Never start rung N+1
   until the user confirms rung N works in their browser.
4. **No frameworks for the frontend.** Vanilla JS + HTML canvas only. No build
   step, no npm for v0.
5. **No polish** (menus, sounds, skins, scoreboards) unless explicitly asked.
   The game is a data-collection instrument first.

## Data schema (LOCKED — do not change without discussing with the user)
All mouse data is recorded per-session as JSON:

```json
{
  "session_id": "<uuid>",
  "started_at": "<ISO8601>",
  "screen": { "w": "<px>", "h": "<px>", "dpr": "<devicePixelRatio>" },
  "refresh_hint_hz": 144,
  "targets": [
    { "target_id": "<int>", "spawn_t": "<ms>", "x": "<px>", "y": "<px>", "r": "<px>",
      "outcome": "hit | miss | timeout", "end_t": "<ms>" }
  ],
  "trajectory": [
    ["<t_ms>", "<x_px>", "<y_px>", "<buttons_bitmask>"]
  ]
}
```

- `t_ms` is `performance.now()` relative to session start (monotonic, **not**
  `Date.now()`).
- Record `mousemove` at native event rate; **do not throttle or smooth**. Use
  `pointerrawupdate` where available, falling back to `mousemove`, and unpack
  coalesced events (`getCoalescedEvents`) so no samples are dropped.
- Clicks are recorded in the trajectory via `buttons` bitmask **and** as target
  outcomes.

## Rung ladder — status
- **Rung 1 — Game shell:** DONE, user-confirmed. Canvas, targets, HUD.
- **Rung 2 — Data capture:** DONE, user-confirmed. Schema implemented in
  `recorder.js`; End Session validates monotonic t_ms + downloads JSON. A real
  user session validated clean (~123 Hz while moving).
- **Rung 3 — Ghost replay:** DONE, user-confirmed. `replay.js` — Load Replay
  button, fading trail, 1x playback, click rings.
- **Rung 4 — Dataset tooling:** `tools/merge_sessions.py` (stdlib-only CLI)
  validates and merges session JSONs into `dataset.json`. Sessions stay
  separate — never concatenate trajectories across time bases.
- **Rung 5 — Bot arena plumbing:** `bot.js` — ghost cursor plays the live game
  via pluggable `brain(state, dt) -> {dx, dy, click}` (contract documented in
  bot.js). Default brain = straight-line mover (plumbing test / baseline).
  Recording is aborted while the bot plays; bot data must never pollute human
  data. The real brain comes from `/ml` via `Bot.setBrain()`.

- **Rung 6 — Neural ghost (`/ml` + `brain.js`):** DONE. `ml/dataset.py`
  preprocesses sessions (segments per hit target, 10 ms resampling,
  reaction-time idle prefix trimmed to prevent freeze-fixpoints, delta +
  relative-target features, zero-padded history). `ml/train.py` is a
  hand-rolled NumPy MLP (12→64 tanh→2), backprop with gradient check, Adam,
  input-noise regularization, best-val checkpointing; exports `brain.json`.
  `brain.js` runs the forward pass in the browser at the model's fixed tick,
  with an anti-stall history perturbation; "Load Brain" button plugs it into
  `Bot.setBrain()`. Verified end-to-end: neural bot hits targets unassisted.
  `ml/brain.json` is trained on the user's one real session — retrain with
  more sessions via `python3 ml/train.py dataset.json -o ml/brain.json`.

- **Rung 7 — Fair fight:** DONE. Model reworked to a TARGET-CENTRIC frame
  (rotate so cursor->target = +x; fixes orbiting, generalizes directions).
  Rollout guards mirrored in `ml/evaluate.py` + `brain.js`: input clipping,
  distance-proportional speed governor. Training noise 0.25 (swept; 100%
  sim reach rate). Ghost gets the user's own reaction delay (median/std of
  trimmed idle prefixes) and auto-calibrated `out_scale` matching the user's
  px/ms. **Duel mode**: shared target, player vs ghost, first click wins,
  HUD scoreboard. Duel/bot play is never recorded.

Known limits: trained on ONE session. More recorded sessions + retraining
(`python3 ml/train.py dataset.json -o ml/brain.json`) is the highest-value
improvement; the calibration re-derives automatically on retrain.

## Git
- Develop on branch `claude/aimghost-setup-iur2ga`.
- Commit after each rung; push with `git push -u origin <branch>`.
- Do not open a PR unless explicitly asked.
