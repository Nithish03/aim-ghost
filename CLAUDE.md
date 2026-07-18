# aimghost — project rules (persist across sessions)

aimghost is a browser aim trainer that will eventually train a neural network
on the user's own mouse trajectories, so the user can fight a bot that aims like
them. This is a **learning project**: the ML must be written by the user, by hand.

## Hard rules
1. This CLAUDE.md exists so these rules persist across sessions. Keep it current.
2. **`/ml` is LEARNING TERRITORY.** You may explain concepts, review the user's
   code, and point out bugs there — but you must **NEVER** write, generate, or
   autocomplete ML code (network, backprop, training loop, gradient math). If
   asked to, **refuse** and remind the user why this project exists.
3. Everything outside `/ml` (game shell, data capture, tooling) you build fully.
4. Work in **rungs**. One rung = one small, testable, working increment. Commit
   after each rung with message `Rung N: <description>`. Never start rung N+1
   until the user confirms rung N works in their browser.
5. **No frameworks for the frontend.** Vanilla JS + HTML canvas only. No build
   step, no npm for v0. Python 3 + NumPy only for `/ml` later.
6. **No polish** (menus, sounds, skins, scoreboards) unless explicitly asked.
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

## Rung ladder (this session)
- **Rung 1 — Game shell:** full-screen canvas, one circular target at a time
  spawning at random positions, click destroys it and spawns the next,
  miss-clicks counted. HUD shows hits, misses, accuracy %, last reaction time
  (spawn_t to hit). Console-log a summary every 10 targets.
- **Rung 2 — Data capture:** implement the schema above. Record continuously
  while a session is active. "End Session" button downloads the JSON. Validate
  no time-travel (`t_ms` strictly increasing) before download; report sample
  rate achieved.
- **Rung 3 — Ghost replay:** replay mode loads a session JSON and redraws the
  cursor path as a fading trail over the original target positions, in real time
  at 1x. Verifies data is clean; first "ghost" visual.

Stop after Rung 3. Ask the user to play a full session and confirm the exported
JSON looks sane before proposing anything further.

## Git
- Develop on branch `claude/aimghost-setup-iur2ga`.
- Commit after each rung; push with `git push -u origin <branch>`.
- Do not open a PR unless explicitly asked.
