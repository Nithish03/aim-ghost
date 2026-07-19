// aimghost — neural bot brain.
// Loads weights exported by ml/train.py (brain.json) and runs the MLP forward
// pass to drive the bot cursor. Mirrors the training-time representation
// exactly: same fixed timestep, same history length, same normalization.

"use strict";

const BrainLoader = (() => {
  function matvec(W, x, b) {
    // W is (n_in x n_out) as nested lists, matching NumPy's layout.
    const out = b.slice();
    for (let i = 0; i < x.length; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = W[i];
      for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
    }
    return out;
  }

  function build(w) {
    const K = w.norm.history;
    const dScale = w.norm.d_scale;
    const pScale = w.norm.p_scale;
    const outScale = w.norm.out_scale || 1.0; // speed calibration from train.py
    const stepS = w.norm.dt_ms / 1000;
    // History of the bot's own last K deltas, normalized units. Starts at
    // rest (zeros) — the same zero-padding the training examples used.
    let hist = new Array(2 * K).fill(0);
    let acc = 0;
    let stallS = 0;
    // Human-like reaction: on each new target, wait a delay sampled from the
    // user's own recorded reaction-time distribution, with a cleared history
    // (each recorded segment also started from rest).
    let curTarget = null;
    let reactS = 0;
    const rMed = (w.norm.reaction_ms_median || 0) / 1000;
    const rStd = (w.norm.reaction_ms_std || 0) / 1000;

    function gauss(std) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += Math.random();
      return (s - 2) * std; // cheap approx-normal
    }

    // Target-centric frame: rotate so cursor->target is +x, run the net,
    // rotate the predicted delta back to screen coordinates. The frame is
    // recomputed every tick, so direction error self-corrects. History is
    // kept in raw screen deltas and rotated per-tick into the current frame.
    function forwardStep(relX, relY) {
      const dist = Math.hypot(relX, relY);
      if (dist < 1e-6) return { dx: 0, dy: 0 };
      const ux = relX / dist, uy = relY / dist;
      const x = new Array(2 * K + 1);
      for (let i = 0; i < K; i++) {
        const hdx = hist[2 * i], hdy = hist[2 * i + 1];
        // Clip to the training range — out-of-distribution inputs are how
        // runaway feedback loops (orbiting) start.
        x[2 * i] = Math.max(-3, Math.min(3, (ux * hdx + uy * hdy) / dScale));
        x[2 * i + 1] = Math.max(-3, Math.min(3, (-uy * hdx + ux * hdy) / dScale));
      }
      x[2 * K] = dist / pScale;
      const h = matvec(w.W1, x, w.b1).map(Math.tanh);
      const out = matvec(w.W2, h, w.b2);
      let dx = (ux * out[0] - uy * out[1]) * dScale * outScale;
      let dy = (uy * out[0] + ux * out[1]) * dScale * outScale;
      // Speed governor: never arrive faster than the training data ever did.
      // Humans brake before the target; an un-braked overshoot puts the model
      // in states it has never seen.
      const step = Math.hypot(dx, dy);
      const cap = Math.min((w.norm.cap_abs || 3) * dScale, Math.max(2, 0.35 * dist));
      if (step > cap) {
        dx *= cap / step;
        dy *= cap / step;
      }
      hist = hist.slice(2).concat([dx, dy]); // raw screen-space history
      return { dx, dy };
    }

    return function brain(state, dt) {
      const tKey = state.target.x + "," + state.target.y;
      if (tKey !== curTarget) {
        curTarget = tKey;
        hist = new Array(2 * K).fill(0);
        acc = 0;
        stallS = 0;
        reactS = Math.max(0.08, rMed + gauss(rStd));
      }
      if (reactS > 0) {
        reactS -= dt;
        return { dx: 0, dy: 0, click: false };
      }
      // The model was trained at a fixed tick; step it as many times as the
      // elapsed frame time covers, accumulating movement.
      acc += dt;
      let dx = 0, dy = 0;
      let cx = state.cursor.x, cy = state.cursor.y;
      while (acc >= stepS) {
        acc -= stepS;
        const d = forwardStep(state.target.x - cx, state.target.y - cy);
        dx += d.dx; dy += d.dy;
        cx += d.dx; cy += d.dy;
      }
      const dist = Math.hypot(state.target.x - cx, state.target.y - cy);

      // Anti-stall: rollout drift can park the cursor near-motionless just
      // outside the target (a fixpoint the training data never had — humans
      // always finish). Perturb the history like a micro-correction so the
      // net re-engages; positions are never teleported.
      const speed = Math.hypot(dx, dy);
      if (speed < 0.3 && dist > state.target.r * 0.5) {
        stallS += dt;
        if (stallS > 0.25) {
          for (let i = 0; i < hist.length; i++) hist[i] += gauss(0.15) * dScale;
        }
      } else {
        stallS = 0;
      }

      return { dx, dy, click: dist < state.target.r };
    };
  }

  const input = document.getElementById("brain-file");
  document.getElementById("load-brain").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const file = input.files[0];
    input.value = "";
    if (!file) return;
    file.text().then((text) => {
      let w;
      try {
        w = JSON.parse(text);
      } catch {
        alert("Not valid JSON.");
        return;
      }
      if (w.kind !== "aimghost-mlp" || w.norm.frame !== "target") {
        alert("Not a current aimghost brain file — retrain with ml/train.py.");
        return;
      }
      Bot.setBrain(build(w));
      document.getElementById("load-brain").textContent = "Brain: neural";
      console.log(
        `[aimghost] neural brain loaded: ${w.W1.length}->${w.b1.length}->2, ` +
        `history ${w.norm.history}, dt ${w.norm.dt_ms} ms`
      );
    });
  });

  return { build };
})();
