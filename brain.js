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
    const stepS = w.norm.dt_ms / 1000;
    // History of the bot's own last K deltas, normalized units. Starts at
    // rest (zeros) — the same zero-padding the training examples used.
    let hist = new Array(2 * K).fill(0);
    let acc = 0;
    let stallS = 0;

    function gauss(std) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += Math.random();
      return (s - 2) * std; // cheap approx-normal
    }

    function forwardStep(relX, relY) {
      const x = hist.concat([relX / pScale, relY / pScale]);
      const h = matvec(w.W1, x, w.b1).map(Math.tanh);
      const out = matvec(w.W2, h, w.b2);
      hist = hist.slice(2).concat(out); // slide the window
      return { dx: out[0] * dScale, dy: out[1] * dScale };
    }

    return function brain(state, dt) {
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
          for (let i = 0; i < hist.length; i++) hist[i] += gauss(0.15);
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
      if (w.kind !== "aimghost-mlp") {
        alert("Not an aimghost brain file (expected ml/train.py output).");
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
