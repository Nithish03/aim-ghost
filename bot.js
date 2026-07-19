// aimghost — bot arena plumbing.
// Renders a second "ghost" cursor that plays the live game through a pluggable
// brain function. The default brain is a dumb straight-line mover so the
// plumbing can be tested; the real brain — the neural network — is written by
// hand in /ml and plugged in via Bot.setBrain().
//
// Brain interface (this is the contract /ml code must satisfy):
//   brain(state, dt) -> { dx, dy, click }
//     state = {
//       cursor: { x, y },          // bot cursor position, CSS px
//       target: { x, y, r },       // current live target
//       screen: { w, h },          // current window size
//     }
//     dt    = seconds since last call (variable, ~1/refresh-rate)
//     dx,dy = movement to apply this tick, in px
//     click = true to press the trigger this tick

"use strict";

const Bot = (() => {
  let active = false;
  let pos = { x: 0, y: 0 };
  let lastTs = null;
  let brain = defaultBrain;

  // Placeholder brain: move straight at the target at a fixed speed, click
  // when close to the center. Deliberately robotic — it exists to prove the
  // plumbing, and as the baseline your hand-written network should crush.
  const BOT_SPEED = 700; // px/s
  function defaultBrain(state, dt) {
    const dxT = state.target.x - state.cursor.x;
    const dyT = state.target.y - state.cursor.y;
    const dist = Math.hypot(dxT, dyT) || 1;
    const step = Math.min(BOT_SPEED * dt, dist);
    return {
      dx: (dxT / dist) * step,
      dy: (dyT / dist) * step,
      click: dist < state.target.r * 0.5,
    };
  }

  function setBrain(fn) {
    brain = fn || defaultBrain;
  }

  function start() {
    active = true;
    pos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    lastTs = null;
  }

  function stop() {
    active = false;
  }

  // Called from the game's draw loop each frame while active.
  // Returns true if the bot clicked inside the target this tick.
  function update(ts, target) {
    if (!target) return false;
    if (lastTs === null) {
      lastTs = ts;
      return false;
    }
    const dt = Math.min((ts - lastTs) / 1000, 0.1); // clamp tab-switch gaps
    lastTs = ts;

    const res = brain(
      {
        cursor: { x: pos.x, y: pos.y },
        target: { x: target.x, y: target.y, r: target.r },
        screen: { w: window.innerWidth, h: window.innerHeight },
      },
      dt
    );
    pos.x = Math.max(0, Math.min(window.innerWidth, pos.x + res.dx));
    pos.y = Math.max(0, Math.min(window.innerHeight, pos.y + res.dy));

    if (res.click) {
      const dx = pos.x - target.x;
      const dy = pos.y - target.y;
      return dx * dx + dy * dy <= target.r * target.r;
    }
    return false;
  }

  function draw(ctx) {
    // Subtle by design — the ghost should be visible, not distracting.
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(120, 200, 255, 0.35)";
    ctx.fill();
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 10, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(120, 200, 255, 0.15)";
    ctx.stroke();
  }

  return {
    get active() { return active; },
    start,
    stop,
    update,
    draw,
    setBrain,
  };
})();
