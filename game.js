// aimghost — Rung 1: game shell.
// One circular target at a time; click destroys it and spawns the next.
// No data recording yet (that's Rung 2).

"use strict";

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const hudHits = document.getElementById("hud-hits");
const hudMisses = document.getElementById("hud-misses");
const hudAcc = document.getElementById("hud-acc");
const hudRt = document.getElementById("hud-rt");

const TARGET_RADIUS = 28; // px (CSS pixels)
const SPAWN_MARGIN = 12;  // keep the full circle on-screen

// --- state ---
let target = null; // { id, x, y, r, spawnT }
let nextTargetId = 0;
let hits = 0;
let misses = 0;
let lastReactionMs = null;
let reactionTimes = []; // reaction times of the last <=10 hits, for the summary log

// --- canvas sizing (DPR-aware; draw in CSS pixel coordinates) ---
function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // If a resize pushed the target off-screen, clamp it back into view.
  if (target) {
    target.x = Math.min(target.x, window.innerWidth - target.r - SPAWN_MARGIN);
    target.y = Math.min(target.y, window.innerHeight - target.r - SPAWN_MARGIN);
  }
}
window.addEventListener("resize", resize);
resize();

// --- targets ---
function spawnTarget() {
  const r = TARGET_RADIUS;
  const minX = r + SPAWN_MARGIN;
  const maxX = window.innerWidth - r - SPAWN_MARGIN;
  const minY = r + SPAWN_MARGIN;
  const maxY = window.innerHeight - r - SPAWN_MARGIN;
  target = {
    id: nextTargetId++,
    x: minX + Math.random() * (maxX - minX),
    y: minY + Math.random() * (maxY - minY),
    r,
    spawnT: performance.now(),
  };
  Recorder.onTargetSpawn(target);
}

// --- input ---
// Replay is defined in replay.js, which loads after this script; a frame or
// event can fire in between, so check it exists before touching it.
function replayActive() {
  return typeof Replay !== "undefined" && Replay.active;
}

// Duel mode: player and ghost race for the same target, first click wins.
let duel = false;
let duelYou = 0;
let duelGhost = 0;

canvas.addEventListener("pointerdown", (e) => {
  if (replayActive() || (Bot.active && !duel)) return;
  if (e.button !== 0 || !target) return;
  const dx = e.clientX - target.x;
  const dy = e.clientY - target.y;
  if (dx * dx + dy * dy <= target.r * target.r) {
    hits++;
    lastReactionMs = performance.now() - target.spawnT;
    reactionTimes.push(lastReactionMs);
    if (duel) duelYou++;
    Recorder.onTargetEnd(target, "hit");
    spawnTarget();
    if (!duel && hits % 10 === 0) logSummary();
  } else {
    misses++;
  }
  updateHud();
});

// --- HUD ---
function updateHud() {
  hudHits.textContent = hits;
  hudMisses.textContent = misses;
  const total = hits + misses;
  hudAcc.textContent = total ? ((hits / total) * 100).toFixed(1) + "%" : "–";
  hudRt.textContent = lastReactionMs !== null ? lastReactionMs.toFixed(0) + " ms" : "–";
  if (duel) {
    document.getElementById("hud-you").textContent = duelYou;
    document.getElementById("hud-ghost").textContent = duelGhost;
  }
}

function logSummary() {
  const total = hits + misses;
  const avgRt = reactionTimes.reduce((a, b) => a + b, 0) / reactionTimes.length;
  console.log(
    `[aimghost] ${hits} hits, ${misses} misses, ` +
    `accuracy ${((hits / total) * 100).toFixed(1)}%, ` +
    `avg reaction (last ${reactionTimes.length} hits) ${avgRt.toFixed(0)} ms`
  );
  reactionTimes = [];
}

// --- render loop ---
function draw(ts) {
  if (replayActive()) {
    // Replay owns the canvas; just keep the loop alive.
    requestAnimationFrame(draw);
    return;
  }
  ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
  if (typeof Bot !== "undefined" && Bot.active) {
    if (Bot.update(ts, target)) {
      // Bot clicked inside the target. No recording either way (the Recorder
      // session is aborted while the bot is on screen).
      if (duel) {
        duelGhost++;
      } else {
        hits++;
        lastReactionMs = performance.now() - target.spawnT;
        reactionTimes.push(lastReactionMs);
      }
      spawnTarget();
      updateHud();
    }
    Bot.draw(ctx);
  }
  if (target) {
    ctx.beginPath();
    ctx.arc(target.x, target.y, target.r, 0, Math.PI * 2);
    ctx.fillStyle = "#e5484d";
    ctx.fill();
    // center dot as a precise aim reference
    ctx.beginPath();
    ctx.arc(target.x, target.y, 2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
  }
  requestAnimationFrame(draw);
}

// --- session control ---
function startSession() {
  hits = 0;
  misses = 0;
  lastReactionMs = null;
  reactionTimes = [];
  Recorder.start();
  spawnTarget();
  updateHud();
}

document.getElementById("end-session").addEventListener("click", () => {
  Recorder.end(); // validates, reports, downloads JSON
  startSession(); // immediately begin a fresh session
});

// Train Ghost: end the session, send it to the server, get a brain back,
// plug it into the bot. The whole practice->fight loop with no downloads.
const trainBtn = document.getElementById("train-ghost");
trainBtn.addEventListener("click", async () => {
  const hitCount = hits;
  const session = Recorder.take();
  if (!session || hitCount < 10) {
    alert("Practice first: hit at least 10 targets, then press Train Ghost.");
    if (session) startSession(); // don't leave recording stopped
    return;
  }
  trainBtn.disabled = true;
  trainBtn.textContent = "Training…";
  try {
    // Self-heal a flaky page load (e.g. host waking from sleep): if brain.js
    // never arrived, fetch it now instead of failing.
    if (typeof BrainLoader === "undefined") {
      await new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = "brain.js";
        s.onload = resolve;
        s.onerror = () => reject(new Error("could not load brain.js — refresh the page"));
        document.body.appendChild(s);
      });
    }
    const res = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(session),
    });
    // A proxy timeout (slow free-tier host) returns an HTML error page, not
    // JSON — turn that into a human message instead of a parse error.
    const text = await res.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      throw new Error("the server took too long (free hosting is slow when busy) — try again in a minute");
    }
    if (!res.ok) throw new Error(data.error || "server error " + res.status);
    Bot.setBrain(BrainLoader.build(data));
    document.getElementById("load-brain").textContent = "Brain: yours";
    alert("Ghost trained on your play. Press \"Duel\" to fight yourself.");
  } catch (err) {
    alert("Training failed: " + err.message +
          "\n(Are you running via server.py? Static hosting can't train.)");
  }
  trainBtn.disabled = false;
  trainBtn.textContent = "Train Ghost";
  startSession();
});

const botBtn = document.getElementById("bot-toggle");
const duelBtn = document.getElementById("duel-toggle");
const hudDuel = document.getElementById("hud-duel");

function stopBotModes() {
  Bot.stop();
  duel = false;
  botBtn.textContent = "Bot: off";
  duelBtn.textContent = "Duel: off";
  hudDuel.hidden = true;
}

botBtn.addEventListener("click", () => {
  if (Bot.active) {
    stopBotModes();
    startSession(); // resume human play with a fresh recording
  } else {
    Recorder.abort(); // never record while a bot is on screen
    Bot.start();
    botBtn.textContent = "Bot: ON";
    hits = 0; misses = 0; lastReactionMs = null; reactionTimes = [];
    spawnTarget();
    updateHud();
  }
});

duelBtn.addEventListener("click", () => {
  if (duel) {
    stopBotModes();
    startSession();
  } else {
    Recorder.abort(); // duel play is never recorded as training data
    Bot.start();
    duel = true;
    duelYou = 0; duelGhost = 0;
    hits = 0; misses = 0; lastReactionMs = null; reactionTimes = [];
    duelBtn.textContent = "Duel: ON";
    botBtn.textContent = "Bot: off";
    hudDuel.hidden = false;
    spawnTarget();
    updateHud();
  }
});

startSession();
requestAnimationFrame(draw);
