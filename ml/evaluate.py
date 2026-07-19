"""aimghost rollout evaluation + speed calibration.

Mirrors brain.js inference exactly (target frame, input clipping, speed
governor) so what we measure here is what the browser ghost does. Used by
train.py to (a) report reach-rate/timing and (b) calibrate `out_scale` so the
ghost's cruise speed matches the user's instead of the model's optimism.
"""

import numpy as np

CLIP = 3.0        # input clip, in d_scale units (must match brain.js)
CAP_ABS = 3.0     # absolute step cap, in d_scale units (must match brain.js)
CAP_DIST = 0.35   # distance-proportional step cap (must match brain.js)


def rollout(params, norm, start, target, out_scale=1.0, max_ticks=900,
            rng=None):
    """Simulate the browser rollout. Returns ticks to reach, or None.

    Mirrors brain.js including the anti-stall history jitter — without it,
    slower (human-matched) speeds look like failures here and the calibrator
    is forced to pick aimbot speeds.
    """
    W1, b1, W2, b2 = params
    K = norm["history"]
    d_scale, p_scale = norm["d_scale"], norm["p_scale"]
    cap_abs = norm.get("cap_abs", CAP_ABS)
    if rng is None:
        rng = np.random.default_rng(0)
    pos = np.array(start, float)
    hist = np.zeros(2 * K)
    stall_ticks = 0
    for t in range(max_ticks):
        rel = np.array(target) - pos
        dist = np.hypot(*rel)
        if dist < 28:
            return t
        u = rel / dist
        R = np.array([[u[0], u[1]], [-u[1], u[0]]])
        hr = np.clip((R @ hist.reshape(K, 2).T).T.ravel() / d_scale, -CLIP, CLIP)
        x = np.concatenate([hr, [dist / p_scale]])
        out = np.tanh(x @ W1 + b1) @ W2 + b2
        d = (R.T @ out) * d_scale * out_scale
        step = np.hypot(*d)
        cap = min(cap_abs * d_scale, max(2.0, CAP_DIST * dist))
        if step > cap:
            d *= cap / step
        pos += d
        hist = np.concatenate([hist[2:], d])
        # anti-stall, as in brain.js (0.25 s at 10 ms ticks = 25 ticks)
        if np.hypot(*d) < 0.3 and dist > 14:
            stall_ticks += 1
            if stall_ticks > 25:
                hist = hist + rng.normal(0, 0.15 * d_scale, hist.shape)
        else:
            stall_ticks = 0
    return None


def evaluate(params, norm, out_scale=1.0, n_cases=50, seed=42):
    """Reach rate + median speed (px/ms) over random start/target pairs."""
    rng = np.random.default_rng(seed)
    speeds, times, fails = [], [], 0
    for _ in range(n_cases):
        s = rng.uniform([50, 50], [1900, 1000])
        tg = rng.uniform([50, 50], [1900, 1000])
        dist = np.hypot(*(s - tg))
        if dist < 100:
            continue
        t = rollout(params, norm, s, tg, out_scale)
        if t is None or t == 0:
            fails += 1
        else:
            times.append(t * norm["dt_ms"])
            speeds.append(dist / (t * norm["dt_ms"]))
    n = len(times) + fails
    return {
        "reach_rate": len(times) / n if n else 0.0,
        "median_ms": float(np.median(times)) if times else None,
        "median_px_ms": float(np.median(speeds)) if speeds else None,
    }


def calibrate_out_scale(params, norm, user_px_ms):
    """Find out_scale matching ghost cruise speed to the user's, by scan.

    Speed is not monotonic in out_scale (the governor kicks in and out), so
    bisection mis-converges; a direct scan is cheap and robust. Only scales
    with 100% reach rate qualify — matching speed is pointless if the ghost
    stops converging.
    """
    best, best_err = 1.0, float("inf")
    for scale in np.arange(0.5, 1.15, 0.05):
        r = evaluate(params, norm, out_scale=float(scale), n_cases=30)
        if r["reach_rate"] < 1.0 or r["median_px_ms"] is None:
            continue
        err = abs(r["median_px_ms"] - user_px_ms)
        if err < best_err:
            best, best_err = float(scale), err
    return best
