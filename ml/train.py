"""aimghost trainer: hand-rolled MLP + backprop + Adam, NumPy only.

Usage:
    python3 ml/train.py <session-or-dataset.json> [-o brain.json]

Trains a one-hidden-layer tanh MLP to predict the next cursor delta from
recent movement + relative target position, then writes the weights and
normalization constants to a JSON the browser bot can load.
"""

import argparse
import json

import numpy as np

from dataset import load_sessions, train_val_split, build_examples, reaction_stats
from evaluate import evaluate, calibrate_out_scale

HIDDEN = 64
EPOCHS = 300
BATCH = 128
LR = 1e-3
SEED = 7


def init_layer(rng, n_in, n_out):
    # Xavier: keeps activation variance stable through tanh.
    limit = np.sqrt(6.0 / (n_in + n_out))
    return rng.uniform(-limit, limit, (n_in, n_out)), np.zeros(n_out)


def forward(params, X):
    W1, b1, W2, b2 = params
    h_pre = X @ W1 + b1
    h = np.tanh(h_pre)
    out = h @ W2 + b2
    return out, h


def loss_and_grads(params, X, Y):
    W1, b1, W2, b2 = params
    n = len(X)
    out, h = forward(params, X)
    err = out - Y                      # (n, 2)
    loss = float((err ** 2).mean())

    # Backprop. MSE here is mean over BOTH dims: dL/dout = 2*err / (n*2).
    d_out = 2.0 * err / err.size
    dW2 = h.T @ d_out
    db2 = d_out.sum(axis=0)
    d_h = d_out @ W2.T
    d_pre = d_h * (1.0 - h ** 2)       # tanh'
    dW1 = X.T @ d_pre
    db1 = d_pre.sum(axis=0)
    return loss, [dW1, db1, dW2, db2]


def grad_check(params, X, Y, eps=1e-6):
    """Numerical vs analytic gradient on a few random weights. Trust nothing."""
    _, grads = loss_and_grads(params, X, Y)
    rng = np.random.default_rng(0)
    worst = 0.0
    for p, g in zip(params, grads):
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p.shape)
            orig = p[idx]
            p[idx] = orig + eps
            lp, _ = loss_and_grads(params, X, Y)
            p[idx] = orig - eps
            lm, _ = loss_and_grads(params, X, Y)
            p[idx] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(abs(num), abs(g[idx]), 1e-12)
            worst = max(worst, abs(num - g[idx]) / denom)
    return worst


def adam_step(params, grads, m, v, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for i, (p, g) in enumerate(zip(params, grads)):
        m[i] = b1 * m[i] + (1 - b1) * g
        v[i] = b2 * v[i] + (1 - b2) * g * g
        mh = m[i] / (1 - b1 ** t)
        vh = v[i] / (1 - b2 ** t)
        p -= lr * mh / (np.sqrt(vh) + eps)


def train_model(sessions, epochs=EPOCHS, verbose=True):
    """Full pipeline: sessions -> trained + calibrated brain dict.

    Raises ValueError for data problems (not enough play), so callers — the
    CLI below or server.py — can show a friendly message.
    """
    def log(msg):
        if verbose:
            print(msg)

    train_segs, val_segs = train_val_split(sessions)
    if len(train_segs) < 6 or len(val_segs) < 1:
        raise ValueError("Not enough data: hit at least 10 targets, then train.")
    Xtr, Ytr, norm = build_examples(train_segs)
    Xva, Yva, _ = build_examples(val_segs, norm=norm)
    norm.update(reaction_stats(train_segs))
    log(f"reaction: median {norm['reaction_ms_median']:.0f} ms "
        f"(std {norm['reaction_ms_std']:.0f})")

    log(f"train: {Xtr.shape[0]} examples from {len(train_segs)} segments; "
        f"val: {Xva.shape[0]} from {len(val_segs)}")
    log(f"norm: d_scale={norm['d_scale']:.3f} px/tick, dt={norm['dt_ms']} ms")

    rng = np.random.default_rng(SEED)
    n_in = Xtr.shape[1]
    W1, b1 = init_layer(rng, n_in, HIDDEN)
    W2, b2 = init_layer(rng, HIDDEN, 2)
    params = [W1, b1, W2, b2]

    err = grad_check(params, Xtr[:64], Ytr[:64])
    log(f"gradient check: max relative error {err:.2e} "
        f"({'OK' if err < 1e-4 else 'FAILED — do not trust this training'})")
    if err >= 1e-4:
        raise ValueError("Internal error: gradient check failed.")

    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    t = 0
    n = len(Xtr)
    best_val = float("inf")
    best_params = [p.copy() for p in params]
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        ep_loss = 0.0
        for s in range(0, n, BATCH):
            idx = order[s:s + BATCH]
            # Input noise: at rollout the model sees its own imperfect
            # history, never the pristine training history. Strong noise
            # teaches it to brake/correct from perturbed states instead of
            # orbiting. 0.25 chosen by sweep: 100% simulated reach rate,
            # median 395 ms (0.05 gave 88%, orbits).
            Xb = Xtr[idx] + rng.normal(0, 0.25, Xtr[idx].shape)
            loss, grads = loss_and_grads(params, Xb, Ytr[idx])
            t += 1
            adam_step(params, grads, m, v, t, LR)
            ep_loss += loss * len(idx)
        va_out, _h = forward(params, Xva)
        va_loss = float(((va_out - Yva) ** 2).mean())
        if va_loss < best_val:
            best_val = va_loss
            best_params = [p.copy() for p in params]
            best_epoch = epoch
        if epoch % 25 == 0 or epoch == 1:
            log(f"epoch {epoch:4d}  train {ep_loss / n:.5f}  val {va_loss:.5f}")
    params = best_params
    W1, b1, W2, b2 = params
    log(f"keeping epoch {best_epoch} weights (best val {best_val:.5f})")

    # Calibrate ghost cruise speed to the user's real speed, then verify.
    dists = np.array([np.hypot(*(np.array(s["target"][:2]) - s["pos"][0]))
                      for s in train_segs])
    durs = np.array([len(s["pos"]) * norm["dt_ms"] for s in train_segs])
    user_px_ms = float(np.median(dists / durs))
    norm["out_scale"] = calibrate_out_scale(params, norm, user_px_ms)
    r = evaluate(params, norm, out_scale=norm["out_scale"])
    log(f"user speed {user_px_ms:.2f} px/ms -> out_scale {norm['out_scale']:.2f}; "
        f"sim: reach {r['reach_rate']*100:.0f}%, median {r['median_ms']:.0f} ms, "
        f"{r['median_px_ms']:.2f} px/ms")

    return {
        "kind": "aimghost-mlp",
        "act": "tanh",
        "norm": norm,
        "W1": W1.tolist(), "b1": b1.tolist(),
        "W2": W2.tolist(), "b2": b2.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="session JSON or merged dataset.json")
    ap.add_argument("-o", "--output", default="brain.json")
    args = ap.parse_args()
    brain = train_model(load_sessions(args.data))
    with open(args.output, "w") as f:
        json.dump(brain, f)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
