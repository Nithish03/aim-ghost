# /ml roadmap — concepts, not code

This document explains the ideas you need for each ML rung. It contains **no
code** and never will: everything in `/ml` is written by your hand, in
Python 3 + NumPy only. Claude reviews your code and answers questions but does
not write it — that rule is in CLAUDE.md and it is the point of this project.

## The problem you are solving: behavioral cloning

You want a function that behaves like your hand. Formally: given what the
"player" can see at time t — where the cursor has recently been, and where the
target is — predict the movement over the next small timestep, `(dx, dy)`.

Train that function on your recorded sessions. Then, to make the ghost play,
you *roll it out*: start the cursor somewhere, feed the model its own output
back in step after step, and a full trajectory emerges. If training worked,
that trajectory accelerates, curves, overshoots, and corrects the way yours do.

The plumbing is already waiting: `bot.js` renders a ghost cursor and calls a
`brain(state, dt) -> {dx, dy, click}` function every frame. The default brain
is a straight-line mover. Your job, ultimately, is to replace it.

## ML Rung 0 — load and look (no ML)

Write a loader that reads a session JSON (or the merged `dataset.json` from
`tools/merge_sessions.py`) into NumPy arrays. Then interrogate it:

- How many samples? What is the distribution of time gaps `t[i+1] - t[i]`?
  (You already know the answer is bimodal: ~8 ms while moving, big gaps while
  idle. Your preprocessing must handle both.)
- Slice per-target **segments**: the trajectory between each target's
  `spawn_t` and `end_t`. Those segments — "cursor travels from rest to a
  click on the target" — are your real unit of data.
- Compute per-segment facts: path length vs straight-line distance (the ratio
  measures how curved/wasteful your movement is), duration, peak speed.

Why this rung exists: every ML disaster starts with not knowing your data.

## ML Rung 1 — preprocessing

Three ideas, all of which you implement yourself:

1. **Resampling.** Events arrive irregularly; models want a fixed timestep.
   Linearly interpolate each segment onto a uniform grid (10 ms is a fine
   start). Think about what interpolation does to clicks (a bitmask can't be
   interpolated — carry it, don't blend it).
2. **Deltas, not positions.** Predicting absolute pixel positions makes the
   model memorize your screen. Predicting `(dx, dy)` per step makes it learn
   *movement*. Relatedly, express the target relative to the cursor
   (`target_x - cursor_x`, …) so the model generalizes across screen
   locations.
3. **Normalization.** Inputs spanning [-2056, 2056] px produce huge gradients
   and unstable training. Scale features to roughly [-1, 1] (divide by screen
   size, or standardize: subtract mean, divide by std — computed on training
   data only). Save the constants; you need the exact same transform at
   inference time in the browser.

Your training set then is a big array of examples:
`features = [last N deltas..., relative target vector, maybe distance]` →
`label = next delta`.

## ML Rung 2 — linear regression, by hand

The dumbest model that can learn: `prediction = W · features + b`.

Concepts to learn here, in order:

- **Loss function.** Mean squared error: average of
  `‖prediction - label‖²` over examples. One number that says "how wrong".
- **Gradient.** The vector of partial derivatives of the loss with respect to
  every entry of `W` and `b`. It points uphill; step the other way.
  Derive it on paper for MSE + linear model — it's a clean chain-rule
  exercise and the answer is small enough to check by hand.
- **Gradient descent.** `W ← W - lr * dL/dW`, repeated. The learning rate
  `lr` is your first hyperparameter: too big diverges (loss explodes), too
  small crawls. Plot loss vs iteration; it should fall fast then flatten.
- **Sanity checks that will save you:** (1) numerical gradient check —
  nudge one weight by ±ε, recompute loss, compare the slope to your
  analytic gradient; they must agree to several decimals. (2) Overfit a
  tiny slice — 50 examples — to near-zero loss; if you can't, the bug is in
  your math, not your data.

A linear model rolled out will produce robotic near-straight ghosts.
**That is success**: the plumbing works end to end, and now the only thing
missing is nonlinearity.

## ML Rung 3 — a neural network, by hand

An MLP with one hidden layer:
`hidden = activation(W1 · x + b1)`, `output = W2 · hidden + b2`.

- **Activation** (tanh or ReLU) is what makes it more than linear regression.
  Understand why stacking linear layers without one collapses to a single
  linear map.
- **Backpropagation** is nothing but the chain rule applied layer by layer,
  caching forward-pass values so each gradient is cheap. Derive the two-layer
  case on paper before writing it. Your Rung 2 gradient checker carries over
  unchanged — use it relentlessly; it turns backprop from faith into fact.
- **Initialization** matters: all-zeros makes every hidden unit identical
  forever (symmetry never breaks). Small random values scaled by layer width
  (look up "Xavier initialization" and understand *why* the scale matters).
- **Train/validation split.** Hold out entire sessions, not random rows —
  rows within a session are correlated, and a random split leaks. Falling
  train loss with rising validation loss = overfitting; that's your signal to
  stop training, shrink the net, or record more sessions.

## ML Rung 4 — rollout and the arena

- **Rollout drift:** during training the model always saw *real* history; at
  rollout it sees its *own* slightly-wrong outputs, and errors compound. If
  ghosts fly off screen, this is why, and it's a known hard problem
  (look up "exposure bias"). Mitigations to explore: shorter history,
  predicting a few steps ahead, adding small noise to training inputs.
- **Evaluation:** compare rollouts to held-out real segments — endpoint
  error, duration, path-length ratio, speed profile. Also just *watch* them
  in the replay view; your eyes are a fine discriminator for "does this move
  like me".
- **Going live:** port your trained net's forward pass to JS and plug it into
  `Bot.setBrain()`. The forward pass is network code, so the port is yours
  too. Agree on a weights JSON format (shapes + arrays + your normalization
  constants); the game side that *loads* the file can be built for you —
  ask.
- **Clicking:** either a third output (click probability, threshold it) or a
  simple rule (click when within r/2, like the default brain). Start with the
  rule; earn the learned trigger later.

## Rules of engagement (same as ever)

Work in rungs, commit each (`ML Rung N: ...`), confirm before climbing.
Claude explains, reviews, and finds bugs — and refuses to write `/ml` code,
because a ghost you didn't build isn't yours.
