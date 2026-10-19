# Solving a Second-Order ODE with a Physics-Informed Neural Network (PINN)

This project trains a small neural network to solve the (optionally damped)
mass–spring equation

$$
m\,x''(t) + c\,x'(t) + k\,x(t) = 0,\qquad x(t_0)=x_0,\qquad x'(t_0)=v_0,\qquad t\in[t_0,t_{\text{end}}]
$$

**without ever showing it the answer.** The network learns only from the
equation and the two initial conditions. The analytical solution (in
`physics.py`) is used afterwards, only to measure how well it did.

The default configuration ($m=k=1$, $c=0$, $x_0=1$, $v_0=0$, $t\in[0,2\pi]$)
is the original harmonic oscillator $x''+x=0$ with exact solution $\cos(t)$.
Sections 1–10 below explain the method using that default case.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Runs on a normal CPU (no GPU needed). Expect roughly 30 s – 2 min depending on
your computer. Results appear in `outputs/`.

Run the unit tests (a few seconds; no full training):

```bash
python -m unittest discover -s tests -v
```

## What to edit (all in `config.py`)

| To change… | Edit |
|---|---|
| Mass [kg] | `MASS` (> 0) |
| Spring constant [N/m] | `SPRING_CONSTANT` (> 0) |
| Damping [N·s/m] | `DAMPING_COEFFICIENT` (>= 0) |
| Initial position [m] / velocity [m/s] | `INITIAL_POSITION`, `INITIAL_VELOCITY` |
| Simulation interval [s] | `TIME_START`, `TIME_END` (e.g. `10 * math.pi`) |
| Collocation / evaluation points | `N_COLLOCATION` (e.g. 50, 200, 500, 1000), `N_EVALUATION_POINTS` |
| Network & optimiser | `HIDDEN_SIZE`, `NUM_HIDDEN_LAYERS`, `EPOCHS`, `LEARNING_RATE`, `SEED` |
| Loss weights | `LAMBDA_PHYSICS`, `LAMBDA_IC` |
| Best-weights restore | `KEEP_BEST_WEIGHTS` |
| Time-input scaling | `NORMALIZE_TIME_INPUT` (see below) |
| Training animation / snapshots | `ENABLE_TRAINING_VISUALIZATION`, `VISUALIZATION_INTERVAL`, `VISUALIZATION_POINTS`, `ANIMATION_INTERVAL_MS`, `ANIMATION_FORMAT` |

Damping regimes, with critical damping $c_\text{crit}=2\sqrt{mk}$
(= 2.0 when $m=k=1$):

| Regime | Condition | Example (m = k = 1) |
|---|---|---|
| Undamped | $c=0$ | `DAMPING_COEFFICIENT = 0.0` |
| Underdamped | $0<c<2\sqrt{mk}$ | `DAMPING_COEFFICIENT = 0.4` |
| Critically damped | $c=2\sqrt{mk}$ | `DAMPING_COEFFICIENT = 2.0` |
| Overdamped | $c>2\sqrt{mk}$ | `DAMPING_COEFFICIENT = 5.0` |

To run other configurations from a script without editing `config.py`:

```python
import main
main.main({"DAMPING_COEFFICIENT": 0.4, "TIME_END": 20.0, "OUTPUT_DIR": "outputs_underdamped"})
```

**`NORMALIZE_TIME_INPUT`** (default `False` = the original network): when
`True`, the network maps time from $[t_0, t_\text{end}]$ to $[-1,1]$ as its
first operation, which keeps `tanh` inputs small for a large `TIME_START`.
Because the scaling happens inside the network, autograd still returns
derivatives with respect to the physical time in seconds. In our tests it did
**not** fix long intervals (both settings failed at $10\pi$) and was slightly
less accurate on $[0,2\pi]$ (RMSE 1.7e-4 vs 8.5e-5), so it is off by default.

## The analytical solution (validation only)

With $\tau=t-t_0$, $\omega_n=\sqrt{k/m}$, $\zeta=c/(2\sqrt{mk})$:

* **Undamped** ($c=0$): $x=x_0\cos\omega_n\tau+\frac{v_0}{\omega_n}\sin\omega_n\tau$
* **Underdamped**: $x=e^{-\zeta\omega_n\tau}\left[x_0\cos\omega_d\tau+\frac{v_0+\zeta\omega_n x_0}{\omega_d}\sin\omega_d\tau\right]$, with $\omega_d=\omega_n\sqrt{1-\zeta^2}$
* **Critically damped**: $x=\left[x_0+(v_0+\omega_n x_0)\tau\right]e^{-\omega_n\tau}$
* **Overdamped**: $x=Ae^{r_1\tau}+Be^{r_2\tau}$, $r_{1,2}=-\zeta\omega_n\pm\omega_n\sqrt{\zeta^2-1}$, $A=\frac{v_0-r_2x_0}{r_1-r_2}$, $B=\frac{r_1x_0-v_0}{r_1-r_2}$

"Critically damped" is detected with a relative tolerance of $10^{-9}$ on
$c = 2\sqrt{mk}$. Invalid parameters raise `ValueError`, and a non-finite
result raises an error rather than returning wrong numbers. `pinn_model.py`
never imports `physics.py`, and the training loss (`compute_training_losses()`
in `main.py`) contains only the physics residual and the initial conditions.
A unit test checks that training runs with the analytical functions disabled.

## Training visualization: watching the PINN learn

With `ENABLE_TRAINING_VISUALIZATION = True` (section 4 of `config.py`), the
project also saves:

* **`outputs/training_snapshots.png`**: one panel per milestone epoch
  (0, 1, 10, 100, 500, 1000, 2500, 5000, 7500, whichever are ≤ `EPOCHS`), plus
  the final epoch and the restored best weights. Each panel shows the PINN
  against the analytical curve, with RMSE and loss.
* **`outputs/solution_evolution.gif`** (or `.mp4`): an animation. On the left,
  the PINN curve (red dashed) moves while the analytical curve (black) stays
  fixed; the title shows the epoch, the weighted total loss, the physics and IC
  losses, and the RMSE. On the right, the loss curves are drawn up to the
  current epoch. The last frames show the final weights and then the restored
  best weights (green title), so the two are never confused.

| Setting | Meaning |
|---|---|
| `ENABLE_TRAINING_VISUALIZATION` | `False` = training exactly as before, no extra files |
| `VISUALIZATION_INTERVAL` | snapshot every N epochs (plus the milestones above) |
| `VISUALIZATION_POINTS` | time points per snapshot curve |
| `ANIMATION_INTERVAL_MS` | delay between animation frames (playback speed only, not training) |
| `ANIMATION_FORMAT` | `"gif"` (Pillow, always available) or `"mp4"` (needs ffmpeg; falls back to GIF) |

**What "epoch N" means in the snapshots:** the weights after N optimizer
updates. Epoch 0 is the real, randomly initialised network before any update.
The training log and `best_epoch` in `metrics.json` count loop iterations
instead (iteration N sees the weights after N − 1 updates), so the best
weights logged as "epoch 9915" appear in the snapshots as epoch 9914.

**How snapshots are captured.** Inside the training loop, at each snapshot
epoch, the current network is evaluated once on a fixed grid under
`model.eval()` + `torch.no_grad()`. Only that prediction array and the loss
values already computed in that iteration are stored. No weights are copied,
nothing is retrained, and the optimizer, losses and weights are untouched.
The analytical curve and the RMSE are computed only after training. Runs with
more than 300 snapshots are thinned evenly in the animation to keep the GIF
small; the static figure always uses the milestone epochs.

## Project structure

```text
pinn_ode/
├── config.py         # ALL settings: physical, simulation, training
├── main.py           # training loop, evaluation, plots, metrics.json
├── pinn_model.py     # network class + derivative helper + the two loss terms
├── physics.py        # parameter validation + analytical solution (validation only)
├── visualization.py  # training snapshots + animation of the learning process
├── tests/            # unit tests
├── requirements.txt  # torch, numpy, matplotlib
├── README.md         # this file
└── outputs/          # created when you run main.py
    ├── solution_comparison.png
    ├── training_losses.png
    ├── absolute_error.png
    ├── physics_residual.png
    ├── training_snapshots.png   # if ENABLE_TRAINING_VISUALIZATION
    ├── solution_evolution.gif   # if ENABLE_TRAINING_VISUALIZATION
    └── metrics.json
```

---

## 1. What is a second-order ODE?

An **ordinary differential equation (ODE)** relates an unknown function to its
derivatives. It is **second-order** when the highest derivative is the second
derivative. Here the unknown function is the position $x(t)$, and the equation
says that its second derivative (acceleration) must equal minus itself:

$$
x''(t) = -x(t).
$$

Solving the ODE means finding the function $x(t)$ that makes this true at every
time $t$.

## 2. What does the harmonic-oscillator equation represent?

Picture a mass on a spring with no friction. Newton's second law gives
$m\,x'' = -k\,x$ (the spring pulls back proportionally to how far it is
stretched). With $m = k = 1$ this becomes $x'' + x = 0$. The solutions are
oscillations: the mass swings back and forth forever with period $2\pi$. Our
interval $[0, 2\pi]$ covers exactly one full swing.

## 3. Why are the initial conditions necessary?

Every function $x(t) = A\cos(t) + B\sin(t)$ satisfies $x''+x=0$, for **any**
numbers $A$ and $B$ (try differentiating twice). The ODE alone has infinitely
many solutions — even $x(t)=0$ works. A second-order ODE needs **two** extra
facts to pick one solution:

* $x(0)=1$ (start position) forces $A=1$;
* $x'(0)=0$ (start velocity: released from rest) forces $B=0$.

So the unique solution is $x(t)=\cos(t)$. Without the initial-condition loss,
the network could happily learn $x(t)=0$.

## 4. What is a PINN, and how is it different from ordinary supervised learning?

| | Ordinary supervised network | PINN (this project) |
|---|---|---|
| Training data | Pairs $(t_i, x_i)$ with known answers | Only time values $t_i$ — no answers |
| Loss says… | "match these target values" | "obey the equation and the initial conditions" |
| Where the knowledge comes from | The dataset | The laws of physics (the ODE) |

A PINN is a normal neural network $\hat{x}(t)=f_\theta(t)$ (here
$\theta$ = all weights and biases). What makes it "physics-informed" is the
**loss function**: we differentiate the network's output with respect to its
input and penalise it whenever $\hat{x}'' + \hat{x}$ is not zero.

### The network (`PINN` class in `pinn_model.py`)

```
t (1 number) → Linear(1→32) → tanh → Linear(32→32) → tanh → Linear(32→32) → tanh → Linear(32→1) → x̂(t)
```

* 3 hidden layers × 32 neurons, 2,209 trainable parameters.
* **tanh** is used because it is smooth: we need second derivatives, and tanh
  has a meaningful one. (ReLU's second derivative is zero almost everywhere.)
* Weights use **Xavier (Glorot) initialization**, designed for tanh networks;
  biases start at zero.

## 5. Why collocation points without known solution values?

**Collocation points** are just times where we check the equation:
200 evenly spaced values $t_1,\dots,t_{200}$ in $[0,2\pi]$. We don't need to know
$x(t_i)$ — and pretending we didn't is the whole point, since for a real
problem you usually don't know the answer. At each point we ask: *does the
network's current function satisfy $\hat{x}''+\hat{x}=0$ here?* The amount it
fails by is the **residual**. Enough points spread across the interval force
the equation to hold everywhere in between.

## 6. How does automatic differentiation give $\hat{x}'$ and $\hat{x}''$?

A neural network is a long chain of simple operations (multiply, add, tanh).
PyTorch records every operation in a **computation graph**, and can apply the
chain rule backwards through it — that is **automatic differentiation**. It is
exact (to floating-point precision), unlike finite differences.

In `predict_with_derivatives()`:

```python
x_hat = model(t)                                                  # (N, 1)
dx_dt  = torch.autograd.grad(x_hat, t, torch.ones_like(x_hat),  create_graph=True)[0]  # (N, 1)
d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt), create_graph=True)[0]  # (N, 1)
```

* `t` is created with `requires_grad=True`, so PyTorch tracks how the output
  depends on the **input** time (normally it only tracks the weights).
* We evaluate all $N$ points at once. Output $i$ depends only on input $i$,
  so the gradient of $\sum_i \hat{x}(t_i)$ w.r.t. $t$ is exactly the list of
  derivatives $\hat{x}'(t_i)$. `torch.ones_like(x_hat)` (the `grad_outputs`
  argument) tells autograd to use that sum.
* `create_graph=True` keeps a graph **of the derivative itself**. That is what
  lets us (a) differentiate a second time to get $\hat{x}''$, and (b) later
  back-propagate a loss that contains $\hat{x}''$ into the weights. Forgetting
  it is the most common PINN bug.

### Important tensors and their shapes

| Tensor | Shape | Meaning |
|---|---|---|
| `t_collocation` | (200, 1) | collocation times, `requires_grad=True` |
| `x_hat` | (200, 1) | network output $\hat{x}(t_i)$ |
| `dx_dt` | (200, 1) | $\hat{x}'(t_i)$ from autograd |
| `d2x_dt2` | (200, 1) | $\hat{x}''(t_i)$ from autograd |
| `residual` | (200, 1) | $r(t_i)=\hat{x}''(t_i)+\hat{x}(t_i)$ |
| `t0` | (1, 1) | the single time $t=0$ |
| losses | scalar (0-D) | single numbers to minimise |
| `t_eval` | (1000, 1) | dense evaluation grid (after training only) |

The trailing dimension of 1 is there because `nn.Linear(1, 32)` expects each
sample to be a row with one feature.

## 7. How each loss term is calculated

**Physics loss** (`compute_physics_loss()`; mean squared residual over the
$N_f=200$ collocation points). In general the residual is
$r = m\hat{x}''+c\hat{x}'+k\hat{x}$ (in newtons). For the default
$m=k=1$, $c=0$ it reduces to:

$$
r(t_i)=\hat{x}''(t_i)+\hat{x}(t_i),\qquad
L_{\text{physics}}=\frac{1}{N_f}\sum_{i=1}^{N_f} r(t_i)^2
$$

**Initial-condition loss** (`compute_initial_condition_loss()`; both
conditions, evaluated at $t_0$ = `TIME_START`, velocity via autograd):

$$
L_{\text{IC}}=\big(\hat{x}(t_0)-x_0\big)^2+\big(\hat{x}'(t_0)-v_0\big)^2
$$

For the default case this is $(\hat{x}(0)-1)^2+(\hat{x}'(0)-0)^2$. The
position and velocity parts are also logged separately.

**Total loss**:

$$
L_{\text{total}}=\lambda_f\,L_{\text{physics}}+\lambda_{\text{IC}}\,L_{\text{IC}},
\qquad \lambda_f=1,\ \lambda_{\text{IC}}=10
$$

The weights $\lambda$ are **hyperparameters** (chosen by us, not learned).
$\lambda_{\text{IC}}$ is larger because the IC loss is computed at just one
point while the physics loss is averaged over 200, and because a network that
ignores the ICs collapses to the trivial solution $x=0$ (which has zero
physics loss). If training is unstable or the start of the curve is wrong, try
changing them (e.g. $\lambda_{\text{IC}}$ = 1, 10, 100) in `config.py`.

Note that the physics residual is in newtons, so its size scales with $m$ and
$k$. With large $m$ or $k$ the physics term outweighs the IC term more than in
the default case, and you may need to raise `LAMBDA_IC`.

There is deliberately **no data loss**: no term compares $\hat{x}$ with the
analytical solution.

## 8. How backpropagation and Adam update the parameters

Each epoch (`train_pinn()` in `main.py`):

1. `optimizer.zero_grad()` — clear old gradients (PyTorch adds them up otherwise).
2. Compute $L_{\text{physics}}$ and $L_{\text{IC}}$ with the **current** weights,
   then $L_{\text{total}}$.
3. `loss_total.backward()` — **backpropagation**: the chain rule through the
   whole graph (including the derivative graphs built with `create_graph=True`)
   gives $\partial L_{\text{total}}/\partial\theta$ for every weight and bias.
4. `optimizer.step()` — **Adam** updates each parameter. Plain gradient descent
   would do $\theta \leftarrow \theta - \eta\,\nabla_\theta L$. Adam additionally
   keeps a running average of past gradients (momentum, smooths the direction)
   and of their squares (adapts the step size per parameter), so it is
   less sensitive to the learning rate. Here $\eta = 0.001$.

This repeats for 10,000 epochs (configurable as `EPOCHS`). Using all 200 points
each step makes it full-batch training.

**Keeping the best weights.** Adam sometimes takes an occasional bad step that
makes the loss jump up briefly. With `KEEP_BEST_WEIGHTS = True`, the program
remembers the weights with the lowest **training** loss and restores them at
the end. This uses only the training loss, never the exact solution. Set it
to `False` to use the final-epoch weights instead.

## 9. Why is the exact solution used only for evaluation?

If $\cos(t)$ appeared in the loss, the network would just be curve-fitting a
known function — ordinary regression — and we would learn nothing about
whether physics alone can determine the solution. Keeping it out of training
makes the comparison an honest **test**: the network never saw it, so a small
error means the PINN genuinely solved the ODE. In the code,
`analytical_solution()` is called only inside `evaluate_pinn()`.

## 10. Running the project and reading the results

`python main.py` prints:

* the number of parameters and the gradient norm at epoch 1 (must be > 0 —
  proof the network receives gradients);
* losses every 500 epochs;
* the total parameter change after training (must be > 0);
* the evaluation metrics, also saved in `outputs/metrics.json` together with
  the full configuration and the best epoch.

### Evaluation metrics (not training objectives)

Computed on `N_EVALUATION_POINTS` evenly spaced points in
$[t_0, t_\text{end}]$ after training (names as in `metrics.json`):

| Metric | Meaning |
|---|---|
| `mse_position_m2` | mean of $(\hat{x}-x)^2$ [m²] |
| `rmse_position_m` | $\sqrt{\text{MSE}}$ [m] |
| `max_abs_position_error_m` | worst single-point error [m] |
| `physics_residual_mse_eval_grid_N2` | mean $r^2$ on the evaluation grid (mostly points the network did **not** train on) [N²] |
| `initial_position_error_m` | $\lvert\hat{x}(t_0)-x_0\rvert$ [m] |
| `initial_velocity_error_m_per_s` | $\lvert\hat{x}'(t_0)-v_0\rvert$ [m/s] |
| `final_model_loss_*` | training losses (weighted total, physics, IC, IC position, IC velocity) of the weights actually used |
| `last_epoch_loss_*` | training losses at the last epoch, for reference |

For the default case, good values are roughly MSE < 1e-4 and max error < 1e-2.
More collocation points do **not** automatically mean better accuracy, so
compare the metrics.

### Plots in `outputs/`

* **`solution_comparison.png`** — PINN (red dashed) on top of the analytical
  solution (black). The title lists the physical parameters. If training
  worked, the two curves are visually indistinguishable.
* **`training_losses.png`** — loss vs epoch on a log scale. You should see the
  IC loss drop very fast (the start point is easy) and the physics loss fall
  more slowly over several orders of magnitude. Small spikes are normal with
  Adam.
* **`physics_residual.png`** — residual $r(t)$ on the evaluation grid, with a
  zero reference line.
* **`absolute_error.png`** — $|\hat{x}(t)-x(t)|$, with the maximum marked. The error is usually
  smallest near $t=0$ (anchored by the ICs) and tends to grow with $t$,
  because small errors in the equation accumulate as you move away from the
  initial condition.

### If the result looks wrong

* Curve is flat near zero → the IC term is too weak; increase `LAMBDA_IC`.
* Right-hand side drifts away from the analytical curve → train longer
  (`EPOCHS`), use more collocation points, or a wider/deeper network. Long
  intervals with many oscillations are a known hard case for plain PINNs.
* Loss becomes `nan` or explodes → lower `LEARNING_RATE`.

Exact numbers vary slightly between computers and PyTorch versions, even with
the fixed seed.
