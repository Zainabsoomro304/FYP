# Physics-Informed Neural Networks (PINNs) for a Nonlinear Oscillator

This project trains a small neural network to solve the **Duffing oscillator**

$$
m\,x''(t) + c\,x'(t) + k\,x(t) + \alpha\,x(t)^3 = 0,\qquad x(t_0)=x_0,\qquad x'(t_0)=v_0,\qquad t\in[t_0,t_{\text{end}}]
$$

and, in a second experiment, uses a PINN to **estimate an unknown damping
coefficient** $c$ from measured positions.

The cubic term $\alpha x^3$ is a nonlinear spring. With $\alpha>0$ the
spring gets stiffer the more it is stretched, which is why there is no
closed-form solution any more. With `NONLINEAR_COEFFICIENT = 0` the equation
is the linear mass–spring(–damper) equation from earlier versions of this
project, and everything still works for it.

There are two experiments:

| | **Mode A: `physics_only`** | **Mode B: `damping_estimation`** |
|---|---|---|
| Knows | all parameters, incl. $c$ | $m$, $k$, $\alpha$, $x_0$, $v_0$; **not** $c$ |
| Training loss | $\lambda_f L_\text{physics} + \lambda_{IC} L_{IC}$ | $\lambda_f L_\text{physics}(c_\text{est}) + \lambda_{IC} L_{IC} + \lambda_d L_\text{data}$ |
| Learns | network weights | network weights **and** $c_\text{est}$ |
| Uses measurements? | **never** | `N_OBSERVATION_POINTS` positions |
| Compared afterwards with | SciPy numerical reference | SciPy reference + hidden true $c$ |
| Output folder | `outputs/physics_only/` | `outputs/damping_estimation/` |

---

## Quick start

```bash
pip install -r requirements.txt
python main.py                              # runs EXPERIMENT_MODE from config.py
python main.py --mode physics_only          # Mode A
python main.py --mode damping_estimation    # Mode B
python main.py --mode physics_only --epochs 2000   # quick, less accurate run
```

Runs on a normal CPU (no GPU needed). With the default 10,000 epochs each
mode takes about 2 minutes.

Unit tests (about 1.5 minutes; only short smoke-training):

```bash
python -m unittest discover -s tests -v
```

The full damping-recovery test (10,000 epochs) is skipped unless you set
`RUN_SLOW_TESTS=1` (PowerShell: `$env:RUN_SLOW_TESTS = "1"`).

## What to edit (all in `config.py`)

| To change… | Edit |
|---|---|
| Which experiment runs | `EXPERIMENT_MODE` (`"physics_only"` or `"damping_estimation"`) or `--mode` |
| Mass [kg], spring constant [N/m] | `MASS`, `SPRING_CONSTANT` (both > 0) |
| **Nonlinear (cubic) coefficient** [N/m³] | `NONLINEAR_COEFFICIENT` (0 = linear spring) |
| Damping in Mode A [N·s/m] | `DAMPING_COEFFICIENT` (>= 0) |
| **True damping for Mode B's synthetic data** | `TRUE_DAMPING_COEFFICIENT` |
| **Starting guess for the estimate** | `INITIAL_DAMPING_ESTIMATE` (> 0) |
| **Number of measurements** | `N_OBSERVATION_POINTS` |
| **Measurement noise** | `ADD_OBSERVATION_NOISE`, `OBSERVATION_NOISE_STD` [m] (seeded with `SEED`) |
| **Loss weights** | `LAMBDA_PHYSICS`, `LAMBDA_IC`, `LAMBDA_DATA` (Mode B only) |
| Initial position [m] / velocity [m/s] | `INITIAL_POSITION`, `INITIAL_VELOCITY` |
| Simulation interval [s] | `TIME_START`, `TIME_END` |
| Collocation / evaluation points | `N_COLLOCATION`, `N_EVALUATION_POINTS` |
| Reference solver accuracy | `REFERENCE_SOLVER_METHOD`, `REFERENCE_RTOL`, `REFERENCE_ATOL` |
| Network & optimiser | `HIDDEN_SIZE`, `NUM_HIDDEN_LAYERS`, `EPOCHS`, `LEARNING_RATE`, `SEED` |
| Best-weights restore | `KEEP_BEST_WEIGHTS` |
| Time-input scaling | `NORMALIZE_TIME_INPUT` (see below) |
| Training animation / snapshots | `ENABLE_TRAINING_VISUALIZATION`, `VISUALIZATION_INTERVAL`, `VISUALIZATION_POINTS`, `ANIMATION_INTERVAL_MS`, `ANIMATION_FORMAT` |

The loss weights are starting choices for experiments, not universal
constants. A larger `LAMBDA_DATA` makes the curve follow the measurements more
closely, which also makes it follow their noise. A larger `LAMBDA_PHYSICS`
trusts the equation more.

To run other configurations from a script without editing `config.py`:

```python
import main
main.main({"EXPERIMENT_MODE": "damping_estimation", "TRUE_DAMPING_COEFFICIENT": 0.5,
           "OUTPUT_DIR": "outputs_c05"})
```

Damping regimes of the linear part, with $c_\text{crit}=2\sqrt{mk}$ (= 2.0 when
$m=k=1$): undamped $c=0$, underdamped $0<c<c_\text{crit}$, critically damped
$c=c_\text{crit}$, overdamped $c>c_\text{crit}$. With $\alpha\neq0$ these
labels only describe the small-amplitude behaviour.

**`NORMALIZE_TIME_INPUT`** (default `False` = the original network): when
`True`, the network maps time from $[t_0, t_\text{end}]$ to $[-1,1]$ as its
first operation. Autograd still returns derivatives with respect to the
physical time. In earlier tests it did not fix long intervals and was slightly
less accurate on $[0,2\pi]$, so it is off by default.

## The numerical reference solution (validation only)

There is no closed-form solution for $\alpha\neq0$, so `solve_reference_ode()`
in `physics.py` computes one with `scipy.integrate.solve_ivp`. The
second-order ODE is rewritten as a first-order system in $(x, v)$:

$$
x' = v,\qquad v' = -\frac{c\,v + k\,x + \alpha\,x^3}{m}
$$

The default method is DOP853 (8th-order Runge–Kutta) with `rtol = 1e-10` and
`atol = 1e-12`. This is a very accurate **numerical approximation**, not an
exact formula. If the solver fails (for example a negative $\alpha$ with a
large amplitude, which blows up in finite time), it raises `RuntimeError` with
SciPy's message instead of returning bad numbers.

For $\alpha=0$ the closed-form `analytical_solution()` is still available. Mode A
then also reports `reference_vs_analytical_max_abs_diff_m`, which was
1.9e-10 m in testing.

## The unknown-damping experiment (Mode B) in detail

1. **Synthetic measurements.** The reference solver is run with
   `TRUE_DAMPING_COEFFICIENT` at `N_OBSERVATION_POINTS` evenly spaced times.
   Optional Gaussian noise (seeded with `SEED`) is added. The data are saved
   to `observations.csv` (time, observed position, noise-free position).
2. **What the model gets.** A copy of the configuration with
   `TRUE_DAMPING_COEFFICIENT` and `DAMPING_COEFFICIENT` **removed**
   (`make_known_parameter_config()`), the observation times and positions, and
   the known parameters and initial conditions. Any accidental use of the true
   value inside training would therefore crash, and a unit test checks this.
3. **Trainable damping.** `TrainableDampingPINN` (in `pinn_model.py`) adds one
   parameter `raw_damping` with $c_\text{est}=\text{softplus}(\text{raw})>0$.
   Negative damping would mean the system gains energy, and softplus keeps the
   estimate positive while staying smooth for gradients. The model starts at
   `INITIAL_DAMPING_ESTIMATE`, and Adam updates `raw_damping` together with the
   weights.
4. **How the losses work together.** $L_\text{physics}$ (with $c_\text{est}$)
   asks for a curve that obeys the equation, $L_{IC}$ fixes its start, and
   $L_\text{data}$ asks it to pass through the measurements. The only way to
   satisfy all three is for $c_\text{est}$ to approach the damping that
   produced the data.
5. **Evaluation (after training only).** The estimate is compared with the
   hidden true value (absolute and relative error), and the trained curve with
   the reference solution and the measurements.

A small training loss alone does not prove $c$ was identified. The report
always shows the coefficient error and the position errors.

## Results (actually run, 10,000 epochs, default settings, CPU)

| Run | Key results |
|---|---|
| Mode A, Duffing ($\alpha=0.2$, $c=0$, $[0,10]$ s) | RMSE 3.1e-3 m, max error 6.8e-3 m, IC errors < 4e-5 |
| Mode A, Duffing, only 3,000 epochs | RMSE 0.44 m: **not converged** (why the default is 10,000) |
| Mode A, linear check ($\alpha=0$, $[0,2\pi]$) | RMSE 8.46e-5 m: same as the original project |
| Mode B, clean data (true $c=0.3$, start 0.8) | **$c_\text{est}=0.30048$**, rel. error 0.16%; RMSE vs reference 3.5e-4 m; data MSE 1.2e-7 m² |
| Mode B, noisy data (std 0.01 m) | **$c_\text{est}=0.30119$**, rel. error 0.40%; RMSE vs reference 2.0e-3 m; data MSE 5.2e-5 m² |

These come from a single seed and one set of settings. Different settings
(fewer or badly placed observations, much more noise, a longer interval, other
loss weights) can give worse estimates. Always check
`damping_relative_error` and `rmse_position_m` in `metrics.json`.

## Output files

```text
outputs/physics_only/                outputs/damping_estimation/
├── duffing_solution_comparison.png  ├── duffing_solution_comparison.png
├── duffing_absolute_error.png       ├── duffing_absolute_error.png
├── duffing_training_losses.png      ├── duffing_training_losses.png   (incl. data loss)
├── physics_residual.png             ├── physics_residual.png
├── training_snapshots.png  (*)      ├── damping_estimation_history.png
├── solution_evolution.gif  (*)      ├── damping_observations.png
└── metrics.json                     ├── observations.csv
                                     ├── training_snapshots.png  (*)
                                     ├── solution_evolution.gif  (*)
                                     └── metrics.json
(*) only if ENABLE_TRAINING_VISUALIZATION = True
```

## Training visualization: watching the PINN learn

With `ENABLE_TRAINING_VISUALIZATION = True`, each experiment also saves
`training_snapshots.png` and `solution_evolution.gif` (or `.mp4` if ffmpeg is
installed and `ANIMATION_FORMAT = "mp4"`; otherwise it falls back to GIF).
They show the real network's prediction at epochs 0, 1, 10, 100, 500, 1000,
2500, 5000, 7500 (those ≤ `EPOCHS`), every `VISUALIZATION_INTERVAL` epochs,
the final epoch, and the restored best weights (green title), against the
fixed reference curve. `ANIMATION_INTERVAL_MS` only sets playback speed.

"Epoch N" in the snapshots = weights after N optimizer updates (epoch 0 = the
untrained network). The training log counts loop iterations instead, so best
weights logged as "epoch 9915" appear as snapshot epoch 9914. Snapshots store
only prediction arrays, captured with `model.eval()` + `torch.no_grad()`, and
never change training.

## Project structure

```text
pinn_ode/
├── config.py              # ALL settings (grouped by purpose)
├── main.py                # entry point: chooses the mode; Mode A (physics_only)
├── damping_estimation.py  # Mode B: synthetic data, estimation run, its plots
├── training.py            # shared: config loading, losses, training loop, evaluation
├── pinn_model.py          # PINN, TrainableDampingPINN, derivatives, loss terms
├── physics.py             # parameter checks, SciPy reference solver, linear exact solution
├── plots.py               # shared plots + metrics.json helpers
├── visualization.py       # training snapshots + animation
├── tests/                 # test_pinn_ode.py (linear/visualization), test_duffing.py
├── requirements.txt       # torch, numpy, matplotlib, scipy
└── outputs/               # results, one sub-folder per experiment
```

Files directly inside `outputs/` (not in a sub-folder) come from runs of the
older single-experiment version and can be deleted.

---

> **Sections 1–10 below** explain the PINN method step by step using the
> simplest case, the linear oscillator $x''+x=0$ (= `NONLINEAR_COEFFICIENT = 0`,
> `DAMPING_COEFFICIENT = 0`, `TIME_END = 2π`). Everything carries over: the
> Duffing residual just adds $+\alpha\hat{x}^3$, and Mode B adds the data loss.

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

Each epoch (`train_pinn()` in `training.py`):

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
reference solutions are computed only after training (`run_physics_only()` in `main.py`).

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

### Plots in `outputs/physics_only/`

* **`duffing_solution_comparison.png`** — PINN (red dashed) on top of the reference
  solution (black). The title lists the physical parameters. If training
  worked, the two curves are visually indistinguishable.
* **`duffing_training_losses.png`** — loss vs epoch on a log scale. You should see the
  IC loss drop very fast (the start point is easy) and the physics loss fall
  more slowly over several orders of magnitude. Small spikes are normal with
  Adam.
* **`physics_residual.png`** — residual $r(t)$ on the evaluation grid, with a
  zero reference line.
* **`duffing_absolute_error.png`** — $|\hat{x}(t)-x(t)|$, with the maximum marked. The error is usually
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
