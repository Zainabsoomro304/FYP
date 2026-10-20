"""
config.py
=========

Every setting you might want to change lives here, in these groups:

    0. Experiment selection    -- which experiment `python main.py` runs
    1. Physical parameters     -- the oscillator itself
    2. Simulation settings     -- the time interval and how many points are used
    3. Reference solver        -- SciPy settings for the numerical reference solution
    4. Training settings       -- the network, the optimiser and the loss weights
    5. Training visualization  -- snapshots / animation of the learning process
    6. Damping estimation      -- the unknown-damping experiment (Mode B only)

The governing equation (SI units throughout) is the Duffing oscillator

        m x''(t) + c x'(t) + k x(t) + alpha x(t)^3 = 0,
        x(TIME_START) = INITIAL_POSITION,   x'(TIME_START) = INITIAL_VELOCITY.

With NONLINEAR_COEFFICIENT = 0 it is the linear mass-spring(-damper) equation.

Damping regimes of the LINEAR part (c_crit = 2 * sqrt(m * k) is "critical damping"):
        c = 0                undamped          (oscillates forever)
        0 < c < c_crit       underdamped       (oscillates while decaying)
        c = c_crit           critically damped (fastest return without oscillating)
        c > c_crit           overdamped        (slow return without oscillating)
With MASS = SPRING_CONSTANT = 1, c_crit = 2.0. With alpha != 0 these labels
are only a guide (they describe the small-amplitude behaviour).
"""

import math  # noqa: F401  (lets you write e.g. TIME_END = 10 * math.pi)
import os

# ----------------------------------------------------------------------------
# 0. Experiment selection (can also be set with:  python main.py --mode ...)
# ----------------------------------------------------------------------------
#   "physics_only"        Mode A: all parameters known; trains on physics + ICs only
#   "damping_estimation"  Mode B: c unknown; learns c from position measurements
EXPERIMENT_MODE = "physics_only"

# ----------------------------------------------------------------------------
# 1. Physical parameters (SI units)
# ----------------------------------------------------------------------------
MASS = 2.0                    # m      [kg],      must be > 0
SPRING_CONSTANT = 3.0         # k      [N/m],     must be > 0
DAMPING_COEFFICIENT = 2.5     # c      [N·s/m],   must be >= 0. Mode A only; Mode B
                              #                   uses TRUE_DAMPING_COEFFICIENT (section 6)
NONLINEAR_COEFFICIENT = 0.6   # alpha  [N/m³],    cubic stiffness (0 = linear spring)
INITIAL_POSITION = 1.0        # x0 = x(TIME_START)   [m]
INITIAL_VELOCITY = 0.0        # v0 = x'(TIME_START)  [m/s]

# ----------------------------------------------------------------------------
# 2. Simulation settings
# ----------------------------------------------------------------------------
TIME_START = 0.0              # t0 [s]; the initial conditions are applied here
TIME_END = 10.0               # [s]
N_COLLOCATION = 200           # points where the ODE is enforced during training (try 50, 200, 500, 1000)
N_EVALUATION_POINTS = 1000    # dense grid used ONLY for evaluation and plots

# ----------------------------------------------------------------------------
# 3. Reference solver (scipy.integrate.solve_ivp) -- validation / synthetic data only
# ----------------------------------------------------------------------------
REFERENCE_SOLVER_METHOD = "DOP853"   # high-order Runge-Kutta method
REFERENCE_RTOL = 1e-10               # relative tolerance
REFERENCE_ATOL = 1e-12               # absolute tolerance [m, m/s]

# ----------------------------------------------------------------------------
# 4. Training settings
# ----------------------------------------------------------------------------
SEED = 42                     # network initialisation AND observation noise
HIDDEN_SIZE = 32              # neurons per hidden layer
NUM_HIDDEN_LAYERS = 3
EPOCHS = 10_000               # 3000 was not enough for the Duffing default on [0, 10] s
LEARNING_RATE = 1e-3
PRINT_EVERY = 500
KEEP_BEST_WEIGHTS = True      # at the end, restore the weights with the lowest TRAINING loss

# Loss weights. They are starting choices for experiments, not universal
# constants: they set how strongly each loss term pulls on the network.
LAMBDA_PHYSICS = 1.0          # weight of the physics (ODE residual) loss
LAMBDA_IC = 10.0              # weight of the initial-condition loss
LAMBDA_DATA = 10.0            # weight of the measured-position loss (Mode B only)

# If True, the network internally rescales time from [TIME_START, TIME_END] to
# [-1, 1] before the first layer. This keeps tanh inputs in a sensible range
# for long intervals or a nonzero TIME_START. Derivatives are still taken with
# respect to the real (physical) time, because the rescaling is part of the
# network that autograd differentiates through.
# Default False = the original network. In tests it did not fix long intervals
# (10*pi failed either way) and was slightly less accurate on [0, 2*pi].
NORMALIZE_TIME_INPUT = False

# ----------------------------------------------------------------------------
# 5. Training visualization (how the predicted curve evolves during training)
# ----------------------------------------------------------------------------
# "Epoch N" in the snapshots means: the weights after N optimizer updates
# (epoch 0 = the randomly initialised network, before any update).
# Snapshots are taken at epochs 0, 1, 10, 100, 500, 1000, 2500, 5000, 7500
# (those that are <= EPOCHS), at every multiple of VISUALIZATION_INTERVAL, and
# at the final epoch. Only prediction arrays are stored, never model weights.
ENABLE_TRAINING_VISUALIZATION = True
VISUALIZATION_INTERVAL = 100  # take a snapshot every this many epochs
VISUALIZATION_POINTS = 500    # time points per snapshot curve
ANIMATION_INTERVAL_MS = 100   # delay between animation frames (playback speed only)
ANIMATION_FORMAT = "gif"      # "gif" or "mp4" (mp4 needs ffmpeg; falls back to gif)

# ----------------------------------------------------------------------------
# 6. Damping estimation (Mode B: EXPERIMENT_MODE = "damping_estimation")
# ----------------------------------------------------------------------------
# The equation's form, MASS, SPRING_CONSTANT, NONLINEAR_COEFFICIENT and the
# initial conditions are known; the damping coefficient is not.
#
# TRUE_DAMPING_COEFFICIENT is HIDDEN from the estimation model: it is used only
# to generate the synthetic measurements and, after training, to judge how
# good the estimate is. It never enters the model or its loss.
TRUE_DAMPING_COEFFICIENT = 0.3     # [N·s/m]
INITIAL_DAMPING_ESTIMATE = 0.8     # [N·s/m] starting guess, must be > 0

N_OBSERVATION_POINTS = 50          # measured positions, evenly spaced over the interval
ADD_OBSERVATION_NOISE = False      # add Gaussian measurement noise (seeded with SEED)
OBSERVATION_NOISE_STD = 0.01       # [m] standard deviation of that noise

# ----------------------------------------------------------------------------
# Output (each experiment writes to its own sub-folder, e.g. outputs/physics_only)
# ----------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
