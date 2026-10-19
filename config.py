"""
config.py
=========

Every setting you might want to change lives here, in four groups:

    1. Physical parameters     -- the mass-spring-damper system itself
    2. Simulation settings     -- the time interval and how many points are used
    3. Training settings       -- the network and the optimiser
    4. Training visualization  -- snapshots / animation of the learning process

The governing equation (SI units throughout) is

        m x''(t) + c x'(t) + k x(t) = 0,
        x(TIME_START) = INITIAL_POSITION,   x'(TIME_START) = INITIAL_VELOCITY.

Damping regimes (c_crit = 2 * sqrt(m * k) is the "critical damping"):
        c = 0                undamped          (oscillates forever)
        0 < c < c_crit       underdamped       (oscillates while decaying)
        c = c_crit           critically damped (fastest return without oscillating)
        c > c_crit           overdamped        (slow return without oscillating)
With MASS = SPRING_CONSTANT = 1, c_crit = 2.0.
"""

import math
import os

# ----------------------------------------------------------------------------
# 1. Physical parameters (SI units)
# ----------------------------------------------------------------------------
MASS = 2.0                    # m  [kg],      must be > 0
SPRING_CONSTANT = 5.0         # k  [N/m],     must be > 0
DAMPING_COEFFICIENT = 0.0     # c  [N·s/m],   must be >= 0  (0 = undamped)
INITIAL_POSITION = 1.0        # x0 = x(TIME_START)   [m]
INITIAL_VELOCITY = 0.0        # v0 = x'(TIME_START)  [m/s]

# ----------------------------------------------------------------------------
# 2. Simulation settings
# ----------------------------------------------------------------------------
TIME_START = 0.0              # t0 [s]; the initial conditions are applied here
TIME_END = 2 * math.pi        # [s]; try 10 * math.pi or 20 * math.pi for longer runs
N_COLLOCATION = 200           # points where the ODE is enforced during training (try 50, 200, 500, 1000)
N_EVALUATION_POINTS = 1000    # dense grid used ONLY for evaluation and plots

# ----------------------------------------------------------------------------
# 3. Training settings
# ----------------------------------------------------------------------------
SEED = 42
HIDDEN_SIZE = 32              # neurons per hidden layer
NUM_HIDDEN_LAYERS = 3
EPOCHS = 3000
LEARNING_RATE = 1e-3
LAMBDA_PHYSICS = 1.0          # weight of the physics loss
LAMBDA_IC = 10.0              # weight of the initial-condition loss
PRINT_EVERY = 500
KEEP_BEST_WEIGHTS = True      # at the end, restore the weights with the lowest TRAINING loss

# If True, the network internally rescales time from [TIME_START, TIME_END] to
# [-1, 1] before the first layer. This keeps tanh inputs in a sensible range
# for long intervals or a nonzero TIME_START. Derivatives are still taken with
# respect to the real (physical) time, because the rescaling is part of the
# network that autograd differentiates through.
# Default False = the original network. In tests it did not fix long intervals
# (10*pi failed either way) and was slightly less accurate on [0, 2*pi].
NORMALIZE_TIME_INPUT = False

# ----------------------------------------------------------------------------
# 4. Training visualization (how the predicted curve evolves during training)
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
# Output
# ----------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
