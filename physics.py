"""
physics.py
==========

Parameter checks and REFERENCE solutions for the (Duffing) oscillator

        m x''(t) + c x'(t) + k x(t) + alpha x(t)^3 = 0,   x(t0) = x0,   x'(t0) = v0.

* solve_reference_ode(): a NUMERICAL solution from SciPy, valid for any alpha.
  It is a high-accuracy approximation, not a closed-form formula.
* analytical_solution(): the EXACT closed-form solution, which only exists for
  the linear case alpha = 0. Used to check the numerical solver.

Both are used ONLY for validation (metrics and plots) and for generating
synthetic measurements in the damping-estimation experiment. They are never
imported by pinn_model.py, and the physics-only PINN never trains on them.

All functions work on NumPy arrays and use SI units.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

# Relative tolerance used to decide that c equals the critical damping
# 2*sqrt(m*k). An exact floating-point comparison would almost never be true
# for values like c = 2*sqrt(2.0).
CRITICAL_DAMPING_RTOL = 1e-9


def validate_physical_parameters(mass, spring_constant, damping_coefficient):
    """Raise ValueError if the parameters do not describe a valid mass-spring system."""
    for name, value in [("MASS", mass), ("SPRING_CONSTANT", spring_constant),
                        ("DAMPING_COEFFICIENT", damping_coefficient)]:
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if mass <= 0:
        raise ValueError(f"MASS must be > 0 kg, got {mass}")
    if spring_constant <= 0:
        raise ValueError(f"SPRING_CONSTANT must be > 0 N/m, got {spring_constant}")
    if damping_coefficient < 0:
        raise ValueError(f"DAMPING_COEFFICIENT must be >= 0 N*s/m, got {damping_coefficient}")


def critical_damping(mass, spring_constant):
    """c_crit = 2*sqrt(m*k)  [N·s/m]."""
    return 2.0 * math.sqrt(mass * spring_constant)


def classify_damping_regime(mass, spring_constant, damping_coefficient):
    """Return 'undamped', 'underdamped', 'critically damped' or 'overdamped'."""
    validate_physical_parameters(mass, spring_constant, damping_coefficient)
    c_crit = critical_damping(mass, spring_constant)
    if damping_coefficient == 0.0:
        return "undamped"
    if math.isclose(damping_coefficient, c_crit, rel_tol=CRITICAL_DAMPING_RTOL):
        return "critically damped"
    if damping_coefficient < c_crit:
        return "underdamped"
    return "overdamped"


def _analytical_position_and_velocity(t, mass, spring_constant, damping_coefficient,
                                      initial_position, initial_velocity, time_start):
    """Exact x(t) and x'(t). Uses tau = t - t0 so the ICs hold at t0 = time_start."""
    regime = classify_damping_regime(mass, spring_constant, damping_coefficient)
    for name, value in [("INITIAL_POSITION", initial_position),
                        ("INITIAL_VELOCITY", initial_velocity), ("TIME_START", time_start)]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    tau = np.asarray(t, dtype=np.float64) - time_start      # time since the start [s]
    x0, v0 = float(initial_position), float(initial_velocity)

    omega_n = math.sqrt(spring_constant / mass)              # natural frequency [rad/s]
    zeta = damping_coefficient / critical_damping(mass, spring_constant)  # damping ratio [-]

    if regime == "undamped":
        # x = x0 cos(w_n tau) + (v0 / w_n) sin(w_n tau)
        cos, sin = np.cos(omega_n * tau), np.sin(omega_n * tau)
        x = x0 * cos + (v0 / omega_n) * sin
        v = -x0 * omega_n * sin + v0 * cos

    elif regime == "underdamped":
        # x = exp(-zeta w_n tau) [A cos(w_d tau) + B sin(w_d tau)],  w_d = w_n sqrt(1 - zeta^2)
        omega_d = omega_n * math.sqrt(1.0 - zeta ** 2)       # damped frequency [rad/s]
        sigma = zeta * omega_n                               # decay rate [1/s]
        a = x0
        b = (v0 + sigma * x0) / omega_d
        envelope = np.exp(-sigma * tau)
        cos, sin = np.cos(omega_d * tau), np.sin(omega_d * tau)
        x = envelope * (a * cos + b * sin)
        v = envelope * ((b * omega_d - sigma * a) * cos - (a * omega_d + sigma * b) * sin)

    elif regime == "critically damped":
        # x = (A + B tau) exp(-w_n tau)   (double root r = -w_n)
        a = x0
        b = v0 + omega_n * x0
        envelope = np.exp(-omega_n * tau)
        x = (a + b * tau) * envelope
        v = (b - omega_n * (a + b * tau)) * envelope

    else:  # overdamped
        # x = A exp(r1 tau) + B exp(r2 tau), with two distinct real roots
        # r1,2 = -zeta w_n +/- w_n sqrt(zeta^2 - 1)  (both negative)
        root = omega_n * math.sqrt(zeta ** 2 - 1.0)
        r1, r2 = -zeta * omega_n + root, -zeta * omega_n - root
        a = (v0 - r2 * x0) / (r1 - r2)                       # from A + B = x0, r1 A + r2 B = v0
        b = (r1 * x0 - v0) / (r1 - r2)
        e1, e2 = np.exp(r1 * tau), np.exp(r2 * tau)
        x = a * e1 + b * e2
        v = a * r1 * e1 + b * r2 * e2

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
        raise ArithmeticError(
            f"Analytical solution is not finite for this configuration ({regime}, "
            f"m={mass}, k={spring_constant}, c={damping_coefficient}). This usually means "
            "the time interval is so long that exp(...) overflows (e.g. when evaluating "
            "far BEFORE TIME_START).")
    return x, v


def analytical_solution(t, mass, spring_constant, damping_coefficient,
                        initial_position, initial_velocity, time_start):
    """Exact position x(t) [m] for every regime. FOR VALIDATION ONLY."""
    x, _ = _analytical_position_and_velocity(t, mass, spring_constant, damping_coefficient,
                                             initial_position, initial_velocity, time_start)
    return x


def analytical_velocity(t, mass, spring_constant, damping_coefficient,
                        initial_position, initial_velocity, time_start):
    """Exact velocity x'(t) [m/s] for every regime. FOR VALIDATION ONLY."""
    _, v = _analytical_position_and_velocity(t, mass, spring_constant, damping_coefficient,
                                             initial_position, initial_velocity, time_start)
    return v


# ----------------------------------------------------------------------------
# Numerical reference solution (works for the nonlinear Duffing equation)
# ----------------------------------------------------------------------------
def solve_reference_ode(t_eval, mass, spring_constant, damping_coefficient, nonlinear_coefficient,
                        initial_position, initial_velocity, time_start,
                        method="DOP853", rtol=1e-10, atol=1e-12):
    """Numerical solution of m x'' + c x' + k x + alpha x^3 = 0. FOR VALIDATION ONLY.

    How it works: SciPy's solve_ivp integrates FIRST-order systems, so the
    second-order ODE is rewritten with the velocity v = x' as a second unknown:

            x' = v
            v' = -(c v + k x + alpha x^3) / m

    Starting from (x0, v0) at t0 = time_start, the solver takes many small,
    adaptively chosen time steps. Its local error is kept below
    atol + rtol * |value|. The result is a very accurate APPROXIMATION, not an
    exact formula (no closed-form solution exists when alpha != 0).

    t_eval: increasing times >= time_start at which to report the solution.
    Returns (positions [m], velocities [m/s]) as NumPy arrays.
    Raises ValueError for invalid input and RuntimeError if the solver fails.
    """
    validate_physical_parameters(mass, spring_constant, damping_coefficient)
    for name, value in [("NONLINEAR_COEFFICIENT", nonlinear_coefficient),
                        ("INITIAL_POSITION", initial_position),
                        ("INITIAL_VELOCITY", initial_velocity), ("TIME_START", time_start)]:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    t_eval = np.asarray(t_eval, dtype=np.float64).ravel()
    if t_eval.size == 0:
        raise ValueError("t_eval must contain at least one time")
    if np.any(np.diff(t_eval) < 0):
        raise ValueError("t_eval must be sorted in increasing order")
    if t_eval[0] < time_start:
        raise ValueError(f"t_eval starts at {t_eval[0]}, before TIME_START = {time_start}; "
                         "the solver only integrates forward from the initial condition")

    m, c, k, alpha = float(mass), float(damping_coefficient), float(spring_constant), \
        float(nonlinear_coefficient)

    def right_hand_side(t, state):
        x, v = state
        # Newton's law m a = -(damping force + linear spring force + cubic spring force).
        acceleration = -(c * v + k * x + alpha * x ** 3) / m
        return [v, acceleration]

    t_final = float(t_eval[-1])
    if t_final == time_start:                    # nothing to integrate
        return (np.full(t_eval.shape, float(initial_position)),
                np.full(t_eval.shape, float(initial_velocity)))

    solution = solve_ivp(right_hand_side, (time_start, t_final),
                         [float(initial_position), float(initial_velocity)],
                         t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError(f"Reference ODE solver ({method}) failed: {solution.message} "
                           f"(m={m}, k={k}, c={c}, alpha={alpha}, x0={initial_position}, "
                           f"v0={initial_velocity}). A negative NONLINEAR_COEFFICIENT with a large "
                           "amplitude can make the solution blow up in finite time.")
    x, v = solution.y
    if x.shape != t_eval.shape or not (np.all(np.isfinite(x)) and np.all(np.isfinite(v))):
        raise RuntimeError("Reference ODE solver returned missing or non-finite values")
    return x, v


def make_reference_solver(cfg, damping_coefficient):
    """Return a function t -> reference positions x(t), using the settings in cfg.

    The damping coefficient is passed explicitly so the caller decides which
    value is meant (the known c in Mode A, the hidden TRUE c in Mode B).
    """
    def reference_position(t):
        x, _ = solve_reference_ode(t, cfg.MASS, cfg.SPRING_CONSTANT, damping_coefficient,
                                   cfg.NONLINEAR_COEFFICIENT, cfg.INITIAL_POSITION,
                                   cfg.INITIAL_VELOCITY, cfg.TIME_START,
                                   method=cfg.REFERENCE_SOLVER_METHOD,
                                   rtol=cfg.REFERENCE_RTOL, atol=cfg.REFERENCE_ATOL)
        return x
    return reference_position
