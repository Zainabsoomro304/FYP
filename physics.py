"""
physics.py
==========

Parameter checks and the ANALYTICAL (exact) solution of the damped
mass-spring system

        m x''(t) + c x'(t) + k x(t) = 0,     x(t0) = x0,   x'(t0) = v0.

The analytical solution is used ONLY for validation (metrics and plots).
It is never imported by pinn_model.py and never appears in the training loss.

All functions work on NumPy arrays and use SI units.
"""

import math

import numpy as np

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
