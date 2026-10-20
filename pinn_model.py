"""
pinn_model.py
=============

The neural networks and the PINN loss terms for the (Duffing) oscillator

        m x''(t) + c x'(t) + k x(t) + alpha x(t)^3 = 0,   x(t0) = x0,   x'(t0) = v0.

Nothing in this file knows the answer. The reference solutions live in
physics.py and are deliberately NOT imported here.

Loss terms:
    physics loss  -- the ODE residual at collocation points (both experiments)
    IC loss       -- the initial position and velocity (both experiments)
    data loss     -- mismatch with MEASURED positions (damping estimation only)

Tensor shape convention used everywhere: (N, 1)
    N = number of time points, 1 = one feature (time) or one output (position).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PINN(nn.Module):
    """Fully connected network  t -> x_hat(t).

    Architecture (defaults): 1 -> 32 -> 32 -> 32 -> 1, with tanh after each
    hidden layer.

    Optional input scaling: if t_min and t_max are given, the time is first
    mapped linearly from [t_min, t_max] to [-1, 1]. This is part of forward(),
    so autograd differentiates through it and x_hat'(t) is still the
    derivative with respect to the PHYSICAL time t (in seconds).

    Why tanh? The physics loss needs the SECOND derivative of the network's
    output. tanh is smooth (infinitely differentiable), so its second
    derivative is meaningful. ReLU's second derivative is zero almost
    everywhere, which would make x'' useless.
    """

    def __init__(self, hidden_size: int = 32, num_hidden_layers: int = 3,
                 t_min: float = None, t_max: float = None):
        super().__init__()
        if (t_min is None) != (t_max is None):
            raise ValueError("Give both t_min and t_max, or neither.")
        if t_min is not None and not t_max > t_min:
            raise ValueError(f"t_max must be > t_min, got t_min={t_min}, t_max={t_max}")
        self.t_min, self.t_max = t_min, t_max

        layers = [nn.Linear(1, hidden_size), nn.Tanh()]           # input layer
        for _ in range(num_hidden_layers - 1):                    # remaining hidden layers
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers.append(nn.Linear(hidden_size, 1))                  # output layer (no activation)
        self.net = nn.Sequential(*layers)

        # Xavier (Glorot) initialization is designed for tanh networks: it keeps
        # the size of the signals roughly constant from layer to layer.
        # Biases start at zero.
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    # A plain PINN uses the KNOWN damping coefficient from the configuration.
    learns_damping = False

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N, 1)  ->  x_hat: (N, 1)
        if self.t_min is not None:
            t = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0   # -> [-1, 1]
        return self.net(t)


class TrainableDampingPINN(PINN):
    """A PINN that ALSO learns the unknown damping coefficient c.

    The network part is identical to PINN. On top of it there is one extra
    trainable number, `raw_damping`, which the optimizer updates together with
    the weights and biases. The damping coefficient is

            c = softplus(raw_damping) = log(1 + exp(raw_damping))  > 0.

    Why softplus? Negative damping would mean the oscillator GAINS energy,
    which is unphysical. softplus maps any real number to a positive one, so
    the optimizer can move raw_damping freely and c still stays positive. It is
    also smooth, so gradients flow into raw_damping without problems.

    The model is created from INITIAL_DAMPING_ESTIMATE only. It never sees the
    true damping coefficient.
    """

    learns_damping = True

    def __init__(self, hidden_size: int = 32, num_hidden_layers: int = 3,
                 initial_damping: float = 1.0, t_min: float = None, t_max: float = None):
        super().__init__(hidden_size, num_hidden_layers, t_min=t_min, t_max=t_max)
        if not (math.isfinite(initial_damping) and initial_damping > 0):
            raise ValueError("INITIAL_DAMPING_ESTIMATE must be a finite number > 0 "
                             f"(softplus can only produce positive values), got {initial_damping!r}")
        # Inverse of softplus, so that softplus(raw) == initial_damping exactly:
        # raw = log(exp(c) - 1) = c + log(1 - exp(-c))  (second form avoids overflow)
        raw_initial = initial_damping + math.log(-math.expm1(-initial_damping))
        self.raw_damping = nn.Parameter(torch.tensor(raw_initial, dtype=torch.float32))

    def damping_coefficient(self) -> torch.Tensor:
        """Current estimate of c [N*s/m] as a scalar tensor (keeps the autograd link)."""
        return F.softplus(self.raw_damping)


def predict_with_derivatives(model: nn.Module, t: torch.Tensor):
    """Return x_hat(t), x_hat'(t) and x_hat''(t), all with shape (N, 1).

    `t` must have requires_grad=True so PyTorch records how the output depends
    on the input, which is what lets us differentiate with respect to time.

    How torch.autograd.grad is used here
    ------------------------------------
    The network is evaluated on N points at once. Point i's output depends
    only on point i's input, so the gradient of sum_i x_hat(t_i) with respect
    to t is exactly the vector of derivatives [x_hat'(t_1), ..., x_hat'(t_N)].
    Passing grad_outputs=ones_like(x_hat) asks autograd for precisely that sum.

    create_graph=True is essential: it keeps the computation graph of the
    derivative itself. Without it we could not (a) differentiate x_hat' again
    to get x_hat'', or (b) back-propagate the losses (which contain x_hat'
    and x_hat'') into the network weights.
    """
    x_hat = model(t)                                    # (N, 1)

    dx_dt = torch.autograd.grad(
        outputs=x_hat,
        inputs=t,
        grad_outputs=torch.ones_like(x_hat),
        create_graph=True,                              # keep graph for 2nd derivative + backprop
    )[0]                                                # (N, 1)

    d2x_dt2 = torch.autograd.grad(
        outputs=dx_dt,
        inputs=t,
        grad_outputs=torch.ones_like(dx_dt),
        create_graph=True,                              # keep graph for backprop into weights
    )[0]                                                # (N, 1)

    return x_hat, dx_dt, d2x_dt2


def compute_physics_residual(model: nn.Module, t: torch.Tensor, mass: float,
                             damping_coefficient, spring_constant: float,
                             nonlinear_coefficient: float = 0.0) -> torch.Tensor:
    """Residual r(t) = m x'' + c x' + k x + alpha x^3, shape (N, 1), units N (newtons).

    r = 0 exactly when the network satisfies the equation of motion at t.
    The derivatives come from automatic differentiation (no finite differences).
    `t` must have requires_grad=True.

    The cubic term alpha x^3 is the Duffing "nonlinear spring": its restoring
    force grows faster than linearly with stretch (alpha > 0: hardening spring).
    With alpha = 0 the residual is exactly the linear mass-spring residual.

    damping_coefficient may be a plain number (known damping) or a tensor that
    requires grad (the trainable estimate in the damping-estimation experiment);
    in the latter case the physics loss also produces gradients for c.
    """
    x_hat, dx_dt, d2x_dt2 = predict_with_derivatives(model, t)
    residual = (
        mass * d2x_dt2                              # inertia:        m x''        [N]
        + damping_coefficient * dx_dt               # damping:        c x'         [N]
        + spring_constant * x_hat                   # linear spring:  k x          [N]
        + nonlinear_coefficient * x_hat ** 3        # cubic spring:   alpha x^3    [N]
    )
    return residual


def compute_physics_loss(model: nn.Module, t_collocation: torch.Tensor, mass: float,
                         damping_coefficient, spring_constant: float,
                         nonlinear_coefficient: float = 0.0) -> torch.Tensor:
    """L_physics = (1/N_f) * sum_i r(t_i)^2   [N^2].

    t_collocation: (N_f, 1), requires_grad=True. These are just time values;
    no target positions are attached to them.
    """
    residual = compute_physics_residual(model, t_collocation, mass, damping_coefficient,
                                        spring_constant, nonlinear_coefficient)
    return torch.mean(residual ** 2)        # scalar


def compute_data_loss(model: nn.Module, t_observed: torch.Tensor,
                      x_observed: torch.Tensor) -> torch.Tensor:
    """L_data = mean (x_hat(t_j) - x_obs_j)^2 over the MEASURED positions  [m^2].

    Used ONLY in the damping-estimation experiment. The physics-only
    experiment never calls this function.
    """
    return torch.mean((model(t_observed) - x_observed) ** 2)


def compute_initial_condition_loss(model: nn.Module, time_start: float,
                                   initial_position: float, initial_velocity: float):
    """L_IC = (x_hat(t0) - x0)^2 + (x_hat'(t0) - v0)^2, evaluated at t0 = time_start.

    Returns (loss_ic, position_error, velocity_error):
        loss_ic         scalar tensor, used in the training loss
        position_error  scalar tensor  x_hat(t0)  - x0   [m]
        velocity_error  scalar tensor  x_hat'(t0) - v0   [m/s]   (via autograd)
    """
    t0 = torch.full((1, 1), float(time_start), requires_grad=True)   # shape (1, 1)
    x0_hat, v0_hat, _ = predict_with_derivatives(model, t0)
    position_error = (x0_hat - initial_position).squeeze()
    velocity_error = (v0_hat - initial_velocity).squeeze()
    loss_ic = position_error ** 2 + velocity_error ** 2
    return loss_ic, position_error, velocity_error
