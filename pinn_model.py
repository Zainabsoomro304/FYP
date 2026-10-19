"""
pinn_model.py
=============

The neural network and the two PINN loss terms for the damped mass-spring system

        m x''(t) + c x'(t) + k x(t) = 0,    x(t0) = x0,   x'(t0) = v0.

Nothing in this file knows the exact answer. The network learns only from the
ODE (physics loss) and the two initial conditions (IC loss). The analytical
solution lives in physics.py and is deliberately NOT imported here.

Tensor shape convention used everywhere: (N, 1)
    N = number of time points, 1 = one feature (time) or one output (position).
"""

import torch
import torch.nn as nn


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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N, 1)  ->  x_hat: (N, 1)
        if self.t_min is not None:
            t = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0   # -> [-1, 1]
        return self.net(t)


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
                             damping_coefficient: float, spring_constant: float) -> torch.Tensor:
    """Residual r(t) = m x_hat'' + c x_hat' + k x_hat, shape (N, 1), units N (newtons).

    r = 0 exactly when the network satisfies the equation of motion at t.
    The derivatives come from automatic differentiation (no finite differences).
    `t` must have requires_grad=True.
    """
    x_hat, dx_dt, d2x_dt2 = predict_with_derivatives(model, t)
    residual = (
        mass * d2x_dt2                      # inertia:      m x''  [N]
        + damping_coefficient * dx_dt       # damping:      c x'   [N]
        + spring_constant * x_hat           # spring force: k x    [N]
    )
    return residual


def compute_physics_loss(model: nn.Module, t_collocation: torch.Tensor, mass: float,
                         damping_coefficient: float, spring_constant: float) -> torch.Tensor:
    """L_physics = (1/N_f) * sum_i r(t_i)^2   [N^2].

    t_collocation: (N_f, 1), requires_grad=True. These are just time values;
    no target positions are attached to them.
    """
    residual = compute_physics_residual(model, t_collocation, mass,
                                        damping_coefficient, spring_constant)
    return torch.mean(residual ** 2)        # scalar


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
