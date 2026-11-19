"""AdaBelief optimizer.

Implementation of "AdaBelief Optimizer: Adapting Stepsizes by the Belief in
Observed Gradients" (Zhuang et al., NeurIPS 2020), re-written for the RVC
training pipeline. Instead of tracking the second moment of the raw gradient
(Adam), AdaBelief tracks the variance of the *prediction residual*
``g - m``, which adapts the step size to how much the gradient stream
deviates from its own momentum — giving faster convergence on noisy
spectral losses and fewer oscillations near minima.

Reference: https://arxiv.org/abs/2010.07468
"""

import math

import torch
from torch.optim.optimizer import Optimizer


class AdaBelief(Optimizer):
    """AdaBelief optimizer with decoupled weight decay and optional rectification.

    Args:
        params: iterable of parameters to optimize or dicts of parameter groups.
        lr: learning rate (default ``1e-3``).
        betas: coefficients for the moment estimates (default ``(0.9, 0.999)``).
        eps: term added to the denominator for numerical stability. Keep it
            small (``1e-16`` by default) — a large epsilon disables the
            "belief" adaptation entirely.
        weight_decay: decoupled weight decay (AdamW style) applied per step.
        amsgrad: use the AMSGrad variant (max of historical ``v``).
        rectify: apply AdaBelief-Rectify, clamping the effective step size to
            the Adam step magnitude (``rt`` correction).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-16,
        weight_decay=0.0,
        amsgrad=False,
        rectify=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "rectify": rectify,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(param)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if group["amsgrad"]:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decay the first and second moment running averages. The
                # second moment tracks the residual (g - m), not g itself.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1.0 - beta2)
                if group["amsgrad"]:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                update = exp_avg / denom

                if group["rectify"]:
                    # AdaBelief-Rectify: limit the update magnitude to the
                    # upper bound of the equivalent Adam step ("rt" term).
                    # step_size * ||m / (sqrt(v)/sqrt(bc2))|| vs. lr * ||m||.
                    rt = min(
                        1.0,
                        bias_correction2_sqrt / (denom.max() / (exp_avg.norm() / (exp_avg.numel() ** 0.5) + 1e-12)),
                    ) if exp_avg.norm() > 0 else 1.0
                    step_size = step_size * rt

                if group["weight_decay"] > 0:
                    # Decoupled weight decay (AdamW style).
                    param.add_(param, alpha=-group["lr"] * group["weight_decay"])

                param.add_(update, alpha=-step_size * bias_correction2_sqrt)

        return loss
