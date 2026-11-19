"""PolOpt optimizer.

AdamW variant used by the Politrees (PolTrain) RVC training fork. On top of
standard AdamW with decoupled weight decay it clips the *effective update*
of each parameter to ``max_step_clip * lr`` per optimizer step. This bounds
how far a single noisy spectral-loss batch can move the generator weights,
which stabilises the early epochs of RVC training (the generator and the
discriminator are updated with different learning rates and separate
``max_step_clip`` values by the caller).

Parameters use the same state layout as ``torch.optim.AdamW`` so existing
checkpoints (``checkpoint.pth``) stay loadable.
"""

import math

import torch
from torch.optim.optimizer import Optimizer


class PolOpt(Optimizer):
    """AdamW with per-parameter update clipping.

    Args:
        params: iterable of parameters to optimize or dicts of parameter groups.
        lr: learning rate (default ``1e-3``).
        betas: coefficients for the moment estimates (default ``(0.8, 0.99)``).
        eps: term added to the denominator for numerical stability.
        weight_decay: decoupled weight decay coefficient (AdamW style).
        max_step_clip: maximum allowed magnitude of the normalized update per
            step, expressed in multiples of ``lr``. ``1.0`` (default) means a
            single step can change a weight by at most ``lr``.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.8, 0.99),
        eps=1e-8,
        weight_decay=0.01,
        max_step_clip=1.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= max_step_clip:
            raise ValueError(f"Invalid max_step_clip value: {max_step_clip}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "max_step_clip": max_step_clip,
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
            clip_limit = group["max_step_clip"] * group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("PolOpt does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Standard AdamW moment updates.
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                update = exp_avg / bias_correction1 / denom

                # AdamW-style decoupled weight decay.
                if group["weight_decay"] > 0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])

                # Clip the FINAL step so one update can never change a weight
                # by more than max_step_clip * lr in absolute value.
                step_update = group["lr"] * update
                if clip_limit > 0:
                    step_update.clamp_(-clip_limit, clip_limit)
                param.add_(step_update, alpha=-1.0)

        return loss
