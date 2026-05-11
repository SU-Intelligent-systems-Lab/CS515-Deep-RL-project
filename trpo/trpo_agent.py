"""
TRPOAgent: trust-region policy update.

Maximise the surrogate E[ratio * A] subject to KL(pi_old || pi_new) <= delta.
The policy step uses the natural gradient via conjugate gradient + a
backtracking line search; the value net is trained separately with Adam.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn.functional as F

from . import ptu
from .conjugate_gradient import conjugate_gradient, fisher_vector_product
from .networks import PolicyNet, ValueNet
from .rollout_buffer import RolloutBuffer


# flat-parameter helpers used by the line search

def get_flat_params(net: torch.nn.Module) -> torch.Tensor:
    """Flatten all parameters of `net` into a 1-D tensor."""
    return torch.cat([p.data.reshape(-1) for p in net.parameters()])


def set_flat_params(net: torch.nn.Module, flat_params: torch.Tensor) -> None:
    """Write a flat parameter vector back into `net` in-place."""
    offset = 0
    for p in net.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset : offset + numel].reshape(p.shape))
        offset += numel


def get_flat_grad(loss: torch.Tensor, net: torch.nn.Module) -> torch.Tensor:
    """Flat gradient of `loss` w.r.t. all parameters of `net`."""
    grads = autograd.grad(loss, net.parameters(), retain_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


class TRPOAgent:
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        max_kl: float = 0.01,
        cg_steps: int = 10,
        cg_damping: float = 0.1,
        backtrack_steps: int = 10,
        backtrack_coef: float = 0.5,
        value_epochs: int = 5,
        n_layers: int = 2,
        size: int = 64,
    ) -> None:
        self.max_kl = max_kl
        self.cg_steps = cg_steps
        self.cg_damping = cg_damping
        self.backtrack_steps = backtrack_steps
        self.backtrack_coef = backtrack_coef
        self.value_epochs = value_epochs
        self.discrete = discrete

        self.policy = PolicyNet(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            discrete=discrete,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

        self.value_net = ValueNet(
            ob_dim=ob_dim,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

        # value net uses Adam; the policy has no optimiser (natural gradient step)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)

    def act(self, obs_np: np.ndarray):
        """Return (action, log_prob, value), same interface as PPOAgent.act."""
        action, log_prob = self.policy.act(obs_np)
        value = self.value_net.get_value(obs_np)
        return action, log_prob, value

    def bootstrap_value(self, obs_np: np.ndarray) -> float:
        """V(obs) for the rollout-end bootstrap."""
        return self.value_net.get_value(obs_np)

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run one TRPO update over a full rollout buffer.

        Steps:
          1. normalise advantages
          2. compute the surrogate loss and its flat gradient g
          3. build the Fisher-vector product callable Fv
          4. solve F x = g with conjugate gradient
          5. step size = sqrt(2 delta / (x^T F x))
          6. backtracking line search subject to KL <= delta and improvement
          7. value-net Adam steps (MSE)
        """
        obs_t = ptu.from_numpy(buffer.obs)                    # (N, ob_dim)
        if self.discrete:
            actions_t = torch.from_numpy(buffer.actions).to(ptu.device)
        else:
            actions_t = ptu.from_numpy(buffer.actions)
        old_log_probs_t = ptu.from_numpy(buffer.log_probs)    # (N,)
        returns_t = ptu.from_numpy(buffer.returns)            # (N,)

        advantages_np = buffer.advantages.copy()
        adv_mean = float(np.mean(advantages_np))
        adv_std = float(np.std(advantages_np)) + 1e-8
        advantages_t = ptu.from_numpy((advantages_np - adv_mean) / adv_std)

        # explained variance, computed before the value update
        returns_all = buffer.returns.copy()
        values_all = buffer.values.copy()
        var_returns = np.var(returns_all)
        explained_var = (
            1.0 - np.var(returns_all - values_all) / var_returns
            if var_returns > 0
            else 0.0
        )

        # 2. surrogate loss and its gradient
        # surrogate = E[ exp(new_logp - old_logp) * A ]; we maximise it.
        new_log_probs, entropy = self.policy.evaluate_actions(obs_t, actions_t)
        entropy_mean = entropy.mean()

        ratio = torch.exp(new_log_probs - old_log_probs_t)
        surrogate = (ratio * advantages_t).mean()
        policy_loss_before = -surrogate.item()

        # differentiate -surrogate for convenience, then flip sign to get the
        # ascent direction g = d(surrogate)/d(theta)
        surrogate_loss = -surrogate
        policy_grad_flat = -get_flat_grad(surrogate_loss, self.policy)

        # 3. Fisher-vector product callable closed over obs_t
        def Fv(v: torch.Tensor) -> torch.Tensor:
            return fisher_vector_product(
                self.policy, obs_t, v, damping=self.cg_damping
            )

        # 4. natural-gradient direction via conjugate gradient
        # cannot wrap in torch.no_grad: autograd.grad is needed inside FVP
        natural_grad = conjugate_gradient(
            Fv, policy_grad_flat.detach(), n_steps=self.cg_steps
        ).detach()

        # 5. step size from the trust-region constraint
        Fng = Fv(natural_grad)
        sHs = (natural_grad * Fng).sum().item()
        step_size = (2.0 * self.max_kl / (sHs + 1e-8)) ** 0.5
        fullstep = step_size * natural_grad

        # 6. backtracking line search
        old_params = get_flat_params(self.policy).detach().clone()

        def surrogate_and_kl(params: torch.Tensor):
            """Set the policy to `params`, return (surrogate, mean KL)."""
            set_flat_params(self.policy, params)
            with torch.no_grad():
                lp_new, _ = self.policy.evaluate_actions(obs_t, actions_t)
                ratio_new = torch.exp(lp_new - old_log_probs_t)
                surr_new = (ratio_new * advantages_t).mean().item()
                # quadratic approximation of KL(old || new)
                kl_new = 0.5 * ((old_log_probs_t - lp_new) ** 2).mean().item()
            return surr_new, kl_new

        expected_improve = (policy_grad_flat * fullstep).sum().item()
        accept_ratio = 0.1  # minimum fraction of expected improvement we accept

        step_accepted = False
        backtrack_iters = 0
        for i in range(self.backtrack_steps):
            frac = self.backtrack_coef ** i
            new_params = old_params + frac * fullstep
            surr_new, kl_new = surrogate_and_kl(new_params)
            actual_improve = surr_new - (-policy_loss_before)

            if kl_new <= self.max_kl and actual_improve >= accept_ratio * frac * expected_improve:
                step_accepted = True
                backtrack_iters = i
                break
        else:
            # no valid step found in the line search, revert to old policy
            set_flat_params(self.policy, old_params)
            backtrack_iters = self.backtrack_steps

        # KL at the final parameters
        with torch.no_grad():
            lp_final, _ = self.policy.evaluate_actions(obs_t, actions_t)
            approx_kl = 0.5 * ((old_log_probs_t - lp_final) ** 2).mean().item()

        # 7. value net: a few Adam steps on MSE
        value_losses = []
        for _ in range(self.value_epochs):
            predicted = self.value_net(obs_t)
            v_loss = F.mse_loss(predicted, returns_t)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()
            value_losses.append(v_loss.item())

        return {
            "losses/policy_loss": float(policy_loss_before),
            "losses/value_loss": float(np.mean(value_losses)),
            "losses/entropy": float(entropy_mean.item()),
            "trpo/approx_kl": float(approx_kl),
            "trpo/backtrack_iters": float(backtrack_iters),
            "trpo/step_accepted": float(step_accepted),
            "trpo/explained_variance": float(explained_var),
        }
