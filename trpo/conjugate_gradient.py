"""
Fisher-vector product and conjugate gradient solver for TRPO.

The Fisher matrix F is the Hessian of the KL between the old and new
policies. Computing F directly is intractable, so we expose only the
matrix-vector product Fv using two backward passes, and solve F x = g
iteratively with conjugate gradient.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.autograd as autograd


def fisher_vector_product(
    policy: torch.nn.Module,
    obs: torch.Tensor,
    v: torch.Tensor,
    damping: float = 0.1,
) -> torch.Tensor:
    """Compute (F + damping * I) * v without forming F.

    Args:
        policy : PolicyNet with parameters theta.
        obs    : batch of observations covering the rollout (B, ob_dim).
        v      : flat parameter-space vector of shape (d,).
        damping: small ridge on the Fisher diagonal for numerical stability.
    """
    # detached snapshot of the current distribution (acts as constant pi_old)
    with torch.no_grad():
        dist_fixed = policy.distribution(obs)
        if hasattr(dist_fixed, 'logits'):
            fixed_logits = dist_fixed.logits.detach()
            from torch.distributions import Categorical
            dist_fixed = Categorical(logits=fixed_logits)
        else:
            fixed_loc = dist_fixed.loc.detach()
            fixed_scale = dist_fixed.scale.detach()
            from torch.distributions import Normal
            dist_fixed = Normal(fixed_loc, fixed_scale)

    # live distribution (theta is in the graph)
    dist_new = policy.distribution(obs)

    from torch.distributions import kl_divergence
    kl = kl_divergence(dist_fixed, dist_new)
    if kl.dim() > 1:
        kl = kl.sum(-1)
    kl = kl.mean()

    # first backward: d KL / d theta (keep graph for a second backward)
    grads = autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    # second backward on (flat_grad . v) gives F * v
    grad_v = (flat_grad * v.detach()).sum()
    grads2 = autograd.grad(grad_v, policy.parameters(), retain_graph=False)
    flat_fvp = torch.cat([g.reshape(-1) for g in grads2])

    return flat_fvp + damping * v


def conjugate_gradient(
    Fv: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    n_steps: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve Fv(x) = b with conjugate gradient, returns x.

    Args:
        Fv           : callable taking a flat vector v and returning F * v.
        b            : right-hand side (the policy gradient), flat (d,).
        n_steps      : max CG iterations.
        residual_tol : stop when the squared residual drops below this.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr = torch.dot(r, r)

    for _ in range(n_steps):
        Ap = Fv(p)
        pAp = torch.dot(p, Ap)
        alpha = rr / (pAp + 1e-8)

        x = x + alpha * p
        r = r - alpha * Ap

        new_rr = torch.dot(r, r)
        if new_rr.item() < residual_tol:
            break

        beta = new_rr / (rr + 1e-8)
        p = r + beta * p
        rr = new_rr

    return x
