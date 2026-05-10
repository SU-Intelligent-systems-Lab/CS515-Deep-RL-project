"""
Conjugate gradient solver and Fisher-vector product for TRPO.

The two core mathematical building blocks that separate TRPO from PPO:

1.  Fisher-vector product (Pearlmutter trick)
    Computing the full Fisher information matrix F is O(d²) where d is the
    number of policy parameters — completely intractable for neural nets.
    Instead, we can compute F·v for any vector v using only two backward passes:

        Step 1: compute g = ∂KL/∂θ  (with create_graph=True so we can
                differentiate g again)
        Step 2: compute (g · v).backward() → gives ∂(g·v)/∂θ = F·v

    The "Pearlmutter trick" name comes from the 1994 paper on efficient
    Hessian-vector products. Here we apply it to the KL Hessian = Fisher.

    We add a small damping term λI to F for numerical stability:
        Fv(v) = F·v + λ·v
    Typical λ = 0.1 (Schulman 2015 TRPO paper).

2.  Conjugate Gradient
    Solves the linear system (F + λI)·x = b  iteratively without forming F.
    Used to find the natural gradient direction x = F⁻¹·g where g is the
    ordinary policy gradient.
    10 iterations is standard (Schulman 2015, Algorithm 1).
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
    """Compute (F + damping·I)·v without materialising F.

    Args:
        policy : the policy network (PolicyNet). Its parameters are θ.
        obs    : a batch of observations (B, ob_dim). Should cover the whole
                 rollout for a stable Fisher estimate.
        v      : a flat parameter-space vector of shape (d,).
        damping: regularisation on the Fisher diagonal (default 0.1).

    Returns:
        Fv : flat tensor of shape (d,), equal to F·v + damping·v.

    How it works:
        1. Compute KL(π_fixed || π_θ) where π_fixed is a no-grad snapshot.
           The KL between two policies is the local curvature of the policy
           manifold — its Hessian w.r.t. θ IS the Fisher matrix.
        2. First backward: flat_grad = ∂KL/∂θ  (create_graph=True so we can
           differentiate again).
        3. Dot flat_grad with v (detached), then second backward:
           gives ∂(flat_grad·v)/∂θ = (∂²KL/∂θ²)·v = F·v.
    """
    # Step 1: build a fixed (detached) snapshot of the distribution.
    with torch.no_grad():
        dist_fixed = policy.distribution(obs)
        # Detach all distribution parameters so they act as constants.
        if hasattr(dist_fixed, 'logits'):
            # Categorical
            fixed_logits = dist_fixed.logits.detach()
            from torch.distributions import Categorical
            dist_fixed = Categorical(logits=fixed_logits)
        else:
            # Normal
            fixed_loc = dist_fixed.loc.detach()
            fixed_scale = dist_fixed.scale.detach()
            from torch.distributions import Normal
            dist_fixed = Normal(fixed_loc, fixed_scale)

    # Step 2: live distribution (θ participates in graph).
    dist_new = policy.distribution(obs)

    # KL divergence: KL(fixed || new) — average over batch.
    # torch.distributions.kl_divergence handles both Categorical and Normal.
    from torch.distributions import kl_divergence
    kl = kl_divergence(dist_fixed, dist_new)
    if kl.dim() > 1:
        # Normal returns per-dim KL; sum over action dims, mean over batch.
        kl = kl.sum(-1)
    kl = kl.mean()

    # Step 3: ∂KL/∂θ — keep graph so we can differentiate again.
    grads = autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    # Step 4: directional second derivative = F·v.
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
    """Solve Fv(x) = b iteratively using the Conjugate Gradient method.

    Args:
        Fv           : callable, takes a flat vector and returns F·v.
        b            : RHS vector (the policy gradient), flat shape (d,).
        n_steps      : max CG iterations (10 is standard for TRPO).
        residual_tol : stop early if ||r||² < residual_tol.

    Returns:
        x : approximate solution to F·x = b, shape (d,).

    Algorithm (standard PCG without preconditioner):
        x_0 = 0,  r_0 = b,  p_0 = b
        for k in 0..n_steps:
            α_k  = (r_k · r_k) / (p_k · F·p_k)
            x_{k+1} = x_k + α_k · p_k
            r_{k+1} = r_k − α_k · F·p_k
            β_k  = (r_{k+1} · r_{k+1}) / (r_k · r_k)
            p_{k+1} = r_{k+1} + β_k · p_k
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
