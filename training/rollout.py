"""
Episode collection for preference-conditioned policies.

collect_episode() runs one full episode and returns:
  - log_probs : list of scalar Tensors  — log π(aₜ | sₜ, λ), one per step
  - costs     : np.ndarray (T, 3)       — raw per-step cost vectors from the env

Costs are returned un-negated and un-scalarized so this function can be shared
between the single-λ REINFORCE trainer (Phase 2) and the PSL trainer (Phase 3).
The caller is responsible for converting costs to rewards and applying λ.
"""

import numpy as np
import torch

from envs import MOHighwayEnv
from models import ConditionedPolicy


def collect_episode(
    env: MOHighwayEnv,
    policy: ConditionedPolicy,
    lam: np.ndarray,
    gamma: float = 0.99,
    device: str = "cpu",
) -> tuple[list[torch.Tensor], np.ndarray]:
    """
    Run one episode and collect log-probabilities and cost vectors.

    Parameters
    ----------
    env    : MOHighwayEnv  — must already be constructed; reset() is called here
    policy : ConditionedPolicy
    lam    : np.ndarray (3,)  — preference vector, sums to 1
    gamma  : float  — discount factor for return computation
    device : str

    Returns
    -------
    log_probs : list of Tensor scalars, length T
        log π(aₜ | sₜ, λ) for each timestep. Kept in the computation graph.
    returns   : np.ndarray (T, 3)
        Discounted cumulative cost per objective from each timestep onward:
        Gᵢ(t) = Σ_{t'≥t} γ^(t'-t) · cost_i(t')
    """
    lam_t = torch.tensor(lam, dtype=torch.float32, device=device)

    obs, _ = env.reset()
    log_probs: list[torch.Tensor] = []
    step_costs: list[np.ndarray] = []

    done = False
    while not done:
        obs_t = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
        action, log_prob = policy.act(obs_t, lam_t)
        obs, cost_vec, terminated, truncated, _ = env.step(action)

        log_probs.append(log_prob)
        step_costs.append(cost_vec)          # shape (3,)
        done = terminated or truncated

    # --- Compute discounted returns per objective (backward pass) ---
    T = len(step_costs)
    costs_arr = np.stack(step_costs)         # (T, 3)
    returns = np.zeros_like(costs_arr)       # (T, 3)

    G = np.zeros(3)
    for t in reversed(range(T)):
        G = costs_arr[t] + gamma * G
        returns[t] = G

    return log_probs, returns


def episode_summary(returns: np.ndarray) -> dict:
    """
    Summarise the per-objective cumulative costs for logging.
    returns : np.ndarray (T, 3)
    """
    G0 = returns[0]   # total discounted cost from t=0
    return {
        "G_safety":  float(G0[0]),
        "G_speed":   float(G0[1]),
        "G_comfort": float(G0[2]),
        "G_total":   float(G0.sum()),
        "length":    len(returns),
    }
