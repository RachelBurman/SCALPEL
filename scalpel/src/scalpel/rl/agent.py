"""
PPO agent for SCALPEL retrieval optimisation.

Implements PPO from scratch:
  - Clipped surrogate objective
  - Generalised Advantage Estimation (GAE)
  - Value function loss (MSE)
  - Entropy bonus for exploration
  - Gradient clipping
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scalpel.rl.policy import ActorCritic
from scalpel.rl.environment import ACTION_DIMS, STATE_DIM, RetrievalParams


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PPOConfig:
    lr:            float = 3e-4
    gamma:         float = 0.99    # discount factor
    gae_lambda:    float = 0.95    # GAE lambda
    clip_eps:      float = 0.2     # PPO clip epsilon
    value_coef:    float = 0.5     # weight for value loss
    entropy_coef:  float = 0.01    # weight for entropy bonus
    n_epochs:      int   = 4       # gradient epochs per update
    batch_size:    int   = 32
    hidden_dim:    int   = 256
    max_grad_norm: float = 0.5


# ── Rollout buffer ────────────────────────────────────────────────────────────

@dataclass
class RolloutBuffer:
    """Accumulates transitions for one PPO update cycle."""

    states:    list = field(default_factory=list)
    actions:   list = field(default_factory=list)
    rewards:   list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    values:    list = field(default_factory=list)
    dones:     list = field(default_factory=list)

    def store(
        self,
        state: np.ndarray,
        action: list[int],
        reward: float,
        log_prob: float,
        value: float,
        done: bool = True,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(float(done))

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent that controls retrieval parameters.

    Typical training loop:
        agent = PPOAgent()
        for iteration in range(n_iterations):
            # collect rollouts externally via train.py
            agent.buffer.store(...)
            metrics = agent.update()
    """

    def __init__(
        self,
        config: PPOConfig | None = None,
        device: str = "cpu",
    ):
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.policy = ActorCritic(hidden_dim=self.config.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.buffer = RolloutBuffer()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _t(self, x, dtype=torch.float32) -> torch.Tensor:
        return torch.tensor(np.array(x), dtype=dtype).to(self.device)

    # ── Action selection ───────────────────────────────────────────────────

    def select_action(
        self, state: np.ndarray
    ) -> tuple[list[int], float, float]:
        """
        Sample an action from the current policy.

        Returns:
            (action indices, log_prob, value estimate)
        """
        state_t = self._t(state).unsqueeze(0)
        with torch.no_grad():
            actions, log_prob = self.policy.act(state_t)
            _, value = self.policy(state_t)
        return actions, log_prob.item(), value.squeeze().item()

    # ── GAE ────────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float = 0.0,
    ) -> np.ndarray:
        """Generalised Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = (
                rewards[t]
                + self.config.gamma * next_val * (1.0 - dones[t])
                - values[t]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    # ── PPO update ─────────────────────────────────────────────────────────

    def update(self) -> dict[str, float]:
        """
        Run PPO update on the current rollout buffer.

        Returns:
            Dict of averaged training metrics.
        """
        if len(self.buffer) == 0:
            return {}

        states    = self._t(self.buffer.states)
        actions   = self._t(self.buffer.actions, dtype=torch.long)
        old_lps   = self._t(self.buffer.log_probs)
        values_np = np.array(self.buffer.values,  dtype=np.float32)
        rewards_np= np.array(self.buffer.rewards, dtype=np.float32)
        dones_np  = np.array(self.buffer.dones,   dtype=np.float32)

        advantages = self._compute_gae(rewards_np, values_np, dones_np)
        returns    = advantages + values_np

        adv_t = self._t(advantages)
        ret_t = self._t(returns)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(self.buffer)
        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, total_loss=0.0)
        n_updates = 0

        for _ in range(self.config.n_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, self.config.batch_size):
                b = idx[start : start + self.config.batch_size]

                log_probs, values, entropy = self.policy.evaluate(states[b], actions[b])

                ratio = torch.exp(log_probs - old_lps[b])
                surr1 = ratio * adv_t[b]
                surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * adv_t[b]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = nn.functional.mse_loss(values, ret_t[b])
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef  * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"]  += value_loss.item()
                totals["entropy"]     += (-entropy_loss).item()
                totals["total_loss"]  += loss.item()
                n_updates += 1

        self.buffer.clear()
        return {k: v / max(n_updates, 1) for k, v in totals.items()}

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path | str):
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config":    self.config,
        }, path)

    def load(self, path: Path | str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    # ── Inference ──────────────────────────────────────────────────────────

    def get_params(self, query_embedding: np.ndarray) -> RetrievalParams:
        """
        Inference-time: select retrieval params for a query embedding.
        Called by the retrieval layer after training.
        """
        state_t = self._t(query_embedding).unsqueeze(0)
        with torch.no_grad():
            actions, _ = self.policy.act(state_t)
        return RetrievalParams.from_action(actions)
