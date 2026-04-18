"""
Actor-Critic policy network for the SCALPEL PPO agent.

Architecture:
  Shared trunk:  MLP over query embedding → latent representation
  Actor heads:   one Categorical head per action dimension (multi-discrete)
  Critic head:   scalar value estimate V(s)
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from scalpel.rl.environment import ACTION_DIMS, STATE_DIM


class ActorCritic(nn.Module):
    """
    Shared-trunk Actor-Critic for a multi-discrete action space.

    The trunk is shared so the value function and policy learn a common
    representation of the query embedding, which speeds convergence on
    small datasets.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # One head per action dimension
        self.actor_heads = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for dim in ACTION_DIMS
        ])

        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[list[Categorical], torch.Tensor]:
        """
        Args:
            state: [batch, STATE_DIM]

        Returns:
            (list of Categorical distributions, value [batch, 1])
        """
        features = self.trunk(state)
        distributions = [Categorical(logits=head(features)) for head in self.actor_heads]
        value = self.critic_head(features)
        return distributions, value

    def act(
        self, state: torch.Tensor
    ) -> tuple[list[int], torch.Tensor]:
        """
        Sample an action (no grad).

        Returns:
            (action index list, summed log_prob scalar tensor)
        """
        distributions, _ = self(state)
        actions = [d.sample() for d in distributions]
        log_prob = sum(d.log_prob(a) for d, a in zip(distributions, actions))
        return [a.item() for a in actions], log_prob

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate stored (state, action) pairs for the PPO update.

        Args:
            states:  [batch, STATE_DIM]
            actions: [batch, n_action_dims]  (long tensor)

        Returns:
            log_probs [batch], values [batch], entropy [batch]
        """
        distributions, values = self(states)
        log_probs = sum(
            d.log_prob(actions[:, i]) for i, d in enumerate(distributions)
        )
        entropy = sum(d.entropy() for d in distributions)
        return log_probs, values.squeeze(-1), entropy
