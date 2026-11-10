"""Terminal correctness and optional frozen prefix-value shaping."""
from bisect import bisect_right
import math

import torch
from torch import nn

from .rollout import EXPLORATION


class PrefixValue(nn.Module):
    def __init__(self, hidden_size=2048, width=256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(nn.Linear(hidden_size + 2, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, hidden, lengths):
        lengths = lengths.float()
        metadata = torch.stack((torch.log1p(lengths), lengths / 8192), dim=-1)
        return self.net(torch.cat((self.norm(hidden.detach().float()), metadata), dim=-1)).squeeze(-1)


def load_potential(path, device):
    saved = torch.load(path, weights_only=True, map_location="cpu")
    model = PrefixValue(**saved["architecture"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def answer_classes(nodes):
    classes = {}
    for i, node in enumerate(nodes):
        if node.finished and node.answer is not None:
            classes.setdefault(node.answer, []).append(i)
    if classes:
        return classes
    # Keep singleton classes with zero correctness when every EOS is unparseable.
    return {i: [i] for i, node in enumerate(nodes) if node.finished}


def vote_distribution(scores, classes):
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for group in classes.values():
        mask[group] = True
    weights = scores.masked_fill(~mask, -torch.inf).softmax(0)
    probabilities = torch.stack([weights[group].sum() for group in classes.values()])
    return torch.distributions.Categorical(
        probs=(1 - EXPLORATION) * probabilities + EXPLORATION / len(classes)
    )


def returns_to_go(rewards):
    total = 0.
    result = []
    for reward in reversed(rewards):
        total += reward
        result.append(total)
    return result[::-1]


@torch.no_grad()
def action_returns(nodes, terminal, potential, weight, device):
    if potential is None:
        return [terminal] * (len(nodes) - 1)

    hidden = torch.stack([n.hidden for n in nodes]).to(device)
    lengths = torch.tensor([len(n.tokens) for n in nodes], device=device)
    values = potential(hidden, lengths).sigmoid().tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite prefix reward")

    seen = {nodes[0].tokens: values[0]}
    best = values[0]
    rewards = []
    for node, value in zip(nodes[1:], values[1:]):
        value = seen.setdefault(node.tokens, value)
        rewards.append(weight * (value - best))
        best = max(best, value)
    return returns_to_go(rewards + [terminal])[:-1]


def leave_one_out(returns, progress):
    advantages = []
    for i, (rewards, steps) in enumerate(zip(returns, progress)):
        row = []
        for reward, step in zip(rewards, steps):
            others = [
                values[max(0, bisect_right(other_steps, step) - 1)]
                for j, (values, other_steps) in enumerate(zip(returns, progress)) if j != i
            ]
            row.append(reward - sum(others) / len(others))
        advantages.append(row)
    return advantages
