"""Train the small controller with SC imitation or grouped PPO."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import tempfile

import torch

from .data import read_questions
from .model import load_controller
from .rewards import action_returns, answer_classes, leave_one_out, load_potential, vote_distribution
from .rollout import (
    MODEL, REVISION, FrozenQwen, action_distribution, answer_key, collect,
    expansion_mask, observation, stream_seed,
)


CLIP = 0.2
ENTROPY_WEIGHT = 1e-4


def imitate(controller, optimizer, nodes, actions, device):
    uniform_states = sum(any(n.finished for n in nodes[:a.count]) for a in actions)
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for action in actions:
        args, kwargs = observation(nodes, action.count, device)
        scores = controller(*args, **kwargs)
        valid = expansion_mask(args, kwargs)
        loss = -scores.masked_fill(~valid, -torch.inf).log_softmax(0)[action.node] / len(actions)
        if args[1].any():
            eos = scores[args[1]].double()
            uniform_kl = -eos.log_softmax(0).mean() - eos.new_tensor(len(eos)).log()
            variance = (eos - eos.mean()).square().mean()
            loss = loss + (uniform_kl + variance) / uniform_states
        loss.backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return total


@dataclass
class Episode:
    nodes: list
    actions: list
    classes: dict
    labels: torch.Tensor
    old_votes: torch.Tensor | None
    terminal: float
    returns: list[float]


@torch.no_grad()
def episode(controller, generator, prompt, seed, config, gold, potential):
    nodes, actions = collect(generator, prompt, seed, config.budget, config.chunk, controller)
    classes = answer_classes(nodes)
    labels = torch.tensor([float(answer == gold) for answer in classes], device=config.device)
    old_votes, terminal = None, 0.
    if classes:
        args, kwargs = observation(nodes, len(nodes), config.device)
        old_votes = vote_distribution(controller(*args, **kwargs), classes).probs
        terminal = float(old_votes @ labels)
    returns = action_returns(nodes, terminal, potential, config.shaping_weight, config.device)
    return Episode(nodes, actions, classes, labels, old_votes, terminal, returns)


def clipped_gain(ratio, advantage):
    return torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage)


def ppo(controller, optimizer, episodes, device):
    advantages = leave_one_out(
        [e.returns for e in episodes], [[a.spent for a in e.actions] for e in episodes]
    )
    states = sum(len(e.actions) + bool(e.classes) for e in episodes)
    entropy_scale = ENTROPY_WEIGHT / states
    for epoch in range(2):
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for e, row in zip(episodes, advantages):
            for action, advantage in zip(e.actions, row):
                args, kwargs = observation(e.nodes, action.count, device)
                dist = action_distribution(controller(*args, **kwargs), expansion_mask(args, kwargs))
                log_prob = dist.log_prob(torch.tensor(action.node, device=device))
                change = log_prob - action.old_log_prob
                if epoch == 0 and abs(float(change.detach())) > 1e-4:
                    raise ValueError("Sampled expansion probability changed before the update")
                loss = -clipped_gain(change.exp(), advantage) / len(episodes) - entropy_scale * dist.entropy()
                loss.backward()
                total += float(loss.detach())

            if e.classes:
                args, kwargs = observation(e.nodes, len(e.nodes), device)
                dist = vote_distribution(controller(*args, **kwargs), e.classes)
                ratio = dist.probs / e.old_votes
                if epoch == 0 and float(ratio.detach().log().abs().max()) > 1e-4:
                    raise ValueError("Terminal class probabilities changed before the update")
                gain = (e.old_votes * clipped_gain(ratio, e.labels - e.terminal)).sum()
                loss = -gain / len(episodes) - entropy_scale * dist.entropy()
                loss.backward()
                total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    return total


def save_checkpoint(path, controller, config, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as f:
        temporary = Path(f.name)
    try:
        torch.save(dict(
            policy=controller.config, hidden_layer=controller.hidden_layer,
            model_state={key: tensor.detach().cpu() for key, tensor in controller.state_dict().items()},
            epoch=epoch, method=config.method, reward=getattr(config, "reward", None),
            model=MODEL, revision=REVISION, budget=config.budget, chunk=config.chunk,
        ), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="method", required=True)
    for method in ("imitate", "ppo"):
        command = commands.add_parser(method)
        command.add_argument("output", type=Path)
        command.add_argument("--dataset", required=True, help="math, numina, deepmath or a JSONL path")
        command.add_argument("--limit", type=int, help="Maximum number of training questions")
        command.add_argument("--config", type=Path, help="Controller architecture JSON")
        command.add_argument("--epochs", type=int, default=1)
        command.add_argument("--budget", type=int, default=8192)
        command.add_argument("--chunk", type=int, choices=(128, 512), default=128)
        command.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--lr", type=float, default=1e-4)
        if method == "ppo":
            command.add_argument("--init", required=True)
            command.add_argument("--reward", choices=("terminal", "prefix"), default="terminal")
            command.add_argument("--potential", type=Path)
            command.add_argument("--shaping-weight", type=float, default=0.2)
        else:
            command.add_argument("--init")
    config = parser.parse_args()
    if config.epochs < 1 or config.budget < 1 or not 0 < config.lr < float("inf"):
        parser.error("Expected positive epochs, budget, and learning rate")
    if config.limit is not None and config.limit < 1:
        parser.error("--limit must be positive")
    if (config.init is not None and not config.init.startswith("hf://")
            and Path(config.init).resolve() == config.output.resolve()):
        parser.error("Choose a different output checkpoint from --init")
    if config.method == "ppo":
        if (config.reward == "prefix") != (config.potential is not None):
            parser.error("--potential is required exactly when --reward prefix is used")
        if config.reward == "prefix" and not 0 < config.shaping_weight <= 1:
            parser.error("Expected a shaping weight in (0, 1]")
    return config


def main():
    config = parse_args()
    is_ppo = config.method == "ppo"
    rows = read_questions(config.dataset, with_answers=is_ppo, limit=config.limit)
    torch.manual_seed(config.seed)
    policy = json.loads(config.config.read_text()) if config.config else None
    controller = load_controller(config.init, config.device, policy)
    potential = load_potential(config.potential, config.device) if is_ppo and config.potential else None
    generator = FrozenQwen(config.device, controller.hidden_layer)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=config.lr, weight_decay=0.)
    order_rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        for index in order_rng.sample(range(len(rows)), len(rows)):
            row = rows[index]
            prompt = generator.prompt(row["question"])
            seed = stream_seed(config.seed, epoch, index)
            if is_ppo:
                gold = answer_key(row["answer"])
                episodes = [
                    episode(controller, generator, prompt, stream_seed(seed, trial), config, gold, potential)
                    for trial in range(4)
                ]
                loss = ppo(controller, optimizer, episodes, config.device)
            else:
                nodes, actions = collect(generator, prompt, seed, config.budget, config.chunk)
                loss = imitate(controller, optimizer, nodes, actions, config.device)
            print(json.dumps(dict(epoch=epoch, question_index=index, loss=loss)), flush=True)
        save_checkpoint(config.output, controller, config, epoch)


if __name__ == "__main__":
    main()
