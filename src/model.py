"""Checkpoint-compatible small tree scorer."""
import json
import math
from pathlib import Path

import torch
from torch import nn

from .rollout import MODEL, REVISION


SMALL_POLICY = dict(hidden_size=2048, width=256, layers=3, heads=4, use_hidden=True)
LARGE_POLICY = dict(hidden_size=2048, width=512, layers=4, heads=8, use_hidden=True)
FEATURE_LAYER = 14


def tree_features(hidden, finished, parents, token_lengths):
    flags = finished.tolist()
    lengths = token_lengths.tolist()
    parent_ids = parents.tolist()
    if not lengths or len(hidden) != len(lengths) or len(flags) != len(lengths) or len(parent_ids) != len(lengths):
        raise ValueError("Expected equally sized, nonempty tree inputs")
    if parent_ids[0] != -1 or lengths[0] != 0:
        raise ValueError("Expected a root with zero generated tokens")

    depth = [0] * len(lengths)
    children = [0] * len(lengths)
    spent = 0
    for child, parent in enumerate(parent_ids[1:], start=1):
        if not 0 <= parent < child or flags[parent] or lengths[child] <= lengths[parent]:
            raise ValueError("Expected ordered, positive-length edges from unfinished parents")
        depth[child] = depth[parent] + 1
        children[parent] += 1
        spent += lengths[child] - lengths[parent]

    metadata = hidden.new_tensor([
        [float(flags[i]), float(i == 0), math.log1p(depth[i]),
         math.log1p(lengths[i]), math.log1p(children[i])]
        for i in range(len(lengths))
    ])
    context = hidden.new_tensor([
        math.log1p(len(lengths)), math.log1p(sum(flags)), math.log1p(spent), spent / 32768,
    ])
    return metadata, context


class DirectedSumAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.keys = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.values = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.final_norm = nn.LayerNorm(width)

    def forward(self, nodes, parents):
        indices = torch.arange(len(nodes), device=nodes.device)
        children, ancestors = indices[parents >= 0], parents[parents >= 0]
        shape = (-1, self.heads, self.head_width)
        query = self.query(nodes).reshape(shape)
        aggregated = nodes.new_zeros((len(nodes), self.heads, self.head_width))

        # Separate messages for self, parent-to-child, and child-to-parent edges.
        relations = ((indices, indices), (ancestors, children), (children, ancestors))
        for (source, target), key_layer, value_layer in zip(relations, self.keys, self.values):
            keys = key_layer(nodes).reshape(shape)
            values = value_layer(nodes).reshape(shape)
            weights = ((query[target] * keys[source]).sum(-1) / self.head_width**0.5).sigmoid()
            aggregated.index_add_(0, target, values[source] * weights[..., None])

        updated = self.norm(nodes + self.output(aggregated.flatten(1)))
        return self.final_norm(updated + self.feedforward(updated))


class TreeBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.local = DirectedSumAttention(width, heads)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(width)
        self.update = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.update_norm = nn.LayerNorm(width)

    def forward(self, nodes, parents):
        nodes = self.local(nodes, parents)
        normalized = self.attention_norm(nodes).unsqueeze(0)
        message, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        nodes = nodes + message.squeeze(0)
        return nodes + self.update(self.update_norm(nodes))


class SingleScoreController(nn.Module):
    def __init__(self, policy=None, hidden_layer=FEATURE_LAYER):
        super().__init__()
        self.config = dict(SMALL_POLICY if policy is None else policy)
        if self.config not in (SMALL_POLICY, LARGE_POLICY):
            raise ValueError("Expected the small or large single-score policy config")
        if hidden_layer not in (-1, FEATURE_LAYER):
            raise ValueError("Expected Qwen hidden layer 14 or the legacy final layer (-1)")
        self.hidden_layer = hidden_layer
        width, hidden_size = self.config["width"], self.config["hidden_size"]
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.input = nn.Linear(hidden_size, width)
        self.node_input = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, width))
        self.context_input = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            TreeBlock(width, self.config["heads"]) for _ in range(self.config["layers"])
        )
        self.score = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, hidden, finished, parents, *, token_lengths):
        hidden = hidden.detach().float()
        metadata, context = tree_features(hidden, finished, parents, token_lengths)
        nodes = self.input(self.hidden_norm(hidden)) + self.node_input(metadata)
        nodes = nodes + self.context_input(context)[None]
        for block in self.blocks:
            nodes = block(nodes, parents)
        pooled = nodes.mean(0, keepdim=True).expand_as(nodes)
        return self.score(torch.cat((nodes, pooled), dim=-1)).squeeze(-1)


def checkpoint_path(checkpoint):
    source = str(checkpoint)
    if not source.startswith("hf://"):
        return Path(source)
    from huggingface_hub import hf_hub_download

    try:
        owner, repository, filename = source.removeprefix("hf://").split("/", 2)
    except ValueError as error:
        raise ValueError("Use hf://owner/repo[@revision]/file.safetensors") from error
    repo, _, revision = repository.partition("@")
    if not owner or not repo or not filename.endswith(".safetensors"):
        raise ValueError("Use hf://owner/repo[@revision]/file.safetensors")
    return Path(hf_hub_download(f"{owner}/{repo}", filename, revision=revision or None))


def load_controller(checkpoint, device, policy=None):
    if checkpoint is None:
        return SingleScoreController(policy).to(device).eval()
    path = checkpoint_path(checkpoint)
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as saved:
            metadata = saved.metadata() or {}
            if (metadata.get("model"), metadata.get("revision")) != (MODEL, REVISION):
                raise ValueError("Controller uses a different frozen generator")
            config = json.loads(metadata["policy"])
            hidden_layer = int(metadata["hidden_layer"])
            state = {name: saved.get_tensor(name) for name in saved.keys()}
    else:
        saved = torch.load(path, weights_only=True, map_location="cpu")
        config = saved.get("policy") or saved.get("contract", {}).get("policy")
        # Standalone checkpoints predating the feature-layer field used the final layer.
        legacy_local = saved.get("model") == MODEL and saved.get("method") in ("imitate", "ppo")
        hidden_layer = saved.get("hidden_layer", -1 if legacy_local else FEATURE_LAYER)
        state = saved["model_state"]
    if policy is not None and config != policy:
        raise ValueError("Checkpoint policy does not match --config")
    if config is None:
        raise ValueError("Checkpoint has no controller policy config")
    model = SingleScoreController(config, hidden_layer)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
