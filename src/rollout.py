"""Frozen generator and causal tree collection for controller training."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re

import torch


MODEL = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PATH_LIMIT = 8192
TEMPERATURE = 0.5
EXPLORATION = 0.1


def stream_seed(seed, *parts):
    key = json.dumps((seed, *parts), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") % (2**63)


def answer_key(text):
    return "".join(text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac").split())


def boxed_answer(text):
    for match in reversed(list(re.finditer(r"\\boxed\{", text))):
        depth = 1
        for end in range(match.end(), len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            if depth == 0:
                return answer_key(text[match.end():end]) or None
    return None


class FrozenQwen:
    def __init__(self, device, hidden_layer):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.hidden_layer = hidden_layer
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError("Use a CUDA or CPU device")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=dtype)
        self.model.to(self.device).eval().requires_grad_(False)
        self.eos = self.model.generation_config.eos_token_id

    def prompt(self, question):
        messages = [
            {"role": "system", "content": r"Reason step by step and put the final answer in \boxed{}."},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return tuple(self.tokenizer.encode(text, add_special_tokens=False))

    def ids(self, tokens):
        return torch.tensor([tokens], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def hidden(self, tokens):
        intermediate = self.hidden_layer != -1
        output = self.model.base_model(
            input_ids=self.ids(tokens), use_cache=False, output_hidden_states=intermediate,
        )
        hidden = output.hidden_states[self.hidden_layer] if intermediate else output.last_hidden_state
        return hidden[0, -1].cpu()

    @torch.inference_mode()
    def draw(self, tokens, allowance, seed):
        ids = self.ids(tokens)
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            output = self.model.generate(
                ids, attention_mask=torch.ones_like(ids), max_new_tokens=allowance,
                do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                eos_token_id=self.eos, pad_token_id=self.model.generation_config.pad_token_id,
            )
        suffix = tuple(output[0, ids.shape[-1]:].tolist())
        eos = self.eos if isinstance(self.eos, list) else [self.eos]
        return suffix, bool(suffix and suffix[-1] in eos)

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


@dataclass
class Node:
    parent: int
    tokens: tuple[int, ...]
    hidden: torch.Tensor
    finished: bool = False
    answer: str | None = None


@dataclass
class Action:
    count: int
    node: int
    spent: int
    old_log_prob: float | None


def observation(nodes, count, device):
    observed = nodes[:count]
    return (
        torch.stack([n.hidden for n in observed]).to(device),
        torch.tensor([n.finished for n in observed], device=device),
        torch.tensor([n.parent for n in observed], device=device),
    ), {"token_lengths": torch.tensor([len(n.tokens) for n in observed], device=device)}


def expansion_mask(args, kwargs):
    return ~args[1] & (kwargs["token_lengths"] < PATH_LIMIT)


def action_distribution(scores, valid):
    probabilities = (scores / TEMPERATURE).masked_fill(~valid, -torch.inf).softmax(0)
    return torch.distributions.Categorical(
        probs=(1 - EXPLORATION) * probabilities + EXPLORATION * valid.float() / valid.sum()
    )


def collect(generator, prompt, seed, budget, chunk, controller=None):
    nodes = [Node(-1, (), generator.hidden(prompt))]
    visits = Counter()
    actions = []
    spent = 0
    rng = torch.Generator().manual_seed(seed)
    device = next(controller.parameters()).device if controller is not None else "cpu"
    while spent < budget:
        count = len(nodes)
        if controller is None:
            tip = nodes[-1]
            parent = 0 if tip.finished or len(tip.tokens) == PATH_LIMIT else count - 1
            old_log_prob = None
        else:
            args, kwargs = observation(nodes, count, device)
            with torch.no_grad():
                dist = action_distribution(controller(*args, **kwargs), expansion_mask(args, kwargs))
                parent = int(torch.multinomial(dist.probs.cpu(), 1, generator=rng))
                old_log_prob = float(dist.log_prob(torch.tensor(parent, device=device)))
        prefix = nodes[parent].tokens
        allowance = min(chunk, budget - spent, PATH_LIMIT - len(prefix))
        draw_seed = stream_seed(seed, "expansion", prefix, visits[prefix])
        suffix, eos = generator.draw(prompt + prefix, allowance, draw_seed)
        if not 0 < len(suffix) <= allowance:
            raise ValueError("The frozen generator exceeded its allowance")
        visits[prefix] += 1
        tokens = prefix + suffix
        answer = boxed_answer(generator.decode(tokens)) if eos else None
        actions.append(Action(count, parent, spent, old_log_prob))
        nodes.append(Node(parent, tokens, generator.hidden(prompt + tokens), eos, answer))
        spent += len(suffix)
    return nodes, actions
