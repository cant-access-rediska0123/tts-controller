# TTS Controller

Train a small tree scorer around frozen Qwen3-1.7B. Self-consistency (SC)
imitation learns when to continue the current path or restart from the root.
PPO can sample any unfinished node with room to grow.

## Datasets

### Hugging Face

Pass a short name to `--dataset`. The loader reads the pinned Hub TRAIN split.

| Name | Hub dataset | Question → final answer |
| --- | --- | --- |
| `math` | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3) (MATH mirror, all seven subjects) | `problem` → last box in `solution` |
| `numina` | [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5/tree/1b05109f9e5c1ad06c0663519502416c30b300f8) | `problem` → `answer` |
| `deepmath` | [zwhe99/DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K/tree/5cf055d1fe3d7a2eb19719ac020211469736ae44) | `question` → `final_answer` |

```bash
tts-controller-train imitate checkpoints/sc.pt --dataset math --limit 200
```

Use `--limit` for a short run. Built-in loaders skip missing final answers,
and the Numina loader keeps valid problems with a final answer. For the exact
curated research split or cross-source deduplication, supply your own JSONL.

### Local JSONL

Pass `--dataset data/train.jsonl` and put one object per line. SC imitation needs only
`question`. PPO also needs `answer`, the bare content expected inside the
generator's final box.

```json
{"question": "What is 2 + 3?", "answer": "5"}
```

## Install and train

```bash
python -m pip install -e .
```

### Controller configs

- `configs/small.json` uses width 256, 3 layers and 4 heads (4.75M parameters).
- `configs/large.json` uses width 512, 4 layers and 8 heads (23.13M parameters).

Both take 2,048-dimensional features from frozen Qwen3-1.7B. Pass a file
with `--config` when starting from scratch. Warm starts infer the architecture
from the checkpoint and check it against any explicit config.

### SC imitation

```bash
tts-controller-train imitate checkpoints/sc.pt \
  --dataset data/train.jsonl --config configs/small.json --epochs 2
```

`output` is the checkpoint to write. Start from random controller weights or pass `--init checkpoints/ckpt.pt` to
warm-start. Imitation uses masked cross-entropy on SC actions and encourages
uniform scores on finished nodes.

### PPO with terminal correctness

```bash
tts-controller-train ppo checkpoints/terminal.pt \
  --dataset numina --limit 200 --init checkpoints/sc.pt --reward terminal
```

`--init` is required for PPO. `--reward terminal` is the default.

### PPO with prefix shaping

```bash
tts-controller-train ppo checkpoints/prefix.pt \
  --dataset deepmath --limit 200 --init checkpoints/sc.pt --reward prefix \
  --potential checkpoints/phi.pt --shaping-weight 0.2
```

`--potential` requires a frozen `PrefixValue` checkpoint with `architecture`
and `state_dict` entries. `--shaping-weight` defaults to `0.2` and accepts
values in `(0, 1]`.

### Rewards

1. **Terminal correctness** (`--reward terminal`) uses
   `R_terminal = Σ_a p(a | tree) × 𝟙[a = gold]`.
   Here `p(a | tree)` is the weighted vote probability of a finished answer
   class. With no finished answer, the reward is zero.
2. **Prefix shaping** (`--reward prefix`) keeps `R_terminal` and gives each
   expansion `r_t = w × (Φ(v_new) − max_{v ∈ T_before} Φ(v))`.
   The frozen predictor `Φ` estimates the probability that a continuation is
   correct. The total reward is `R_terminal + Σ_t r_t`.

### Arguments shared by imitation and PPO

| Argument | Default | Meaning |
| --- | --- | --- |
| `--dataset` | Required | Hub name or local JSONL path |
| `--limit` | All eligible rows | Read at most this many training questions |
| `output` | Required | Destination for controller weights |
| `--config` | Small for new runs | Architecture JSON. Warm starts use the checkpoint config. |
| `--epochs` | `1` | Passes over the dataset |
| `--budget` | `8192` | Total generated tokens per tree |
| `--chunk` | `128` | Tokens per expansion, either `128` or `512` |
| `--device` | `cuda` if available, otherwise `cpu` | Device for the controller and generator |
| `--seed` | `7` | Seed for question order and sampling |
| `--lr` | `1e-4` | AdamW learning rate |

Each path has a fixed cap of 8,192 generated tokens. To see the CLI arguments,
run `tts-controller-train imitate --help` or `tts-controller-train ppo --help`.

## Checkpoints

Each epoch atomically saves the controller configuration and weights at
`output`. Use `--init` with a different output path to start another run from
saved weights. The optimizer starts fresh. Research checkpoints with
`contract.policy` also load.

```python
from src.model import load_controller

controller = load_controller("checkpoints/sc.pt", "cpu")
```

### Hugging Face checkpoints

The two training examples are in the private [controller weights repository](https://huggingface.co/rediska0123/tts-controller-weights/commit/e684aea2ff3f0e47638311cbba56f6ac53e1c610).
Sign in with `hf auth login` using a token with access to it.

- `sc-imitation.safetensors` is the small SC imitation checkpoint at epoch 40.
- `ppo-terminal-prefix.safetensors` is the small PPO checkpoint trained with
  terminal correctness and prefix shaping.
- `fork-choice-math40.safetensors` is the separate continuation-reward
  checkpoint used in the 40-question branching pilot.

```python
from src.model import load_controller

repo = "hf://rediska0123/tts-controller-weights@e684aea2ff3f0e47638311cbba56f6ac53e1c610"
sc = load_controller(f"{repo}/sc-imitation.safetensors", "cpu")
rl = load_controller(f"{repo}/ppo-terminal-prefix.safetensors", "cpu")
```

A Hub URI also works as `--init`. To publish your own checkpoint, sign in
with a write token and run

```bash
tts-controller-push checkpoints/sc.pt YOUR_USERNAME/tts-controller-weights \
  --filename sc-imitation.safetensors
```

The upload contains controller tensors and architecture metadata. Qwen
weights, optimizer state and training data stay out of the export.

The standalone runner uses Transformers for generation and grades normalized
boxed strings. Euler runs use vLLM and symbolic answer equivalence. Research
snapshots use Qwen hidden layer 14. Older standalone checkpoints retain their
original final-layer features.
