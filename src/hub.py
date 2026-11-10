"""Export and upload a controller-only checkpoint to Hugging Face."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi
from safetensors.torch import save_file

from .model import load_controller
from .rollout import MODEL, REVISION


def export_controller(checkpoint, output):
    controller = load_controller(checkpoint, "cpu")
    weights = {
        name: tensor.detach().contiguous()
        for name, tensor in controller.state_dict().items()
    }
    save_file(weights, output, metadata={
        "policy": json.dumps(controller.config, sort_keys=True),
        "hidden_layer": str(controller.hidden_layer),
        "model": MODEL,
        "revision": REVISION,
        "source_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    })


def private_repo(api, repo_id):
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="model")
    if not info.private:
        raise ValueError("Choose a private Hugging Face model repository")
    return info


def push_controller(checkpoint, repo_id, filename):
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("Use a .safetensors filename")
    with TemporaryDirectory() as temp:
        exported = Path(temp) / filename
        export_controller(checkpoint, exported)
        api = HfApi()
        info = private_repo(api, repo_id)
        return api.upload_file(
            path_or_fileobj=exported, path_in_repo=filename,
            repo_id=repo_id, repo_type="model", parent_commit=info.sha,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local controller checkpoint")
    parser.add_argument("repo", help="Private Hub model repository, e.g. owner/tts-controller-weights")
    parser.add_argument("--filename", default="controller.safetensors")
    args = parser.parse_args()
    print(push_controller(args.checkpoint, args.repo, args.filename).commit_url)


if __name__ == "__main__":
    main()
