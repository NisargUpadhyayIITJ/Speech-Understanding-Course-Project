import os

import hydra
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs, Accelerator
from safetensors.torch import save_file

from model import aasist3
from datasets import print_fancy


DEFAULT_EXPORT_PATH = "/data/home/borodin_sam/another_workspace/AASIST3/weights/train/FINAL/model.safetensors"


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config):
    print_fancy(str(OmegaConf.to_container(config)))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config["find_unused_parameters"])
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    model_config = OmegaConf.to_container(config.get("model"), resolve=True)
    encoder_config = OmegaConf.to_container(config.get("encoder"), resolve=True)
    graph_config = OmegaConf.to_container(config.get("graph"), resolve=True)
    projection_config = OmegaConf.to_container(config.get("projection"), resolve=True)

    if config.get("loss", {}).get("contrastive_weight", 0.0) > 0:
        projection_config["enabled"] = True

    model = aasist3(
        model_config=model_config,
        encoder_config=encoder_config,
        graph_config=graph_config,
        projection_config=projection_config,
    )
    model = accelerator.prepare(model)

    checkpoint_path = config.get("resume_from_checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        accelerator.print(f"Restoring checkpoint from {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        print_fancy("Checkpoint restored.")

    unwrapped_model = accelerator.unwrap_model(model)

    export_path = config.get("export", {}).get("weights_path", DEFAULT_EXPORT_PATH)
    export_dir = os.path.dirname(export_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    save_file(unwrapped_model.state_dict(), export_path)
    print_fancy(f"Saved safetensors weights to {export_path}", style="success")

    repo_id = config.get("export", {}).get("repo_id")
    if repo_id:
        unwrapped_model.push_to_hub(repo_id)
        print_fancy(f"Pushed model to {repo_id}", style="success")


if __name__ == "__main__":
    main()
