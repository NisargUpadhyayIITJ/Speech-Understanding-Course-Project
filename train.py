import os
import logging
from accelerate.logging import get_logger

import hydra
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


from model import aasist3
from model.losses import build_training_objective
from datasets import ASVspoof2019Dev, ASVspoof2019Train, ASVspoof5Dev, MAILABS, MLAAD, ASVspoof5Train
from utils import train_one_epoch, compute_scores, compute_antispoofing_metrics


logger = get_logger(__name__)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # os.environ['NCCL_P2P_DISABLE'] = '1'
    # os.environ['NCCL_IB_DISABLE'] = '1'
    two_views = config.get("augmentation", {}).get("two_views", False)
    model_config = OmegaConf.to_container(config.get("model"), resolve=True)
    encoder_config = OmegaConf.to_container(config.get("encoder"), resolve=True)
    graph_config = OmegaConf.to_container(config.get("graph"), resolve=True)
    projection_config = OmegaConf.to_container(config.get("projection"), resolve=True)
    loss_config = OmegaConf.to_container(config.get("loss"), resolve=True)
    if loss_config.get("contrastive_weight", 0.0) > 0:
        projection_config["enabled"] = True
    if loss_config.get("name") in {"ce_supcon", "ce_infonce"} and not two_views:
        raise ValueError("Contrastive objectives require `augmentation.two_views=true`.")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config["find_unused_parameters"])

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="tensorboard",
        project_dir=config.get("tensorboard_dir", "./runs"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps")
    )

    logger.info(str(OmegaConf.to_container(config)), main_process_only=True)
    logger.info("Accelerator loaded", main_process_only=True)

    asvspoof19train = ASVspoof2019Train(
        root_dir=config['data']["asvspoof2019_train"]["root_dir"],
        meta_path=config['data']["asvspoof2019_train"]["meta_path"],
        two_views=two_views,
    )
    asvspoof24train = ASVspoof5Train(
        root_dir=config['data']["asvspoof5_train"]["root_dir"],
        meta_path=config['data']["asvspoof5_train"]["meta_path"],
        two_views=two_views,
    )
    mlaad = MLAAD(
        root_dir=config['data']["mlaad"]["root_dir"],
        two_views=two_views,
    )
    mailabs = MAILABS(
        root_dir=config['data']["m_ailabs"]["root_dir"],
        two_views=two_views,
    )
    train_dataset = ConcatDataset([asvspoof19train, asvspoof24train, mlaad, mailabs])

    logger.info("train datasets loaded", main_process_only=True)

    asvspoof5dev = ASVspoof5Dev(
        root_dir=config['data']["asvspoof5_dev"]["root_dir"],
        meta_path=config['data']['asvspoof5_dev']['meta_path']
    )

    asvspoof19dev = ASVspoof2019Dev(
        root_dir=config['data']['asvspoof2019_dev']["root_dir"],
        meta_path=config['data']['asvspoof2019_dev']['meta_path']
    )

    logger.info('validation datasets loaded', main_process_only=True)

    train_dl = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    asv19_dl = DataLoader(
        asvspoof19dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    asv5_dl = DataLoader(
        asvspoof5dev,
        batch_size=config['val_batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    logger.info('dataloaders initialised', main_process_only=True)

    objective = build_training_objective(loss_config)
    model = aasist3(
        model_config=model_config,
        encoder_config=encoder_config,
        graph_config=graph_config,
        projection_config=projection_config,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=1e-7,
        weight_decay=0
    )

    train_dl, asv19_dl, asv5_dl, model, optimizer = accelerator.prepare(
        train_dl,
        asv19_dl,
        asv5_dl,
        model,
        optimizer
    )

    logger.info("Important entities created", main_process_only=True)

    # Initialize TensorBoard experiment through Accelerator
    # TensorBoard's add_hparams only accepts flat primitives (int/float/str/bool),
    # so filter out any nested dicts/lists from the Hydra config.
    flat_config = {
        k: v for k, v in OmegaConf.to_container(config, resolve=True).items()
        if isinstance(v, (int, float, str, bool))
    }
    accelerator.init_trackers(
        project_name=config.get("project_name", "antispoof"),
        config=flat_config,
    )
    logger.info("TensorBoard tracker initialized", main_process_only=True)

    checkpoint_path = config.get("checkpoint_path", -1)
    resume_epoch = 0
    if config.get("resume_from_checkpoint"):
        checkpoint_path = config.get("resume_from_checkpoint")
        if os.path.exists(checkpoint_path):
            model_weights_before = {name: param.clone().detach() for name, param in model.named_parameters()}
            logger.info(f"Restoring checkpoint from {checkpoint_path}", main_process_only=True)
            accelerator.load_state(checkpoint_path)

            weights_changed = False
            for name, param in model.named_parameters():
                if not torch.equal(model_weights_before[name], param):
                    weights_changed = True
                    logger.info(f"Weights changed for parameter: {name}", main_process_only=True)
                    break

            if weights_changed:
                logger.info(" Model weights successfully loaded from checkpoint", main_process_only=True)
            else:
                logger.warning(" Warning: Model weights did not change after loading checkpoint", main_process_only=True)

    logger.info("Model restorated.", main_process_only=True)

    run_name = config.get("run_name", "run")
    plots_dir = os.path.join(config.get("checkpoint_base_path"), "plots", run_name)
    if accelerator.is_main_process:
        os.makedirs(plots_dir, exist_ok=True)

    # History buffers for plotting
    history = {
        "epoch": [],
        "train_loss": [],
        "asv19_eer": [],
        "asv19_dcf": [],
        "asv5_eer": [],
        "asv5_dcf": [],
    }

    checkpoint_interval = config.get("checkpoint_interval", 5)

    for epoch in tqdm(range(resume_epoch, config.get("num_epochs"))):
        current_loss, epoch_train_metrics = train_one_epoch(
            model,
            train_dl,
            objective,
            optimizer,
            accelerator,
            max_batches=config.get("max_train_batches"),
        )
        accelerator.log({"avg_loss_per_epoch": current_loss, **epoch_train_metrics}, step=epoch)

        asv19_scores, asv19_labels = compute_scores(asv19_dl, model, accelerator, max_batches=config.get("max_val_batches"))
        asv19dcf, asv19_eer, asv19_cllr = compute_antispoofing_metrics(asv19_scores, asv19_labels)
        accelerator.log({
            "asv19_dev_dcf": asv19dcf,
            "asv19_dev_eer": asv19_eer,
            "asv19_dev_cllr": asv19_cllr
        }, step=epoch)
        logger.info(f"asv19 eer: {asv19_eer},\nasv19 dcf: {asv19dcf}", main_process_only=True)

        asv5_scores, asv5_labels = compute_scores(asv5_dl, model, accelerator, max_batches=config.get("max_val_batches"))
        asv5dcf, asv5_eer, asv5_cllr = compute_antispoofing_metrics(asv5_scores, asv5_labels)
        accelerator.log({
            "asv5_dev_dcf": asv5dcf,
            "asv5_dev_eer": asv5_eer,
            "asv5_dev_cllr": asv5_cllr
        }, step=epoch)
        logger.info(f"asv5 eer: {asv5_eer},\nasv5 dcf: {asv5dcf}", main_process_only=True)

        # ── Accumulate history ──────────────────────────────────────────────
        history["epoch"].append(epoch)
        history["train_loss"].append(current_loss)
        history["asv19_eer"].append(asv19_eer)
        history["asv19_dcf"].append(asv19dcf)
        history["asv5_eer"].append(asv5_eer)
        history["asv5_dcf"].append(asv5dcf)

        # ── Plot curves (main process only) ────────────────────────────────
        if accelerator.is_main_process:
            epochs_so_far = history["epoch"]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"[{run_name}]  epoch {epoch + 1}", fontsize=13, fontweight="bold")

            axes[0].plot(epochs_so_far, history["train_loss"], marker="o", color="steelblue", label="train loss")
            axes[0].set_title("Train Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, linestyle="--", alpha=0.5)

            axes[1].plot(epochs_so_far, history["asv19_eer"], marker="o", color="tomato", label="ASV2019 EER")
            axes[1].plot(epochs_so_far, history["asv19_dcf"], marker="s", color="coral", linestyle="--", label="ASV2019 DCF")
            axes[1].set_title("ASVspoof2019 Val")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Metric")
            axes[1].legend()
            axes[1].grid(True, linestyle="--", alpha=0.5)

            axes[2].plot(epochs_so_far, history["asv5_eer"], marker="o", color="mediumseagreen", label="ASV5 EER")
            axes[2].plot(epochs_so_far, history["asv5_dcf"], marker="s", color="seagreen", linestyle="--", label="ASV5 DCF")
            axes[2].set_title("ASVspoof5 Val")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Metric")
            axes[2].legend()
            axes[2].grid(True, linestyle="--", alpha=0.5)

            plt.tight_layout()
            plot_path = os.path.join(plots_dir, f"epoch_{epoch:03d}.png")
            plt.savefig(plot_path, dpi=120)
            plt.close(fig)
            logger.info(f"Saved plot → {plot_path}", main_process_only=True)

        # ── Checkpoint every N epochs ──────────────────────────────────────
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_name = f"{run_name}_epoch_{epoch}"
            checkpoint_path = os.path.join(config.get("checkpoint_base_path"), checkpoint_name)
            accelerator.save_state(checkpoint_path)
            logger.info(f"Checkpoint saved → {checkpoint_path}", main_process_only=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
