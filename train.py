from comet_ml import Experiment

import os


import hydra
import torch, torch.nn as nn
from omegaconf import OmegaConf
from accelerate import DistributedDataParallelKwargs, Accelerator
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm


from model import aasist3
from datasets import print_fancy
from datasets import ASVspoof2019Dev, ASVspoof2019Train, ASVspoof5Dev, MAILABS, MLAAD, ASVspoof5Train
from utils import train_one_epoch


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(config):
    print_fancy(str(OmegaConf.to_container(config)))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config["find_unused_parameters"])

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs]
    )

    print_fancy("Accelerator loaded")

    asvspoof19train = ASVspoof2019Train(
        root_dir=config['data']["asvspoof2019_train"]["root_dir"],
        meta_path=config['data']["asvspoof2019_train"]["meta_path"],
    )
    asvspoof24train = ASVspoof5Train(
        root_dir=config['data']["asvspoof5_train"]["root_dir"],
        meta_path=config['data']["asvspoof5_train"]["meta_path"],
    )
    mlaad = MLAAD(
        root_dir=config['data']["mlaad"]["root_dir"],
    )
    mailabs = MAILABS(
        root_dir=config['data']["m_ailabs"]["root_dir"],
    )
    train_dataset = ConcatDataset([asvspoof19train, asvspoof24train, mlaad, mailabs])

    print_fancy("train datasets loaded")

    asvspoof5dev = ASVspoof5Dev(
        root_dir=config['data']["asvspoof5_dev"]["root_dir"],
        meta_path=config['data']['asvspoof5_dev']['meta_path']
    )

    asvspoof19dev = ASVspoof2019Dev(
        root_dir=config['data']['asvspoof2019_dev']["root_dir"],
        meta_path=config['data']['asvspoof2019_dev']['meta_path']
    )

    print_fancy('validation datasets loaded')

    train_dl = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True
    )

    asv19_dl = DataLoader(
        asvspoof19dev,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )

    asv5_dl = DataLoader(
        asvspoof5dev,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=False
    )

    print_fancy('dataloaders initialised')

    loss_fn = nn.CrossEntropyLoss()

    model = aasist3()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        eps=1e-7,
        weight_decay=0
    )

    train_dl, asv19_dl, asv5_dl, loss_fn, model, optimizer = accelerator.prepare(
        train_dl,
        asv19_dl,
        asv5_dl,
        loss_fn,
        model,
        optimizer
    )

    print_fancy("Important entities created")

    experiment = Experiment(
        api_key=os.environ.get("COMET_KEY", None),
        project_name=config.get("comet_project_name", "default"),
        workspace=config.get("comet_workspace", None),
        auto_output_logging="simple"
    )
    experiment.set_name(config.get("comet_run_name"))
    experiment.log_parameters(OmegaConf.to_container(config))
    print_fancy("Comet experiment initialized")

    for epoch in tqdm(range(config.get("num_epochs"))):
        current_loss = train_one_epoch(model, train_dl, loss_fn, optimizer, accelerator, max_batches=config.get("max_train_batches"))
        experiment.log_metric("avg_loss_per_epoch", current_loss)

    experiment.end()


if __name__ == "__main__":
    main()