import os
import json
import torch
import models
import argparse
import dataloaders

from utils import losses
from utils import Logger
from trainer import Trainer

def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])

def main(config, resume):

    print("CONFIG", f"\n{config}\n")

    train_logger = Logger()

    # DATA LOADERS
    train_loader = get_instance(dataloaders, "train_loader", config)
    val_loader = get_instance(dataloaders, "val_loader", config)

    # MODEL
    model = get_instance(models, "arch", config, train_loader.dataset.num_classes)

    print("MODEL", f"\n{model}\n")

    # LOSS
    loss = get_instance(losses, "loss", config)

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger
    )

    trainer.train()

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")

    parser.add_argument("-c", "--config", default="config.json", type=str, help="Path to the config file (default: config.json)")
    parser.add_argument("-r", "--resume", default=None, type=str, help="Path to the .pth model checkpoint to resume training")
    parser.add_argument("-d", "--device", default=None, type=str, help="Indices of GPUs to enable (default: all)")
    
    args = parser.parse_args()
    config = json.load(open(args.config))

    print(f"Using config file {args.config}")

    if args.resume:
        config = torch.load(args.resume)["config"]
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    main(config, args.resume)
