import argparse
import os
from tqdm import tqdm
from loguru import logger
import yaml
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism

from loss import DiceCrossEntropyLoss
from lr_scheduler import LinearWarmupCosineAnnealingLR
from loader import load_data

from models.backbone import BackboneModel


def get_parser():

    parser = argparse.ArgumentParser(description="Supervised Pretraining on spinal cord T2w MRI data")

    parser.add_argument("--path-data", required=True, type=str,
                        help="Path to the folder containing datalist(s) for each dataset.")
    parser.add_argument("--datalists", nargs="+", type=str, default=None,
                        help="List of JSON datalist(s) for each dataset. If not provided (None), all datalists in the "
                             "'--path-data' folder will be used. Default: None.")
    parser.add_argument("--path-out", type=str, required=True,
                        help="Path to the output directory. The model and the log will be saved here.")
    parser.add_argument('-m', '--model', choices=['nnunet', 'monai-unet', 'unetr', 'swinunetr'], 
                        default='nnunet', type=str, 
                        help=f"Model to be used for pretraining.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML config file containing all training details.")
    parser.add_argument("-rfc", "--resume-from-checkpoint", action="store_true",
                        help="Resume training from checkpoint.")
    parser.add_argument("--run-dir", help="Location of model to resume.")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dist", action="store_true", default=False,
                        help="Use distributed training")
    parser.add_argument("--local-rank", type=int, default=0)

    return parser


def train_one_epoch(train_loader, model, optimizer, scheduler, epoch, loss_function, scaler, writer, device):
    # set in train mode
    model.train()
    epoch_loss_train = 0
    
    epoch_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, x in enumerate(epoch_iterator):

        x, y = x["image"].to(device), x["label_sc"].to(device)
        
        optimizer.zero_grad()
        with autocast(enabled=True):
            logits = model(x)
            # get probabilities from logits
            y_hat = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)
            loss = loss_function(y_hat, y)

            epoch_loss_train += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    writer.add_scalar("train/loss", scalar_value=epoch_loss_train/len(train_loader), global_step=epoch)

    return epoch_loss_train/len(train_loader)

@torch.no_grad()
def evaluate(val_loader, model, loss_function, writer, epoch, device):

    # set in eval mode
    model.eval()
    epoch_loss_val = 0

    for step, x in enumerate(val_loader):
            
        x, y = x["image"].to(device), x["label_sc"].to(device)

        with autocast(enabled=True):
            
            logits = model(x)
            y_hat = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)
            loss = loss_function(y_hat, y)
    
            epoch_loss_val += loss.item()
    
    writer.add_scalar("val/loss", scalar_value=epoch_loss_val/len(val_loader), global_step=epoch)

    return epoch_loss_val/len(val_loader)


def run_training(model, train_loader, val_loader, n_epochs, optimizer, scheduler, loss_function, 
                 writer_train, writer_val, best_loss, eval_freq, device, log_dir, use_dist=False):
    
    scaler = GradScaler()
    
    # validation sanity check
    val_loss = evaluate(val_loader, model, loss_function, writer_val, start_epoch=0, device=device)
    logger.info(f"Epoch 0 --> Validation Loss: {val_loss:.3f}")

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, epoch, loss_function, 
                                     scaler, writer_train, device)
        logger.info(f"Epoch {epoch+1}/{n_epochs} --> Training Loss: {train_loss:.3f}")

        if (epoch + 1) % eval_freq == 0:
            val_loss = evaluate(val_loader, model, loss_function, writer_val, epoch, device)
            logger.info(f"Epoch {epoch+1} --> Validation Loss: {val_loss:.3f}")
        
        if val_loss < best_loss:
            best_loss = val_loss

            checkpoint = {
                "epoch": epoch+1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss
                }
            if use_dist:
                if dist.get_rank() == 0:
                    torch.save(checkpoint, str(log_dir / "models" / "checkpoint.pth"))
                    torch.save(model.state_dict(), str(log_dir / "models" / "best_model.pth"))
            else:
                torch.save(checkpoint, str(log_dir / "models" / "checkpoint.pth"))
                torch.save(model.state_dict(), str(log_dir / "models" / "best_model.pth"))

    logger.info("Training completed !")


def main_worker(args):

    # save output to a log file
    log_dir = Path(args.path_out)
    logger.add(str(log_dir / "log.txt"), rotation="10 MB", level="INFO")

    # disable logging for processes except 0 on every node
    if args.local_rank != 0:
        f = open(os.devnull, "w")
        sys.stdout = sys.stderr = f

    if args.dist:
        # initialize the distributed training process, every GPU runs in a process
        # strongly recommended to use ``init_method=env://`` with NCCL backend
        dist.init_process_group(backend="nccl", init_method="env://")
        logger.info(f"Training in distributed mode with multiple processes, 1 GPU per process."
                    f"Process {torch.distributed.get_rank()}, Total {torch.distributed.get_world_size()}.")
    else:
        logger.info("Training with a single process on 1 GPU.")

    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # load config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # for reproducibility purposes set a seed
    set_determinism(config["autoencoderkl"]["seed"])

    # No datalists manually provided --> load all datalists in the folder
    if args.datalists is None:
        datalists_list = [f for f in os.listdir(args.path_data) if f.endswith(".json")]
    else:
        datalists_list = args.datalists
    logger.info(f"The following datalists will be used: {datalists_list}")

    # Get absolute path to the datalists
    datalists_list = [os.path.join(args.path_data, f) for f in datalists_list]

    logger.info("Getting data...")
    train_loader, val_loader = load_data(
        datalists_paths=datalists_list,
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=8,
        use_distributed=True,
        crop_size=config["preprocessing"]["crop_pad_size"],
        patch_size=config["preprocessing"]["patch_size"],
        device=device,
        task="pretraining"
    )

    # model
    logger.info("Building model...")
    model = BackboneModel(model_name=args.model, config=config)
    run_folder = model.run_folder

    run_folder = f"{run_folder}_datetime.now().strftime('%Y%m%d-%H%M')"
    
    if args.dist:
        logger.info("Wrapping the model with Distributed Data Parallel ...")
        model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    else:
        model = model.to(device)

    # loss function
    logger.info("Defining loss function...")
    loss_function = DiceCrossEntropyLoss(weight_ce=1.0, weight_dice=1.0)

    # optimizers
    logger.info("Setting up the optimizer...")
    if config["opt"]["name"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["opt"]["lr"], fused=True)
    elif config["opt"]["name"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["opt"]["lr"], fused=True)
    elif config["opt"]["name"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["opt"]["lr"], momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {config['opt']['name']}")
    
    # Scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config["opt"]["warmup_epochs"], 
                                              max_epochs=config["opt"]["max_epochs"])

    # Tensorboard writers
    if args.local_rank == 0:
        writer_train = SummaryWriter(str(log_dir / "train_logs"))
        writer_val = SummaryWriter(str(log_dir / "val_logs"))

    # Get Checkpoint
    best_loss = float("inf")
    if args.resume_from_checkpoint:
        logger.info(f"Using existing checkpoint ...")
        checkpoint = torch.load(str(log_dir / "models" / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        logger.info(f"No checkpoint found.")

    n_epochs = config["opt"]["max_epochs"]
    eval_freq = config["opt"]["check_val_every_n_epochs"]
    # run training
    logger.info("Running 'Supervised' Pre-training ...")
    run_training(model, train_loader, val_loader, n_epochs, optimizer, scheduler, loss_function,
                    writer_train, writer_val, best_loss, eval_freq, device, log_dir, args.dist)

    # destroy the process group explicitly
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_parser().parse_args()
    # run = setup_wandb_run(args)
    main_worker(args, wandb_run=None)
