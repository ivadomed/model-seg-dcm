import argparse
import os
from tqdm import tqdm
from loguru import logger
import yaml
import sys
from pathlib import Path
from textwrap import dedent
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

from loss import DiceCrossEntropyLoss
from lr_scheduler import LinearWarmupCosineAnnealingLR
from loader import load_data
from utils import dice_score

from models.backbone import BackboneModel
from utils import SmartFormatter

torch.multiprocessing.set_sharing_strategy('file_system')
local_rank = int(os.environ["LOCAL_RANK"])

def get_parser():

    parser = argparse.ArgumentParser(description="Supervised Pretraining on spinal cord T2w MRI data",
                                     formatter_class=SmartFormatter)

    parser.add_argument("--path-data", required=True, type=str,
                        help="Path to the folder containing datalist(s) for each dataset.")
    parser.add_argument("--datalists", nargs="+", type=str, default=None,
                        help="List of JSON datalist(s) for each dataset. If not provided (None), all datalists in the "
                             "'--path-data' folder will be used. Default: None.")
    parser.add_argument("--path-out", type=str, required=True,
                        help="Path to the output directory. The model and the log will be saved here.")
    parser.add_argument('-m', '--model', choices=['nnunet', 'monai-unet', 'swinunetr'], 
                        default='nnunet', type=str, 
                        help=f"Model to be used for pretraining.")
    parser.add_argument("--config", type=str, required=True,
                        help="R|Path to the YAML config file containing all training details.\n"
                             "An example of the config YAML file:\n"
                             + dedent(
                                """
                                train_batch_size: 16
                                val_batch_size: 16
                                preprocessing:
                                  crop_pad_size: [64, 192, 320]
                                  patch_size: [64, 64, 64]
                                seed: 42
                                model:
                                  swinunetr:
                                    in_channels: 1
                                    out_channels: 1
                                opt:
                                  name: adamw
                                  lr: 0.0004
                                  batch_size: 16
                                  warmup_epochs: 10
                                  max_epochs: 500
                                  check_val_every_n_epochs: 2\n
                                """)
                        )
    parser.add_argument("-rfc", "--resume-from-checkpoint", action="store_true",
                        help="Resume training from checkpoint.")
    parser.add_argument("--run-dir", help="Location of model to resume.")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--dist", action="store_true", default=False,
                        help="Use distributed training")

    return parser


def train_one_epoch(train_loader, model, optimizer, scheduler, epoch, loss_function, scaler, writer, device):
    # set in train mode
    model.train()
    epoch_loss_train, epoch_soft_dice_train = 0, 0
    
    epoch_iterator = tqdm(enumerate(train_loader), total=len(train_loader))

    if isinstance(model, DDP):
        # NOTE: if the model is wrapped around DDP, then it has to be unwrapped to access the model's attributes
        # https://github.com/huggingface/transformers/issues/18974 
        model = model.module

    for step, batch_data in epoch_iterator:

        x, y = batch_data["image"].to(device), batch_data["label_sc"].to(device)
        
        optimizer.zero_grad()
        with autocast(enabled=True):
            logits = model(x)

            if model.model_name in ["nnunet"]:

                loss, train_soft_dice = 0.0, 0.0
                for i in range(len(logits)):
                    # give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                    # this gives higher resolution outputs more weight in the loss
                    # NOTE: outputs[0] is the final pred, outputs[-1] is the lowest resolution pred (at the bottleneck)
                    # we're downsampling the GT to the resolution of each deepsupervision feature map output 
                    # (instead of upsampling each deepsupervision feature map output to the final resolution)
                    downsampled_gt = F.interpolate(y, size=logits[i].shape[-3:], mode='trilinear', align_corners=False)
                    # print(f"downsampled_gt.shape: {downsampled_gt.shape} \t output[i].shape: {output[i].shape}")
                    loss += (0.5 ** i) * loss_function(logits[i], downsampled_gt)

                    # get probabilities from logits
                    out = F.relu(logits[i]) / F.relu(logits[i]).max() if bool(F.relu(logits[i]).max()) else F.relu(logits[i])

                    # calculate train dice
                    train_soft_dice += dice_score(out, downsampled_gt) 
                
                # average dice loss across the outputs
                loss /= len(logits)
                train_soft_dice /= len(logits)

            else:
                # calculate training loss   
                loss = loss_function(logits, y)

                # get probabilities from logits
                y_hat = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)

                # calculate train dice
                train_soft_dice = dice_score(y_hat, y)

            epoch_loss_train += loss.item()
            epoch_soft_dice_train += train_soft_dice.detach().cpu()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    writer.add_scalar("train/loss", scalar_value=epoch_loss_train/len(train_loader), global_step=epoch)
    writer.add_scalar("train/dice", scalar_value=epoch_soft_dice_train/len(train_loader), global_step=epoch)

    return epoch_loss_train/len(train_loader)


@torch.no_grad()    # this decorator disables gradient tracking
def evaluate(val_loader, model, loss_function, writer, epoch, device):

    # set in eval mode
    model.eval()
    epoch_loss_val, epoch_soft_dice_val, epoch_hard_dice_val = 0, 0, 0

    if isinstance(model, DDP):
        # NOTE: if the model is wrapped around DDP, then it has to be unwrapped to access the model's attributes
        model = model.module

            
        x, y = x["image"].to(device), x["label_sc"].to(device)

        # TODO: use sliding window inference here. what is this below ?!

        with autocast(enabled=True):
            
            logits = sliding_window_inference(inputs=x, roi_size=x.shape[-3:], sw_batch_size=4, 
                                              predictor=model, overlap=0.5, mode="gaussian")
            
            if model.model_name == "nnunet":
                logits = logits[0]  # take only the highest resolution output

            y_hat = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)
            loss = loss_function(y_hat, y)
            val_dice_soft = dice_score(y_hat, y)
            val_dice_hard = dice_score((y_hat > 0.5).float(), (y > 0.5).float())
    
            epoch_loss_val += loss.item()
            epoch_soft_dice_val += val_dice_soft
            epoch_hard_dice_val += val_dice_hard
    
    writer.add_scalar("val/loss", scalar_value=epoch_loss_val/len(val_loader), global_step=epoch)
    writer.add_scalar("val/dice_soft", scalar_value=epoch_soft_dice_val/len(val_loader), global_step=epoch)
    writer.add_scalar("val/dice_hard", scalar_value=epoch_hard_dice_val/len(val_loader), global_step=epoch)

    return epoch_loss_val/len(val_loader)


def run_training(model, train_loader, val_loader, n_epochs, optimizer, scheduler, loss_function, 
                 writer_train, writer_val, best_loss, eval_freq, device, log_dir, use_dist=False):
    
    scaler = GradScaler()
    
    # validation sanity check
    val_loss = evaluate(val_loader, model, loss_function, writer_val, epoch=0, device=device)
    logger.info(f"Epoch 0 --> Validation Loss: {val_loss:.3f}") if local_rank == 0 else None

    for epoch in range(n_epochs):
        train_loss = train_one_epoch(train_loader, model, optimizer, scheduler, epoch, loss_function, 
                                     scaler, writer_train, device)
        logger.info(f"Epoch {epoch+1}/{n_epochs} --> Training Loss: {train_loss:.3f}") if local_rank == 0 else None

        if (epoch + 1) % eval_freq == 0:
            val_loss = evaluate(val_loader, model, loss_function, writer_val, epoch, device)
            logger.info(f"Epoch {epoch+1} --> Validation Loss: {val_loss:.3f}") if local_rank == 0 else None
        
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
    if local_rank != 0:
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
    logger.info(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True

    # load config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"Loaded config file: {args.config}") if local_rank == 0 else None

    # for reproducibility purposes set a seed
    set_determinism(config["seed"])

    # No datalists manually provided --> load all datalists in the folder
    if args.datalists is None:
        datalists_list = [f for f in os.listdir(args.path_data) if f.endswith(".json")]
    else:
        datalists_list = args.datalists
    logger.info(f"The following datalists will be used: {datalists_list}") if local_rank == 0 else None

    # Get absolute path to the datalists
    datalists_list = [os.path.join(args.path_data, f) for f in datalists_list]

    logger.info("Getting data...") if local_rank == 0 else None
    train_loader, val_loader = load_data(
        datalists_paths=datalists_list,
        train_batch_size=config["train_batch_size"],
        val_batch_size=config["val_batch_size"],
        num_workers=8,
        use_distributed=False,
        crop_size=config["preprocessing"]["crop_pad_size"],
        patch_size=config["preprocessing"]["patch_size"],
        device=local_rank,
        task="pretraining"
    )

    # model
    logger.info("Building model...") if local_rank == 0 else None
    model = BackboneModel(model_name=args.model, config=config)
    model.run_folder = f"{model.run_folder}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    log_dir = log_dir / model.run_folder
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(log_dir / "models", exist_ok=True)  # to save model checkpoints
    
    if args.dist:
        logger.info("Wrapping the model with Distributed Data Parallel ...")
        # the model also has to be moved to the local rank; https://github.com/pytorch/pytorch/issues/46821
        model = DDP(model.to(local_rank), device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.to(local_rank)

    # loss function
    logger.info("Defining loss function...") if local_rank == 0 else None
    loss_function = DiceCrossEntropyLoss(weight_ce=1.0, weight_dice=1.0)

    # optimizers
    logger.info("Setting up the optimizer...") if local_rank == 0 else None
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
    writer_train = SummaryWriter(str(log_dir / "train_logs"))
    writer_val = SummaryWriter(str(log_dir / "val_logs"))

    # Get Checkpoint
    best_loss = float("inf")
    if args.resume_from_checkpoint:
        logger.info(f"Using existing checkpoint ...") if local_rank == 0 else None
        checkpoint = torch.load(str(log_dir / "models" / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        logger.info(f"No checkpoint found.") if local_rank == 0 else None

    n_epochs = config["opt"]["max_epochs"]
    eval_freq = config["opt"]["check_val_every_n_epochs"]
    # run training
    logger.info("Running 'Supervised' Pre-training ...") if local_rank == 0 else None
    run_training(model, train_loader, val_loader, n_epochs, optimizer, scheduler, loss_function,
                    writer_train, writer_val, best_loss, eval_freq, local_rank, log_dir, args.dist)

    # destroy the process group explicitly
    dist.destroy_process_group()


if __name__ == "__main__":
    args = get_parser().parse_args()
    # NOTE: The error "Cannot re-initialize CUDA in forked subprocess" can be resolved by setting the start method to 'spawn'
    # https://github.com/pytorch/pytorch/issues/40403#issuecomment-648439409
    torch.multiprocessing.set_start_method('spawn')
    main_worker(args) 
