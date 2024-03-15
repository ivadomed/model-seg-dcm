"""
Self-Supervised Pre-training using Vision Transformer (ViT)

This script is based on this MONAI tutorial:
https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining/vit_unetr_ssl

Author: Jan Valosek
"""

import os
import time
import torch
import argparse
import matplotlib.pyplot as plt

from loguru import logger
from torch.nn import L1Loss
from monai.utils import set_determinism, first
from monai.networks.nets import ViTAutoEnc
from monai.losses import ContrastiveLoss
from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
)

from transforms import define_pretrain_transforms
from load_data import load_data


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run Self-Supervised Pre-training.')
    parser.add_argument('--dataset-split', required=True, type=str,
                        help='Path to the JSON file with training/validation split. '
                             'If paths are absolute, you do NOT need to use --data. '
                             'If only filenames are provided, you need to use --data to specify the root directory '
                             'of the dataset.')
    parser.add_argument('--data', required=False, type=str, default="",
                        help='Path to the dataset root directory. If not provided, path to data specified in the JSON '
                             'file will be used.')
    parser.add_argument('--logdir', required=True, type=str,
                        help='Path to the directory for logging.')
    parser.add_argument('--cuda', type=int, default=0, help='Index of the CUDA device to use.')

    return parser


def main():

    parser = get_parser()
    args = parser.parse_args()

    # -----------------------------------------------------
    # Define file paths & output directory path
    # -----------------------------------------------------
    json_path = os.path.abspath(args.dataset_split)
    data_root = os.path.abspath(args.data)
    logdir_path = os.path.abspath(args.logdir)

    # -----------------------------------------------------
    # Create result logging directories, manage data paths & set determinism
    # -----------------------------------------------------
    train_list, val_list = load_data(data_root, json_path, logdir_path)

    # save output to a log file
    logger.add(os.path.join(logdir_path, "log.txt"), rotation="10 MB", level="INFO")

    logger.info("Total training data are {} and validation data are {}".format(len(train_list), len(val_list)))

    # Set Determinism
    set_determinism(seed=123)

    # -----------------------------------------------------
    # Define MONAI Transforms
    # -----------------------------------------------------
    SPATIAL_SIZE = (64, 256, 256)
    ROI_SIZE = (64, 64, 64)
    #ROI_SIZE = SPATIAL_SIZE
    keys = ["image", "label"]

    transforms = define_pretrain_transforms(keys=keys, spatial_size=SPATIAL_SIZE, roi_size=ROI_SIZE)

    # -----------------------------------------------------
    # Sanity Check for the transforms
    # -----------------------------------------------------
    check_ds = Dataset(data=train_list, transform=transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    logger.info(f'original image shape: {check_data["gt_image"][0][0].shape}')
    logger.info(f'augmented image 1 shape: {check_data["image"][0][0].shape}')
    logger.info(f'augmented image 2 shape: {check_data["image_2"][0][0].shape}')

    # -----------------------------------------------------
    # Training Config
    # -----------------------------------------------------

    # Define Network ViT backbone & Loss & Optimizer
    patch_size = (16, 16, 16)
    device = torch.device(f"cuda:{args.cuda}")
    model = ViTAutoEnc(
        in_channels=1,
        img_size=ROI_SIZE,
        patch_size=patch_size,
        pos_embed="conv",
        hidden_size=768,
        mlp_dim=3072,
    )

    model = model.to(device)

    # Define Hyper-paramters for training loop
    max_epochs = 500
    val_interval = 2
    batch_size = 4
    lr = 1e-4
    epoch_loss_values = []
    step_loss_values = []
    epoch_cl_loss_values = []
    epoch_recon_loss_values = []
    val_loss_values = []
    best_val_loss = 1000.0

    recon_loss = L1Loss()
    contrastive_loss = ContrastiveLoss(temperature=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------
    # Print hyper-parameters into the log file
    # ------------------------------------------------------
    logger.info(f"img_size: {ROI_SIZE}")
    logger.info(f"patch_size: {patch_size}")
    logger.info(f"max_epochs: {max_epochs}")
    logger.info(f"val_interval: {val_interval}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"lr: {lr}")

    # -----------------------------------------------------
    # Create dataloaders for training
    # -----------------------------------------------------

    NUM_WORKERS = 0

    train_dataset = CacheDataset(data=train_list, transform=transforms, cache_rate=0.5, num_workers=NUM_WORKERS)
    val_dataset = CacheDataset(data=val_list, transform=transforms, cache_rate=0.25, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=False,
                              persistent_workers=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=False,
                            persistent_workers=False)

    # -----------------------------------------------------
    # Training Loop with Validation
    # -----------------------------------------------------

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        epoch_cl_loss = 0
        epoch_recon_loss = 0
        step = 0

        for batch_data in train_loader:
            step += 1
            start_time = time.time()

            inputs, inputs_2, gt_input = (
                batch_data["image"].to(device),
                batch_data["image_2"].to(device),
                batch_data["gt_image"].to(device),
            )
            optimizer.zero_grad()
            outputs_v1, hidden_v1 = model(inputs)
            outputs_v2, hidden_v2 = model(inputs_2)

            flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
            flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

            r_loss = recon_loss(outputs_v1, gt_input)
            cl_loss = contrastive_loss(flat_out_v1, flat_out_v2)

            # Adjust the CL loss by Recon Loss
            total_loss = r_loss + cl_loss * r_loss
            # TODO: verify if the detach is necessary
            total_loss.detach().cpu()

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            step_loss_values.append(total_loss.item())

            # CL & Recon Loss Storage of Value
            epoch_cl_loss += cl_loss.item()
            epoch_recon_loss += r_loss.item()

            end_time = time.time()
            logger.info(
                f"{step}/{len(train_dataset) // train_loader.batch_size}, "
                f"train_loss: {total_loss.item():.4f}, "
                f"time taken: {end_time-start_time}s"
            )

        epoch_loss /= step
        epoch_cl_loss /= step
        epoch_recon_loss /= step

        epoch_loss_values.append(epoch_loss)
        epoch_cl_loss_values.append(epoch_cl_loss)
        epoch_recon_loss_values.append(epoch_recon_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        end_time_epoch = time.time()
        logger.info(f"epoch {epoch + 1} time taken: {end_time_epoch-start_time_epoch}s")

        # Validation every 2 epochs
        if epoch % val_interval == 0:
            logger.info("Entering Validation for epoch: {}".format(epoch + 1))
            total_val_loss = 0
            val_step = 0
            model.eval()
            for val_batch in val_loader:
                val_step += 1
                start_time = time.time()
                inputs, gt_input = (
                    val_batch["image"].to(device),
                    val_batch["gt_image"].to(device),
                )
                logger.info("Input shape: {}".format(inputs.shape))
                # Call forward pass of the model with the inputs
                # The model returns the output and the hidden states
                #   1. outputs:  This is the reconstructed image from the model. It should have the same dimensions as
                # the input image.
                #   2. outputs_v2: This is the hidden representation of the input image, which is the output of the
                # transformer encoder. It's a high-dimensional feature representation of the input.
                outputs, outputs_v2 = model(inputs)
                val_loss = recon_loss(outputs, gt_input)
                total_val_loss += val_loss.item()
                end_time = time.time()

                if val_step == 1:
                    # Plot and save input and output validation images to see how the model is learning
                    plt.figure(1, figsize=(8, 8))
                    plt.subplot(2, 2, 1)
                    plt.imshow(inputs[0, 0, :, :, 32].detach().cpu().numpy(), cmap="gray")
                    plt.title("Input Image")
                    plt.subplot(2, 2, 2)
                    plt.imshow(gt_input[0, 0, :, :, 32].detach().cpu().numpy(), cmap="gray")
                    plt.title("Ground Truth Image")
                    plt.subplot(2, 2, 3)
                    plt.imshow(outputs[0, 0, :, :, 32].detach().cpu().numpy(), cmap="gray")
                    plt.title("Output Image")
                    # Include the epoch number as master title
                    plt.suptitle(f"Epoch {epoch + 1}")
                    fname = os.path.join(logdir_path, f"epoch_{epoch + 1}_val_{val_step}_images.png")
                    plt.savefig(fname)
                    plt.close(1)
                    logger.info(f"Saved validation images to {fname}")

            total_val_loss /= val_step
            val_loss_values.append(total_val_loss)
            logger.info(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

            if total_val_loss < best_val_loss:
                logger.info(f"Saving new model based on validation loss {total_val_loss:.4f}")
                best_val_loss = total_val_loss
                checkpoint = {"epoch": max_epochs, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
                torch.save(checkpoint, os.path.join(logdir_path, "best_model.pt"))

            plt.figure(1, figsize=(8, 8))
            plt.subplot(2, 2, 1)
            plt.plot(epoch_loss_values)
            plt.grid()
            plt.title("Training Loss")

            plt.subplot(2, 2, 2)
            plt.plot(val_loss_values)
            plt.grid()
            plt.title("Validation Loss")

            plt.subplot(2, 2, 3)
            plt.plot(epoch_cl_loss_values)
            plt.grid()
            plt.title("Training Contrastive Loss")

            plt.subplot(2, 2, 4)
            plt.plot(epoch_recon_loss_values)
            plt.grid()
            plt.title("Training Recon Loss")

            plt.savefig(os.path.join(logdir_path, "loss_plots.png"))
            plt.close(1)

    logger.info("Done")


if __name__ == "__main__":
    main()
