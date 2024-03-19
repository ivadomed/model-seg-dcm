"""
Finetuning of the 3D Single-Class Spinal Cord Lesion Segmentation Model Using SSL Pre-trained Weights

This script is based on this MONAI tutorial:
https://github.com/Project-MONAI/tutorials/tree/main/self_supervised_pretraining/vit_unetr_ssl

Author: Jan Valosek
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from loguru import logger
from monai.utils import set_determinism, first
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from monai.transforms import AsDiscrete

from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    Dataset,
    DataLoader,
    CacheDataset,
    decollate_batch,
)

from load_data import load_data
from transforms import define_finetune_train_transforms, define_finetune_val_transforms

# Added this to solve problem with too many files open allowing number of workers > 0
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
# https://github.com/ivadomed/model-seg-dcm/issues/8
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def get_parser():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Run Fine-tuning.')
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
    parser.add_argument('--pretrained-model', required=True, type=str,
                        help='Path to the pretrained model.')
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
    pretrained_model_path = os.path.abspath(args.pretrained_model)
    use_pretrained = True if pretrained_model_path is not None else False

    # -----------------------------------------------------
    # Create result logging directories, manage data paths & set determinism
    # -----------------------------------------------------
    train_list, val_list = load_data(data_root, json_path, logdir_path, is_segmentation=True)

    # save output to a log file
    logger.add(os.path.join(logdir_path, "log.txt"), rotation="10 MB", level="INFO")

    logger.info("Total training data are {} and validation data are {}".format(len(train_list), len(val_list)))

    # Set Determinism
    set_determinism(seed=123)

    # -----------------------------------------------------
    # Define MONAI Transforms
    # -----------------------------------------------------
    # keeping the same image size as for pretraining
    SPATIAL_SIZE = (64, 256, 256)
    ROI_SIZE = (64, 64, 64)

    # roi_size is used to crop samples around the spinal cord
    train_transforms = define_finetune_train_transforms(spatial_size=SPATIAL_SIZE, roi_size=ROI_SIZE)
    val_transforms = define_finetune_val_transforms(spatial_size=SPATIAL_SIZE, roi_size=ROI_SIZE)

    # -----------------------------------------------------
    # Sanity Check for the transforms
    # -----------------------------------------------------
    check_ds = Dataset(data=train_list, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    logger.info(f'original image shape: {check_data["image"][0][0].shape}')
    logger.info(f'original SC label shape: {check_data["label_sc"][0][0].shape}')
    logger.info(f'original lesion label shape: {check_data["label_lesion"][0][0].shape}')

    # -----------------------------------------------------
    # Training Config
    # -----------------------------------------------------

    CUDA_NUM=args.cuda
    device = torch.device(f"cuda:{CUDA_NUM}")
    model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=ROI_SIZE,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="conv",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    # -----------------------------------------------------
    # Load ViT backbone weights into UNETR
    # -----------------------------------------------------
    if use_pretrained is True:
        logger.info(f"Loading Weights from the Path {pretrained_model_path}")
        vit_dict = torch.load(pretrained_model_path)
        vit_weights = vit_dict["state_dict"]

        # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
        # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
        # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
        # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
        model_dict = model.vit.state_dict()

        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.vit.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict
        logger.info("Pretrained Weights Succesfully Loaded !")

    elif use_pretrained is False:
        print("No weights were loaded, all weights being used are randomly initialized!")

    model.to(device)

    # Training Hyper-params
    lr = 1e-4
    max_iterations = 30000
    eval_num = 100
    batch_size = 8
    loss_function = DiceCELoss()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    epoch_loss_values = []
    metric_values = []

    # -----------------------------------------------------
    # Create dataloaders for training
    # -----------------------------------------------------

    NUM_WORKERS = batch_size

    train_dataset = CacheDataset(data=train_list, transform=train_transforms, cache_rate=0.5, num_workers=NUM_WORKERS)
    val_dataset = CacheDataset(data=val_list, transform=val_transforms, cache_rate=0.25, num_workers=NUM_WORKERS)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            persistent_workers=False)

    # -----------------------------------------------------
    # Training Loop with Validation
    # -----------------------------------------------------

    # Create validation_figures directory if it does not exist
    if not os.path.exists(os.path.join(logdir_path, "validation_figures")):
        os.mkdir(os.path.join(logdir_path, "validation_figures"))

    def validation(epoch_iterator_val):
        model.eval()
        dice_vals = []

        with torch.no_grad():
            for _step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].cuda(CUDA_NUM), batch["label_lesion"].cuda(CUDA_NUM))
                val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, batch_size, model)
                val_labels_list = decollate_batch(val_labels)
                val_outputs_list = decollate_batch(val_outputs)
                dice_metric(y_pred=val_outputs_list, y=val_labels_list)
                dice = dice_metric.aggregate().item()
                dice_vals.append(dice)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice=%2.5f)" % (global_step, 10.0, dice))

                # Check whether val_labels is not empty (i.e., contains a lesion)
                if val_labels[0, 0, :, :, :].sum() > 0:
                    logger.info(f"Lesion found in the validation image. Saving the validation images.")
                    # if so, get corresponding slice
                    slice_idx = val_labels[0, 0, :, :, :].detach().cpu().numpy().nonzero()[2][0].item()
                    logger.info(slice_idx)
                    # print unique values in the slice to see if it is binary
                    logger.info(np.unique(val_labels[0, 0, :, :, slice_idx].detach().cpu().numpy()))
                    # Plot and save input and output validation images to see how the model is learning
                    plt.figure(1, figsize=(8, 8))
                    plt.subplot(2, 2, 1)
                    logger.info(f'Input image shape: {val_inputs.detach().cpu().numpy().shape}')
                    plt.imshow(val_inputs[0, 0, :, :, slice_idx].detach().cpu().numpy(), cmap="gray")
                    plt.title("Input Image")
                    plt.subplot(2, 2, 2)
                    logger.info(f'Ground truth shape: {val_labels.detach().cpu().numpy().shape}')
                    plt.imshow(val_inputs[0, 0, :, :, slice_idx].detach().cpu().numpy(), cmap="gray")
                    plt.imshow(val_labels[0, 0, :, :, slice_idx].detach().cpu().numpy(), alpha=0.5, cmap="jet",
                               interpolation='nearest')
                    plt.title("Ground Truth")
                    plt.subplot(2, 2, 3)
                    logger.info(f'Predicted shape: {val_outputs.detach().cpu().numpy().shape}')
                    plt.imshow(val_outputs[0, 0, :, :, slice_idx].detach().cpu().numpy(), cmap="gray")
                    plt.title("Predicted")
                    # Include the global_step as master title
                    plt.suptitle(f"Validation Step: {global_step}")
                    # Use 5 leading zeros for the global_step
                    fname = os.path.join(logdir_path, "validation_figures", f"val_{global_step:05d}_{_step}.png")
                    plt.savefig(fname)
                    plt.close(1)
                    logger.info(f"Saved validation images to {fname}")

            dice_metric.reset()

        mean_dice_val = np.mean(dice_vals)
        return mean_dice_val

    def train(global_step, train_loader, dice_val_best, global_step_best):
        model.train()
        epoch_loss = 0
        epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            step += 1
            x, y = (batch["image"].cuda(CUDA_NUM), batch["label_lesion"].cuda(CUDA_NUM))
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, max_iterations, loss))

            if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = validation(epoch_iterator_val)

                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    torch.save(model.state_dict(), os.path.join(logdir_path, "best_metric_model.pth"))
                    logger.info(f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} "
                                f"Current Avg. Dice: {dice_val}")
                else:
                    logger.info(f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} "
                                f"Current Avg. Dice: {dice_val}")

                plt.figure(1, (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Iteration Average Loss")
                x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
                y = epoch_loss_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [eval_num * (i + 1) for i in range(len(metric_values))]
                y = metric_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.grid()
                plt.savefig(os.path.join(logdir_path, "finetune_quick_update.png"))
                plt.clf()
                plt.close(1)

            global_step += 1
        return global_step, dice_val_best, global_step_best

    while global_step < max_iterations:
        global_step, dice_val_best, global_step_best = train(global_step, train_loader, dice_val_best, global_step_best)
    model.load_state_dict(torch.load(os.path.join(logdir_path, "best_metric_model.pth")))

    logger.info(f"train completed, best_metric: {dice_val_best:.4f} " f"at iteration: {global_step_best}")


if __name__ == "__main__":
    main()
