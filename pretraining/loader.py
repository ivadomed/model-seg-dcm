import os

from monai.data import (DataLoader, Dataset, list_data_collate, DistributedSampler,
                        CacheDataset, load_decathlon_datalist)

from transforms import train_transforms, val_transforms


def load_data(datalists_paths, train_batch_size, val_batch_size, num_workers=8, use_distributed=False,
              crop_size=(64, 192, 320), patch_size=(64, 64, 64), device="cuda", task="pretraining"):
    """
    Return train and val dataloaders from datalist json file
    :param datalists_paths: path(s) to the datalist json file(s)
    :param train_batch_size: batch size for training dataloader
    :param val_batch_size: batch size for validation dataloader
    :param num_workers: number of workers for dataloader
    :param use_distributed: whether to use distributed training
    :param crop_size: crop size for training data
    :param patch_size: patch size for training data
    :param device: device to load data and apply transforms
    :param task: task for train/val transforms; choices: pretraining or finetuning
    """
    train_datalist = []
    val_datalist = []
    for datalist_path in datalists_paths:
        train_datalist += load_decathlon_datalist(data_list_file_path=datalist_path, data_list_key="training")
        val_datalist += load_decathlon_datalist(data_list_file_path=datalist_path, data_list_key="validation")

    train_tfs = train_transforms(crop_size, patch_size, device=device, mode=task)
    val_tfs = val_transforms(crop_size, mode=task)

    # training dataset
    train_ds = CacheDataset(data=train_datalist, transform=train_tfs, cache_rate=0.5, num_workers=8)
    # validation dataset
    val_ds = CacheDataset(data=val_datalist, transform=val_tfs, cache_rate=0.25, num_workers=8)

    if use_distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
        val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # training dataloader    
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=False, num_workers=num_workers, 
                            pin_memory=False, sampler=train_sampler, persistent_workers=True, drop_last=False)
    # validation dataloader
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=False, sampler=val_sampler, persistent_workers=True, drop_last=False)

    return train_loader, val_loader
