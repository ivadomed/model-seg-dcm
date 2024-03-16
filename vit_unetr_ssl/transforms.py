import numpy as np

from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    ResizeWithPadOrCropd,
    AsDiscreted,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    OneOf,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    RandFlipd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandSimulateLowResolutiond,
    Rand3DElasticd,
    RandAffined,
    ToTensord
)


def define_pretrain_transforms(keys, spatial_size, roi_size, number_of_holes=5):
    """
    Define MONAI Transforms for Training/Validation of the self-supervised pretrained model
    :args: keys: list of keys to be used for the transforms, e.g. ["image", "label"]
    :args: spatial_size: spatial size of the input image, e.g. (64, 256, 256)
    :args: roi_size: spatial size of the region of interest, e.g. (64, 64, 64)
    :args: number_of_holes: number of holes to be used for the RandCoarseDropoutd and RandCoarseShuffled transforms
    """
    transforms = Compose(
        [
            # Load image data using the key "image"
            LoadImaged(keys=keys, image_only=True),
            # Ensure that the channel dimension is the first dimension of the image tensor.
            EnsureChannelFirstd(keys=keys),
            # Ensure that the image orientation is consistently RPI
            Orientationd(keys=keys, axcodes="RPI"),
            # Resample the images to a specified pixel spacing
            # NOTE: spine interpolation with order=2 is spline, order=1 is linear
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
            # Normalize the intensity values of the image
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            # Remove background pixels to focus on regions of interest.
            # CropForegroundd(keys=["image"], source_key="image"),
            # Pad the image to a specified spatial size if its size is smaller than the specified dimensions
            ResizeWithPadOrCropd(keys=keys, spatial_size=spatial_size),
            AsDiscreted(keys='label', to_onehot=None, threshold=0.5),
            # Randomly crop samples of a specified size around the label (spinal cord)
            # Note that it seems that the transform randomly selects a foreground point from image, then use it as
            # center crop. This means that it can find the closest voxel that is just outside the SC and use it as the
            # center (hence it includes the SC)
            # https://github.com/Project-MONAI/MONAI/issues/452#issuecomment-636065502
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="label",
                spatial_size=roi_size,
                pos=2,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            # Create copies of items in the dictionary under new keys, allowing for the same image to be manipulated
            # differently in subsequent transformations
            CopyItemsd(keys=["image"], times=2, names=["gt_image", "image_2"], allow_missing_keys=False),

            # AUGMENTED VIEW 1
            OneOf(
                transforms=[
                    # Randomly drop regions of the image
                    # 'dropout_holes=True': the dropped regions will be set to zero, introducing regions of no
                    # information within the image.
                    RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=number_of_holes,
                        spatial_size=roi_size[0] // 4,
                        dropout_holes=True,
                        # max_spatial_size=(roi_size[0]//4, roi_size[1]//4, roi_size[2]//4)
                    ),
                    # 'dropout_holes=False': the areas inside the holes will be filled with random noise
                    RandCoarseDropoutd(
                        keys=["image"],
                        prob=1.0,
                        holes=number_of_holes,
                        spatial_size=roi_size[0] // 2,
                        dropout_holes=False,
                        # max_spatial_size=(roi_size[0]//2, roi_size[1]//2, roi_size[2]//2)
                    ),
                ]
            ),
            # Randomly select regions in the image, then shuffle the pixels within every region
            RandCoarseShuffled(keys=["image"], prob=0.8, holes=number_of_holes, spatial_size=roi_size[2] // 4),

            # AUGMENTED VIEW 2
            # Please note that that if image and image_2 are called via the same transform call because of the
            # determinism they will get augmented the exact same way which is not the required case here, hence two
            # calls are made
            OneOf(
                transforms=[
                    # Randomly drop regions of the image
                    # 'dropout_holes=True': the dropped regions will be set to zero, introducing regions of no
                    # information within the image.
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=number_of_holes,
                        spatial_size=roi_size[0] // 4,
                        dropout_holes=True,
                        # max_spatial_size=(roi_size[0]//4, roi_size[1]//4, roi_size[2]//4)
                    ),
                    # 'dropout_holes=False': the areas inside the holes will be filled with random noise
                    RandCoarseDropoutd(
                        keys=["image_2"],
                        prob=1.0,
                        holes=number_of_holes,
                        spatial_size=roi_size[0] // 2,
                        dropout_holes=False,
                        # max_spatial_size=(roi_size[0]//2, roi_size[1]//2, roi_size[2]//2)
                    ),
                ]
            ),
            # Randomly select regions in the image, then shuffle the pixels within every region
            RandCoarseShuffled(keys=["image_2"], prob=0.8, holes=number_of_holes, spatial_size=roi_size[2] // 4),
        ]
    )

    return transforms


def define_finetune_train_transforms(spatial_size):
    """
    Define MONAI Transforms for Training of the fine-tuned model
    """
    train_transforms = Compose(
        [
            # Load image data and GT using the keys "image" and "label"
            LoadImaged(keys=["image", "label"], image_only=False),
            # Ensure that the channel dimension is the first dimension of the image tensor.
            EnsureChannelFirstd(keys=["image", "label"]),
            # Ensure that the image orientation is consistently RPI
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            # Resample the images to a specified pixel spacing
            # NOTE: spine interpolation with order=2 is spline, order=1 is linear
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
            # Normalize the intensity values of the image
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            # Remove background pixels to focus on regions of interest.
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Pad the image to a specified spatial size if its size is smaller than the specified dimensions
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            # Randomly crop samples of a specified size
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=4,  # if num_samples=4, then 4 samples/image are randomly generated
                image_key="image",
                image_threshold=0,
            ),
            # data-augmentation
            # NOTE: the following transforms are based on contrast-agnostic data augmentation:
            # https://github.com/ivadomed/model-seg-dcm/blob/nk/dcm-zurich-pretraining/monai/transforms.py
            RandAffined(keys=["image", "label"], mode=(2, 1), prob=0.9,
                        rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),
                        # monai expects in radians
                        scale_range=(-0.2, 0.2),
                        translate_range=(-0.1, 0.1)),
            Rand3DElasticd(keys=["image", "label"], prob=0.5,
                           sigma_range=(3.5, 5.5),
                           magnitude_range=(25., 35.)),
            RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
            RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),  # this is monai's RandomGamma
            RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
            RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0),
                                prob=0.3),
            RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),
            # this is nnUNet's BrightnessMultiplicativeTransform
            RandFlipd(keys=["image", "label"], prob=0.3, ),
        ]
    )

    return train_transforms


def define_finetune_val_transforms(spatial_size):
    """
    Define MONAI Transforms for Validation of the fine-tuned model
    """
    val_transforms = Compose(
        [
            # Load image data and GT using the keys "image" and "label"
            LoadImaged(keys=["image", "label"], image_only=False),
            # Ensure that the channel dimension is the first dimension of the image tensor.
            EnsureChannelFirstd(keys=["image", "label"]),
            # Ensure that the image orientation is consistently RPI
            Orientationd(keys=["image", "label"], axcodes="RPI"),
            # Resample the images to a specified pixel spacing
            # NOTE: spine interpolation with order=2 is spline, order=1 is linear
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
            # Normalize the intensity values of the image
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            # Remove background pixels to focus on regions of interest.
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Pad the image to a specified spatial size if its size is smaller than the specified dimensions
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"])
        ]
    )

    return val_transforms
