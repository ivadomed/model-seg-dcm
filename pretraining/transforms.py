
import numpy as np
import monai.transforms as transforms


def train_transforms(crop_size, patch_size, device="cuda", mode="pretraining"):

    if mode == "pretraining":
        # NOTE: the pre-trainining is done for SC segmentation
        all_keys = ["image", "label_sc"]
        
        monai_transforms = [
            # pre-processing    
            transforms.LoadImaged(keys=all_keys, image_only=False),
            transforms.EnsureChannelFirstd(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RPI"),
            # NOTE: spine interpolation with order=2 is spline, order=1 is linear
            transforms.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            transforms.CropForegroundd(keys=all_keys, source_key="image"),
            transforms.ResizeWithPadOrCropd(keys=all_keys, spatial_size=crop_size),
            # convert the data to Tensor without meta, move to GPU and cache it to avoid CPU -> GPU sync in every epoch
            transforms.transforms.EnsureTyped(keys=all_keys, device=device, track_meta=False),
            
            # sample patches from cropped image
            transforms.RandCropByPosNegLabeld(keys=all_keys,
                label_key="label_sc",       # cropping around the SC
                spatial_size=patch_size,
                pos=1, neg=0,
                num_samples=4,  # if num_samples=4, then 4 samples/image are randomly generated
                image_key="image", image_threshold=0),
            
            # data-augmentation
            # NOTE: we use heavily-deforming transformations for SC segmentation
            transforms.RandAffined(keys=all_keys, mode=(2, 1), prob=0.9,
                        rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians 
                        scale_range=(-0.2, 0.2),   
                        translate_range=(-0.1, 0.1)),
            transforms.Rand3DElasticd(keys=all_keys, prob=0.5,
                        sigma_range=(3.5, 5.5), 
                        magnitude_range=(25., 35.)),
            transforms.RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
            transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
            transforms.RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
            transforms.RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
            transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
            transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
            transforms.RandFlipd(keys=all_keys, prob=0.3,),
        ]
    elif mode == "finetuning":

        # NOTE: the fine-tuning is done for lesion segmentation
        all_keys = ["image", "label_sc", "label_lesion"]

        monai_transforms = [    
            # pre-processing
            transforms.LoadImaged(keys=all_keys),
            transforms.EnsureChannelFirstd(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RPI"),
            # NOTE: spine interpolation with order=2 is spline, order=1 is linear
            transforms.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=(2, 1, 1)),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            transforms.ResizeWithPadOrCropd(keys=all_keys, spatial_size=crop_size,),
            # convert the data to Tensor without meta, move to GPU and cache it to avoid CPU -> GPU sync in every epoch
            transforms.EnsureTyped(keys=all_keys, device=device, track_meta=False),
            # data-augmentation
            # transforms.RandAffined(keys=all_keys, mode=(2, 1), prob=0.9,
            #             rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians 
            #             scale_range=(-0.2, 0.2),   
            #             translate_range=(-0.1, 0.1)),
            # transforms.Rand3DElasticd(keys=all_keys, prob=0.5, sigma_range=(3.5, 5.5), magnitude_range=(25., 35.)),
            transforms.RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
            transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.25),    # this is monai's RandomGamma
            # transforms.RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
            # transforms.RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
            transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
            transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
            transforms.RandFlipd(keys=all_keys, prob=0.3,),
        ]
    else:
        raise ValueError("Invalid type: {}. Choices: [pretraining, finetuning]".format(type))

    return transforms.Compose(monai_transforms) 

def inference_transforms(crop_size, lbl_key="label"):
    return transforms.Compose([
            transforms.LoadImaged(keys=["image", lbl_key], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            transforms.Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            transforms.DivisiblePadd(keys=["image", lbl_key], k=2**5),   # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms(crop_size, mode="pretraining"):

    if mode == "pretraining":
        all_keys = ["image", "label_sc"]

        return transforms.Compose([
            transforms.LoadImaged(keys=all_keys, image_only=False),
            transforms.EnsureChannelFirstd(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RPI"),
            transforms.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            transforms.ResizeWithPadOrCropd(keys=all_keys, spatial_size=crop_size,),
        ])
    elif mode == "finetuning":
        all_keys = ["image", "label_sc", "label_lesion"]
        return transforms.Compose([
            transforms.LoadImaged(keys=all_keys, image_only=False),
            transforms.EnsureChannelFirstd(keys=all_keys),
            transforms.Orientationd(keys=all_keys, axcodes="RPI"),
            transforms.Spacingd(keys=all_keys, pixdim=(1.0, 1.0, 1.0), mode=(2, 1, 1)),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            transforms.ResizeWithPadOrCropd(keys=all_keys, spatial_size=crop_size,),
        ])
    else :
        raise ValueError("Invalid type: {}. Choices: [pretraining, finetuning]".format(type))
