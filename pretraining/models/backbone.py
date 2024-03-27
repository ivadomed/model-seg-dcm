
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from monai.networks.nets import SwinUNETR, UNet

from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0

from models.model_utils import InitWeights_He, count_parameters



def create_nnunet_from_plans(plans):
    """
    Adapted from nnUNet's source code: 
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

    """
    num_stages = len(plans["conv_kernel_sizes"])
    dim = len(plans["conv_kernel_sizes"][0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = plans["UNet_class_name"]
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). '
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["n_conv_per_stage_encoder"],
        'n_conv_per_stage_decoder': plans["n_conv_per_stage_decoder"]
    }
    
    # network class name!!
    model = network_class(
        input_channels=plans["in_channels"],
        n_stages=num_stages,
        features_per_stage=[min(plans["UNet_base_num_features"] * 2 ** i, 
                                plans["unet_max_num_features"]) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=plans["conv_kernel_sizes"],
        strides=plans["pool_op_kernel_sizes"],
        num_classes=plans["out_channels"],    
        deep_supervision=plans["deep_supervision"],
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)

    return model



class BackboneModel(nn.Module):
    def __init__(self, model_name, config):
        super(BackboneModel, self).__init__()

        self.model_name = model_name
        self.run_folder = ""

        if self.model_name == "monai-unet":
            self.model = UNet(
                spatial_dims=3,
                in_channels=config["model"]["monai-unet"]["in_channels"],
                out_channels=config["model"]["monai-unet"]["out_channels"],
                channels=config["model"]["monai-unet"]["channels"],
                strides=config["model"]["monai-unet"]["strides"],
                kernel_size=3,
                num_res_units=config["model"]["monai-unet"]["num_res_units"],
                norm="INSTANCE"
            )
            self.run_folder = f"{model_name}_seed={config['seed']}_" \
                                f"nf={config['model']['monai-unet']['channels'][0]}_" \
                                f"nrs={config['model']['monai-unet']['num_res_units']}_" \
                                f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                                f"bs={config['opt']['batch_size']}" \

        elif self.model_name == "swinunetr":
            self.model = SwinUNETR(
                spatial_dims=3,
                img_size=config["preprocessing"]["crop_pad_size"],
                in_channels=config["model"]["swinunetr"]["in_channels"],
                out_channels=config["model"]["swinunetr"]["out_channels"],
                #depths=config["model"]["swinunetr"]["depths"],
                #feature_size=config["model"]["swinunetr"]["feature_size"],
                #num_heads=config["model"]["swinunetr"]["num_heads"],
            )
            self.run_folder =  f"{model_name}_seed={config['seed']}_" \
                                f"d={config['model']['swinunetr']['depths'][0]}_" \
                                f"fs={config['model']['swinunetr']['feature_size']}_" \
                                f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                                f"bs={config['opt']['batch_size']}" \

        elif self.model_name == "nnunet":
            nnunet_plans = {
                "UNet_class_name": "PlainConvUNet",
                "in_channels": config["model"]["nnunet"]["in_channels"],
                "out_channels": config["model"]["nnunet"]["out_channels"],
                "UNet_base_num_features": config["model"]["nnunet"]["base_num_features"],
                "n_conv_per_stage_encoder": config["model"]["nnunet"]["n_conv_per_stage_encoder"],
                "n_conv_per_stage_decoder": config["model"]["nnunet"]["n_conv_per_stage_decoder"],
                "pool_op_kernel_sizes": config["model"]["nnunet"]["pool_op_kernel_sizes"],
                "conv_kernel_sizes": [
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3],
                    [3, 3, 3]
                ],
                "unet_max_num_features": config["model"]["nnunet"]["max_num_features"],
                "deep_supervision": config["model"]["nnunet"]["enable_deep_supervision"]
            }
            self.model = create_nnunet_from_plans(plans=nnunet_plans)
            self.run_folder = f"{model_name}_seed={config['seed']}_" \
                                f"nf={config['model']['nnunet']['base_num_features']}_" \
                                f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                                f"bs={config['opt']['batch_size']}" \

        else:
            raise ValueError(f"Model {model_name} not supported.")

        trainable_params = count_parameters(self.model)
        logger.info(f"Model {model_name} created with {(trainable_params / 1e6):.3f} trainable parameters.")

    def load_pretrained(self, path_pretrained_weights):

        if isinstance(self.model, DDP):
            model = self.model.module
        else:
            model = self.model

        if self.model_name == "swinunetr":
            # does not load the final conv layers used for getting the segmentation map
            skip_strings_pretrained = ["out."]
        
        elif self.model_name == "monai-unet":
            skip_strings_pretrained = []

        elif self.model_name == "nnunet":
            # Segmentation layers (the 1x1(x1) layers that produce the segmentation maps) 
            # identified by keys ending with '.seg_layers') are not transferred!
            skip_strings_pretrained = ["seg_layers."]
        
        else:
            skip_strings_pretrained = []
            
        model_prior_state_dict = model.state_dict()
        model_updated_state_dict = model_prior_state_dict

        pretrained_state_dict = torch.load(path_pretrained_weights)
        # remove the 'model.' prefix from the keys
        # NOTE: some the monai-unet already has a model. prefix in the keys and that is prefixed with 
        # another 'model.' -- so if k.replace("model."", "") is used then it removes both prefixes. 
        pretrained_state_dict = {k[6:]: v for k, v in pretrained_state_dict.items()}

        pretrained_dict_pruned = {k: v for k, v in pretrained_state_dict.items()
                    if k in model_prior_state_dict.keys() and all([i not in k for i in skip_strings_pretrained])}
        
        model_updated_state_dict.update(pretrained_dict_pruned)
        model.load_state_dict(model_updated_state_dict, strict=True)
        # get the final state of the model after updating weights
        model_final_state_dict = model.state_dict()

        # sanity check to ensure that weights got loaded successfully
        layer_ctr = 0
        for k, _v in pretrained_dict_pruned.items():
            if k in model_prior_state_dict:
                layer_ctr +=1 

                old_wts, new_wts = model_prior_state_dict[k], model_final_state_dict[k]
                old_wts, new_wts = old_wts.to("cpu").numpy(), new_wts.to("cpu").numpy()
                diff = np.mean(np.abs(old_wts, new_wts))

            logger.info(f"Layer {k} --> Difference in Random vs Pretrained Weights: {diff}")
            if diff == 0.0:
                logger.info(f"Warning: No difference in weights for layer {k}")

        # for k, _v in model_final_state_dict.items():
        #     if "seg_layers." in k:
        #         old = model_prior_state_dict[k]
        #         new = model_final_state_dict[k]
        #         diff = np.mean(np.abs(old.to("cpu").numpy(), new.to("cpu").numpy()))
        #         logger.info(f"Layer {k} --> Difference in Random vs Pretrained Weights: {diff}")

        # # METHOD 2
        # orig_dict = model.state_dict()
        # pretrained_dict = torch.load(path_pretrained_weights)
        # pretrained_dict = {k.replace("model.", ""): v for k, v in pretrained_dict.items()}

        # for key in list(pretrained_dict.keys()):
        #     if 'out' not in key:
        #         orig_dict[key] = pretrained_dict[key]

        # model.load_state_dict(orig_dict)
        # model_final_state_dict = model.state_dict()

        # for k, _v in model_final_state_dict.items():
        #     if k.startswith("out."):
        #         old = orig_dict[k]
        #         new = model_final_state_dict[k]
        #         diff = np.mean(np.abs(old.to("cpu").numpy(), new.to("cpu").numpy()))
        #         logger.info(f"Layer {k} --> Difference in Random vs Pretrained Weights: {diff}")

        logger.info(f"Total updated layers {layer_ctr} / {len(model_prior_state_dict)}")
        logger.info(f"Pretrained Weights Succesfully Loaded !")
                                                      

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    config = {
        "preprocessing":
        {
            "crop_pad_size": [32, 96, 160]
        },
        "model": {
            "monai-unet": {
                "in_channels": 1,
                "out_channels": 1,
                "channels": [32, 64, 128, 256, 320],
                "strides": [2, 2, 2, 2],
                "num_res_units": 2
            },
            "swinunetr": {
                "in_channels": 1,
                "out_channels": 1,
                "depths": [2, 2, 2, 2],
                "feature_size": 36,
                "num_heads": [3, 6, 12, 24]
            },
            "nnunet": {
                "in_channels": 1,
                "out_channels": 1,
                "base_num_features": 32,
                "max_num_features": 320,
                "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
                "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
                "pool_op_kernel_sizes": [
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [2, 2, 2],
                    [1, 2, 2]],
                "enable_deep_supervision": True
            }
        }
    }

    model_name = "nnunet" # "swinunetr" # "monai-unet"
    backbone = BackboneModel(model_name, config)
    # load pretrained weights
    backbone.load_pretrained(f"{model_name}_checkpoint.pth")

    input = torch.randn(1, 1, 32, 96, 160)
    output = backbone(input)

    # # save the model
    # torch.save(backbone.state_dict(), f"{model_name}_checkpoint.pth")

    # # print the keys of the model
    # orig_keys = backbone.model.state_dict().keys()
    # print(orig_keys)

    # print("AFTER LOADING")

    # # load the model
    # loaded_model = torch.load(f"{model_name}_checkpoint.pth")
    # backbone.load_state_dict(loaded_model)
    # loaded_keys = backbone.model.state_dict().keys()
    # # print(backbone.model.state_dict().keys())
    # print(loaded_keys == orig_keys)



