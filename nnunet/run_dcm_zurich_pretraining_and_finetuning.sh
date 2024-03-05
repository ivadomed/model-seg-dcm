#!/bin/bash
#
# This script combines pre-training on dcm-zurich for compression detection and 
# fine-tuning on dcm-zurich-lesions for lesion segmentation.
#   
# Assumes that the datasets for pretraining and finetuning already exist. 
# These are most likely the outputs of `convert_bids_to_nnUNetv2*.py` scripts.
# 
# Usage:
#     cd ~/code/model-seg-dcm
#     ./nnunet/run_dcm_zurich_pretraining_and_finetuning.sh
#
# Author: Jan Valosek, Naga Karthik
#

# Uncomment for full verbose
# set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Global variables
cuda_visible_devices=2
folds=(3)
sites=(dcm-zurich-lesions dcm-zurich-lesions-20231115)
nnunet_trainer="nnUNetTrainer"
# nnunet_trainer="nnUNetTrainer_2000epochs"       # default: nnUNetTrainer
configuration="3d_fullres"                      # for 2D training, use "2d"

# NOTE: after pre-training for 1000 epochs, fine-tuning doesn't need that many epochs
# hence, creating a new variant with less epochs
nnunet_trainer_ftu="nnUNetTrainer_250epochs"

# Variables for pretraining on dcm-zurich (i.e. source dataset)
dataset_num_ptr="191"
dataset_name_ptr="Dataset${dataset_num_ptr}_dcmZurichPretrain"
dataset_git_annex_name="dcm-zurich"

# Variables for finetuning on dcm-zurich-lesions (i.e. target dataset)
dataset_num_ftu="192"
dataset_name_ftu="Dataset${dataset_num_ftu}_dcmZurichLesionsFinetune"


echo "-------------------------------------------------------"
echo "Running plan_and_preprocess for ${dataset_name_ftu}"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num_ftu} --verify_dataset_integrity -c ${configuration}

echo "-------------------------------------------------------"
echo "Running plan_and_preprocess for ${dataset_name_ptr}"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num_ptr} --verify_dataset_integrity -c ${configuration}

echo "-------------------------------------------------------"
echo "Extracting dataset fingerprint for ${dataset_name_ptr}"
echo "-------------------------------------------------------"
nnUNetv2_extract_fingerprint -d ${dataset_num_ptr}

echo "-------------------------------------------------------"
echo "Moving plans from ${dataset_name_ftu} to ${dataset_name_ptr}"
echo "-------------------------------------------------------"
nnUNetv2_move_plans_between_datasets -s ${dataset_num_ftu} -t ${dataset_num_ptr} -sp nnUNetPlans -tp nnUNetMovedPlans

echo "-------------------------------------------------------"
echo "Running (only) preprocessing for ${dataset_name_ptr} after moving plans"
echo "-------------------------------------------------------"
nnUNetv2_preprocess -d ${dataset_num_ptr} -plans_name nnUNetMovedPlans


echo "-------------------------------------------------------"
echo "Running pretraining on ${dataset_name_ptr} ..."
echo "-------------------------------------------------------"
# training
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num_ptr} ${configuration} all -tr ${nnunet_trainer} -p nnUNetMovedPlans

echo "-------------------------------------------------------"
echo "Running inference on ${dataset_name_ptr} ..."
echo "-------------------------------------------------------"
# running inference on the source dataset
CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name_ptr}/imagesTs_${dataset_git_annex_name} -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name_ptr}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_all/test -d ${dataset_num_ptr} -f all -c ${configuration}


echo "-------------------------------------------------------"
echo "Pretraining done, Running finetuning on ${dataset_name_ftu} ..."
echo "-------------------------------------------------------"
path_ptr_weights=${nnUNet_results}/${dataset_name_ptr}/${nnunet_trainer}__nnUNetMovedPlans__${configuration}/fold_all/checkpoint_best.pth


for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training/Finetuning on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_name_ftu} ${configuration} ${fold} -tr ${nnunet_trainer_ftu} -pretrained_weights ${path_ptr_weights}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # run inference on testing sets for each site
    for site in ${sites[@]}; do
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name_ftu}/imagesTs_${site} -tr ${nnunet_trainer_ftu} -o ${nnUNet_results}/${dataset_name_ftu}/${nnunet_trainer}__nnUNetMovedPlans__${configuration}/fold_${fold}/test_${site} -d ${dataset_num_ftu} -f ${fold} -c ${configuration}

        echo "-------------------------------------------------------"
        echo "Running ANIMA evaluation on Test set for ${site} "
        echo "-------------------------------------------------------"

        python testing/compute_anima_metrics.py --pred-folder ${nnUNet_results}/${dataset_name_ftu}/${nnunet_trainer_ftu}__nnUNetMovedPlans__${configuration}/fold_${fold}/test_${site} --gt-folder ${nnUNet_raw}/${dataset_name_ftu}/labelsTs_${site} --dataset-name ${site}

    done

done