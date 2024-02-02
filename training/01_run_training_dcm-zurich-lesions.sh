#!/bin/bash
#
# Run nnUNet training and testing on dcm-zurich-lesions and dcm-zurich-lesions-20231115 datasets
#
# Usage:
#     cd ~/code/model-seg-dcm
#     ./training/01_run_training_dcm-zurich-lesions.sh
#
# Author: Jan Valosek, Naga Karthik
#

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT


# define arguments for nnUNet
dataset_num="601"
dataset_name="Dataset${dataset_num}_DCMlesions"
nnunet_trainer="nnUNetTrainer"
#nnunet_trainer="nnUNetTrainer_2000epochs"       # default: nnUNetTrainer
configuration="3d_fullres"                      # for 2D training, use "2d"
cuda_visible_devices=1
folds=(1 2)
#folds=(3)
sites=(dcm-zurich-lesions dcm-zurich-lesions-20231115)


echo "-------------------------------------------------------"
echo "Running preprocessing and verifying dataset integrity"
echo "-------------------------------------------------------"
nnUNetv2_plan_and_preprocess -d ${dataset_num} --verify_dataset_integrity -c ${configuration}


for fold in ${folds[@]}; do
    echo "-------------------------------------------"
    echo "Training on Fold $fold"
    echo "-------------------------------------------"

    # training
    CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_train ${dataset_num} ${configuration} ${fold} -tr ${nnunet_trainer}

    echo ""
    echo "-------------------------------------------"
    echo "Training completed, Testing on Fold $fold"
    echo "-------------------------------------------"

    # run inference on testing sets for each site
    for site in ${sites[@]}; do
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices} nnUNetv2_predict -i ${nnUNet_raw}/${dataset_name}/imagesTs_${site} -tr ${nnunet_trainer} -o ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} -d ${dataset_num} -f ${fold} -c ${configuration} # -step_size 0.9 --disable_tta

        echo "-------------------------------------------------------"
        echo "Running ANIMA evaluation on Test set for ${site} "
        echo "-------------------------------------------------------"

        python training/02_compute_anima_metrics.py --pred-folder ${nnUNet_results}/${dataset_name}/${nnunet_trainer}__nnUNetPlans__${configuration}/fold_${fold}/test_${site} --gt-folder ${nnUNet_raw}/${dataset_name}/labelsTs_${site} -dname ${site} --label-type lesion

    done

done
