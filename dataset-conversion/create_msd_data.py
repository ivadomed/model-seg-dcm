"""
Create MSD-style JSON datalist file for BIDS datasets.
The following two keys are included in the JSON file: 'image' and 'label_sc'.

NOTE: the script is meant to be used for pre-training, meaning that the dataset is split into training and validation.
In other words, NO testing set is created.

The script has to be run for each dataset separately, meaning that one JSON file is created for each dataset.

Example usage:
    python create_msd_data.py
        --path-data /Users/user/data/spine-generic
        --dataset-name spine-generic
        --path-out /Users/user/data/spine-generic

    python create_msd_data.py
        --path-data /Users/user/data/dcm-zurich
        --dataset-name dcm-zurich
        --path-out /Users/user/data/dcm-zurich
"""

import os
import json
import argparse
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split

contrast_dict = {
    'spine-generic': 'space-other_T2w',     # iso T2w (preprocessed data)
    'whole-spine': 'T2w',                   # iso T2w
    'canproco': 'ses-M0_T2w',               # iso T2w (session M0)
    'dcm-zurich': 'acq-axial_T2w',          # axial T2w
    'sci-paris': 'T2w',                     # iso T2w
    'sci-colorado': 'T2w'                   # axial T2w
}

# Spinal cord segmentation file suffixes for different datasets
sc_fname_suffix_dict = {
    'spine-generic': 'label-SC_seg',
    'whole-spine': 'seg',
    'canproco': 'seg-manual',
    'dcm-zurich': 'label-SC_mask-manual',
    'sci-paris': 'seg-manual',
    'sci-colorado': 'seg-manual'
}


def get_parser():
    parser = argparse.ArgumentParser(description='Create MSD-style JSON datalist file for BIDS datasets.')

    parser.add_argument('--path-data', required=True, type=str,
                        help='Path to BIDS dataset. Example: /Users/user/data/dcm-zurich')
    parser.add_argument('--dataset-name', required=True, type=str,
                        help='Name of the dataset. Example: spine-generic or dcm-zurich.')
    parser.add_argument('--path-out', type=str, required=True,
                        help='Path to the output directory where dataset json is saved')
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training and validation 0-1. '
                             'Example: --split 0.8 0.2')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")

    return parser


def main():
    args = get_parser().parse_args()

    dataset = os.path.abspath(args.path_data)
    dataset_name = args.dataset_name
    train_ratio, val_ratio = args.split
    seed = args.seed
    path_out = os.path.abspath(args.path_out)

    # Check if the dataset name is valid
    if dataset_name not in contrast_dict.keys():
        raise ValueError(f"Dataset name {dataset_name} is not valid. Choose from {contrast_dict.keys()}")

    contrast = contrast_dict[dataset_name]
    sc_fname_suffix = sc_fname_suffix_dict[dataset_name]
    datalist_fname = f"{dataset_name}_seed{seed}"

    train_images, val_images = {}, {}

    # For spine-generic, we add 'derivatives/data_preprocessed' to the path to use the preprocessed data with the same
    # resolution and orientation as the spinal cord segmentations
    if dataset_name == 'spine-generic':
        root = Path(dataset) / 'derivatives/data_preprocessed'
    else:
        root = Path(dataset)
    # Path to 'derivatives/labels with spinal cord segmentations
    labels = Path(dataset) / 'derivatives/labels'

    # Check if the dataset path exists
    if not os.path.exists(root):
        raise ValueError(f"Path {root} does not exist.")
    if not os.path.exists(labels):
        raise ValueError(f"Path {labels} does not exist.")

    logger.info(f"Root path: {root}")
    logger.info(f"Labels path: {labels}")

    # get recursively all the subjects from the root folder
    subjects = [sub for sub in os.listdir(root) if sub.startswith("sub-")]

    # Get the training and validation splits
    # Note: we are doing SSL pre-training, so we don't need test set
    tr_subs, val_subs = train_test_split(subjects, test_size=val_ratio, random_state=args.seed)

    # recursively find the spinal cord segmentation files under 'derivatives/labels' for training and validation
    # subjects
    tr_seg_files = [str(path) for sub in tr_subs for path in
                    Path(labels).rglob(f"{sub}_{contrast}_{sc_fname_suffix}.nii.gz")]
    val_seg_files = [str(path) for sub in val_subs for path in
                     Path(labels).rglob(f"{sub}_{contrast}_{sc_fname_suffix}.nii.gz")]

    # update the train and validation images dicts with the key as the subject and value as the path to the subject
    train_images.update({sub: os.path.join(root, sub) for sub in tr_seg_files})
    val_images.update({sub: os.path.join(root, sub) for sub in val_seg_files})

    logger.info(f"Found subjects in the training set: {len(train_images)}")
    logger.info(f"Found subjects in the validation set: {len(val_images)}")

    # keys to be defined in the dataset_0.json
    params = {}
    params["dataset_name"] = dataset_name
    params["contrast"] = contrast
    params["labels"] = {
        "0": "background",
        "1": "sc-seg"
    }
    params["modality"] = {
        "0": "MRI"
    }
    params["numTraining"] = len(train_images)
    params["numValidation"] = len(val_images)
    params["seed"] = args.seed
    params["tensorImageSize"] = "3D"

    train_images_dict = {"training": train_images}
    val_images_dict = {"validation": val_images}

    all_images_list = [train_images_dict, val_images_dict]

    for images_dict in all_images_list:

        for name, images_list in images_dict.items():

            temp_list = []
            for label in images_list:

                temp_data_t2w = {}
                # create the image path by replacing the label path
                if dataset_name == 'spine-generic':
                    temp_data_t2w["image"] = label.replace(f'_{sc_fname_suffix}', '').replace('labels',
                                                                                              'data_preprocessed')
                else:
                    temp_data_t2w["image"] = label.replace(f'_{sc_fname_suffix}', '').replace('/derivatives/labels', '')

                # Spinal cord segmentation file
                temp_data_t2w["label_sc"] = label

                if os.path.exists(temp_data_t2w["label_sc"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)
                else:
                    logger.info(f"Either image/label does not exist.")

            params[name] = temp_list
            logger.info(f"Number of images in {name} set: {len(temp_list)}")

    final_json = json.dumps(params, indent=4, sort_keys=False)
    if not os.path.exists(path_out):
        os.makedirs(path_out, exist_ok=True)

    jsonFile = open(path_out + "/" + f"{datalist_fname}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()
    print(f"JSON file saved to {path_out}/{datalist_fname}.json")


if __name__ == "__main__":
    main()

