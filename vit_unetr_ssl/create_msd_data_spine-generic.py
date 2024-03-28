"""
For preprocessed spine-generic dataset, use:
    --path-data ~/data-multi-subject/derivatives/data_preprocessed
"""

import os
import re
import json
from tqdm import tqdm
import argparse
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split


def get_parser():
    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for DCM and SCI lesions dataset.')

    parser.add_argument('--path-data', nargs='+', required=True, type=str,
                        help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-out', type=str, required=True,
                        help='Path to the output directory where dataset json is saved')
    parser.add_argument('--split', nargs='+', type=float, default=[0.8, 0.2],
                        help='Ratios of training and validation 0-1. '
                             'Example: --split 0.8 0.2')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")

    return parser


def main():
    args = get_parser().parse_args()

    train_ratio, val_ratio = args.split
    seed = args.seed

    # spine-generic
    contrast = 'T2w'
    sc_fname_suffix = 'label-SC_seg'
    datalist_fname = f"dataset_spine-generic_seed{seed}"

    # Check if dataset paths exist
    for path in args.path_data:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")

    all_subjects, train_images, val_images, test_images = [], {}, {}, {}

    # loop over the datasets
    for idx, dataset in enumerate(args.path_data, start=1):
        root = Path(dataset)
        # spine-generic
        labels = Path(dataset.replace('data_preprocessed', 'labels'))

        # get recursively all the subjects from the root folder
        subjects = [sub for sub in os.listdir(root) if sub.startswith("sub-")]

        # add to the list of all subjects
        all_subjects.extend(subjects)

        # Get the training and validation splits
        # Note: we are doing SSL pre-training, so we don't need test set
        tr_subs, val_subs = train_test_split(subjects, test_size=val_ratio, random_state=args.seed)

        # recursively find the SC files for training and test subjects
        tr_seg_files = [str(path) for sub in tr_subs for path in
                           Path(labels).rglob(f"{sub}_*{contrast}_{sc_fname_suffix}.nii.gz")]
        val_seg_files = [str(path) for sub in val_subs for path in
                            Path(labels).rglob(f"{sub}_*{contrast}_{sc_fname_suffix}.nii.gz")]

        # update the train and test images dicts with the key as the subject and value as the path to the subject
        train_images.update({sub: os.path.join(root, sub) for sub in tr_seg_files})
        val_images.update({sub: os.path.join(root, sub) for sub in val_seg_files})

    logger.info(f"Found subjects in the training set (combining all datasets): {len(train_images)}")
    logger.info(f"Found subjects in the validation set (combining all datasets): {len(val_images)}")
    logger.info(
        f"Found subjects in the test set (combining all datasets): {len([sub for site in test_images.values() for sub in site])}")

    # keys to be defined in the dataset_0.json
    params = {}
    params["labels"] = {
        "0": "background",
        "1": "sc-seg"
    }
    params["modality"] = {
        "0": "MRI"
    }
    params["name"] = "spine-generic"
    params["numTraining"] = len(train_images)
    params["numValidation"] = len(val_images)
    params["seed"] = args.seed
    params["tensorImageSize"] = "3D"

    train_images_dict = {"training": train_images}
    val_images_dict = {"validation": val_images}
    test_images_dict = {}
    for site, images in test_images.items():
        temp_dict = {f"test_{site}": images}
        test_images_dict.update(temp_dict)

    all_images_list = [train_images_dict, val_images_dict, test_images_dict]

    for images_dict in all_images_list:

        for name, images_list in images_dict.items():

            temp_list = []
            for subject_no, label in enumerate(images_list):

                temp_data_t2w = {}
                # spine-generic
                temp_data_t2w["image"] = label.replace(f'_{sc_fname_suffix}', '').replace('labels', 'data_preprocessed')

                temp_data_t2w["label"] = label

                if os.path.exists(temp_data_t2w["label"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)
                else:
                    logger.info(f"Either image/label does not exist.")

            params[name] = temp_list
            logger.info(f"Number of images in {name} set: {len(temp_list)}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    jsonFile = open(args.path_out + "/" + f"{datalist_fname}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()
    print(f"JSON file saved to {args.path_out}/{datalist_fname}.json")


if __name__ == "__main__":
    main()

