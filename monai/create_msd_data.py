import os
import re
import json
from tqdm import tqdm
import yaml
import argparse
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from utils import get_git_branch_and_commit


def get_parser():
    parser = argparse.ArgumentParser(description='Code for MSD-style JSON datalist for DCM and SCI lesions dataset.')

    parser.add_argument('--path-data', nargs='+', required=True, type=str, help='Path to BIDS dataset(s) (list).')
    parser.add_argument('--path-out', type=str, required=True,
                        help='Path to the output directory where dataset json is saved')
    parser.add_argument('--split', nargs='+', type=float, default=[0.7, 0.2, 0.1], 
                        help='Ratios of training, validation and test splits lying between 0-1. '
                        'Example: --split 0.7 0.2 0.1')
    parser.add_argument('--seed', default=42, type=int, help="Seed for reproducibility")
    parser.add_argument('--pathology', default='dcm', type=str, required=True,
                        help="Type of pathology in the dataset(s). Default: 'dcm' (for dcm-zurich-lesions). "
                        "Options: 'sci' (for sci lesions) ")

    return parser


def find_site_in_path(path, pathology='dcm'):
    """Extracts site identifier from the given path.

    Args:
    path (str): Input path containing a site identifier.

    Returns:
    str: Extracted site identifier or None if not found.
    """
    if pathology == 'dcm':
        # Find 'dcm-zurich-lesions' or 'dcm-zurich-lesions-20231115'
        match = re.search(r'dcm-zurich-lesions(-\d{8})?', path)
        return match.group(0) if match else None
    elif pathology == 'sci':
        # Find 'sci-zurich', 'sci-colorado', or 'sci-paris'
        match = re.search(r'sci-(zurich|colorado|paris)', path)
        return match.group(0) if match else None


def main():
    args = get_parser().parse_args()

    train_ratio, val_ratio, test_ratio = args.split
    seed = args.seed
    root = args.path_data
    pathology = args.pathology
    if pathology == 'dcm':
        lesion_fname_suffix = 'label-lesion'
        sc_fname_suffix = 'label-SC_mask-manual'
        datalist_fname = f"dataset_dcm_lesions_seed{seed}"
    elif pathology == 'sci':
        lesion_fname_suffix = 'T2w_lesion-manual'
        sc_fname_suffix = 'T2w_seg-manual'
        datalist_fname = f"dataset_sci_lesions_seed{seed}"

    # Check if dataset paths exist
    for path in args.path_data:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist.")

    # Get sites from the input paths
    sites = set(find_site_in_path(path, pathology) for path in args.path_data if find_site_in_path(path, pathology))

    all_subjects, train_images, val_images, test_images = [], {}, {}, {}
    # temp dict for storing dataset commits
    dataset_commits = {}

    # loop over the datasets
    for idx, dataset in enumerate(args.path_data, start=1):
        root = Path(dataset)

        # get the git branch and commit ID of the dataset
        dataset_name = os.path.basename(os.path.normpath(dataset))
        branch, commit = get_git_branch_and_commit(dataset)
        dataset_commits[dataset_name] = f"git-{branch}-{commit}"
        
        # get recursively all the subjects from the root folder
        subjects = [sub for sub in os.listdir(root) if sub.startswith("sub-")]

        # add to the list of all subjects
        all_subjects.extend(subjects)

        # Get the training and test splits
        tr_subs, te_subs = train_test_split(subjects, test_size=test_ratio, random_state=args.seed)
        if "sci-paris" in dataset:
            # add all test subjects to the training set
            tr_subs.extend(te_subs)
            te_subs = []
        tr_subs, val_subs = train_test_split(tr_subs, test_size=val_ratio / (train_ratio + val_ratio), random_state=args.seed)

        # recurively find the lesion files for training and test subjects)
        tr_lesion_files = [str(path) for sub in tr_subs for path in Path(root).rglob(f"{sub}_*{lesion_fname_suffix}.nii.gz")]
        val_lesion_files = [str(path) for sub in val_subs for path in Path(root).rglob(f"{sub}_*{lesion_fname_suffix}.nii.gz")]
        te_lesion_files = [str(path) for sub in te_subs for path in Path(root).rglob(f"{sub}_*{lesion_fname_suffix}.nii.gz")]
        
        # update the train and test images dicts with the key as the subject and value as the path to the subject
        train_images.update({sub: os.path.join(root, sub) for sub in tr_lesion_files})
        val_images.update({sub: os.path.join(root, sub) for sub in val_lesion_files})
        test_images.update({
            f"site_{idx}": {sub: os.path.join(root, sub) for sub in te_lesion_files}
        })
        # test_images.update({sub: os.path.join(root, sub) for sub in te_subs})

    # remove empty test sites
    test_images = {k: v for k, v in test_images.items() if v}

    logger.info(f"Found subjects in the training set (combining all datasets): {len(train_images)}")
    logger.info(f"Found subjects in the validation set (combining all datasets): {len(val_images)}")
    logger.info(f"Found subjects in the test set (combining all datasets): {len([sub for site in test_images.values() for sub in site])}")

    # # dump train/val/test splits into a yaml file
    # with open(f"data_split_{contrast}_{args.label_type}_seed{seed}.yaml", 'w') as file:
    #     yaml.dump({'train': train_subjects, 'val': val_subjects, 'test': test_subjects}, file, indent=2, sort_keys=True)

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "spine-generic-uncropped"
    params["labels"] = {
        "0": "background",
        "1": "soft-sc-seg"
        }
    params["license"] = "nk"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = "spine-generic"
    params["numTest"] = len([sub for site in test_images.values() for sub in site])
    params["numTraining"] = len(train_images)
    params["numValidation"] = len(val_images)
    params["seed"] = args.seed
    params["reference"] = "University of Zurich"
    params["tensorImageSize"] = "3D"

    train_images_dict = {"train": train_images}
    val_images_dict = {"validation": val_images}
    test_images_dict = {}
    for site, images in test_images.items():
        temp_dict = {f"test_{site}": images}
        test_images_dict.update(temp_dict)

    all_images_list = [train_images_dict, val_images_dict, test_images_dict]

    for images_dict in tqdm(all_images_list, desc="Iterating through train/val/test splits"):

        for name, images_list in images_dict.items():

            temp_list = []
            for subject_no, image in enumerate(images_list):

                temp_data_t2w = {}
                if pathology == 'dcm':
                    temp_data_t2w["image"] = image.replace('/derivatives/labels', '').replace(f'_{lesion_fname_suffix}', '')
                    temp_data_t2w["label-sc"] = image.replace(f'_{lesion_fname_suffix}', f'_{sc_fname_suffix}')
                elif pathology == 'sci':
                    temp_data_t2w["image"] = image.replace('/derivatives/labels', '').replace(f'_{lesion_fname_suffix}', '_T2w')
                    temp_data_t2w["label-sc"] = image.replace(f'_{lesion_fname_suffix}', f'_{sc_fname_suffix}')
                
                temp_data_t2w["label-lesion"] = image
                
                if os.path.exists(temp_data_t2w["label-lesion"]) and os.path.exists(temp_data_t2w["label-sc"]) and os.path.exists(temp_data_t2w["image"]):
                    temp_list.append(temp_data_t2w)
                else:
                    logger.info(f"Either Image/SC-Seg/Lesion-seg does not exist.")            
            
            params[name] = temp_list
            logger.info(f"Number of images in {name} set: {len(temp_list)}")

    final_json = json.dumps(params, indent=4, sort_keys=True)
    if not os.path.exists(args.path_out):
        os.makedirs(args.path_out, exist_ok=True)

    jsonFile = open(args.path_out + "/" + f"{datalist_fname}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()


if __name__ == "__main__":
    main()
    

