import os

from monai.data import load_decathlon_datalist


def load_data(data_root, json_path, logdir_path, is_segmentation=False):
    """
    Load data from the json file and return the training and validation data
    """
    if os.path.exists(logdir_path) is False:
        os.mkdir(logdir_path)

    train_list = load_decathlon_datalist(
        base_dir=data_root, data_list_file_path=json_path, is_segmentation=is_segmentation, data_list_key="training"
    )

    val_list = load_decathlon_datalist(
        base_dir=data_root, data_list_file_path=json_path, is_segmentation=is_segmentation, data_list_key="validation"
    )

    #train_data[0]

    return train_list, val_list
