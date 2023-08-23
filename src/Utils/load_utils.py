"""
Author: Mélanie Gaillochet
"""
import json
import yaml
import os

import h5py
import nibabel as nib
import numpy as np

import pytorch_lightning as pl

from Configs.config import config_folder


def _read_json_file(file_path):
    """
    We are reading the json file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config json file provided
    with open(file_path, 'r') as file:
        output_dict = json.load(file)
    return output_dict


def _read_yaml_file(file_path):
    """
    We are reading the yaml file and returning a dictionary
    :param file_path:
    :return:
    """
    # We parse the configurations from the config json file provided
    with open(file_path, 'r') as file:
        output_dict = yaml.safe_load(file)
    return output_dict


def get_dict_from_config(config_filename):
    """
    Get the config file (json or yaml) as a dictionary
    :param config_filename: name of config file (located in config folder)
    :return: dictionary
    """
    config_filepath = os.path.join(config_folder, config_filename)
    
    if config_filepath.endswith('.json'):
        config_dict = _read_json_file(config_filepath)
    elif config_filepath.endswith('.yaml'):
        config_dict = _read_yaml_file(config_filepath)

    return config_dict


def load_single_image(folder_path, filename_list, idx):
    """
    We load the data and label for the specific list index given
    :param folder_path:
    :param filename_list:
    :param idx:
    :return: image (array)
    """
    cur_volume_path = os.path.join(folder_path, filename_list[idx])
    ending = cur_volume_path.rpartition('.')[2]

    if ending == 'nii':
        inputImage = nib.load(cur_volume_path)
        img = inputImage.get_data()
        img = np.array(img)

    elif not os.path.isdir(folder_path):
        with h5py.File(folder_path + '.hdf5', 'r') as hf:
            img = hf[filename_list[idx]][:]

    return img


def save_hdf5(data, img_idx, dest_file):
    """
    We are saving an hdf5 object
    :param data:
    :param filename:
    :return:
    """
    with h5py.File(dest_file, "a", libver='latest', swmr=True) as hf:
        hf.swmr_mode = True
        hf.create_dataset(name=str(img_idx), data=data, shape=data.shape, dtype=data.dtype)


def create_unexisting_folder(dir_path):
    """
    We create a folder with the given path.
    If the folder already exists, we add '_1', '_2', ... to it
    :param dir_path:
    """
    i = 0
    created = False
    path = dir_path
    while not created:
        try:
            os.makedirs(path)
            created = True
        except OSError or FileExistsError:
            i += 1
            path = dir_path + '_' + str(i)
            # print(path)
    return path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def save_to_logger(logger, type, data, name, epoch=None):
    """
    We save data to the given logger

    Args:
        logger (tensorboard logger, comet ml logger, etc.): logger of pytorch lightning trainer
        type (str): 'metric' (to save scalar), 'list'
        data (any): what we want to save
        name (str): name to use when saving data
        epoch (int): if we want to assign the data to a given epoch
    """
    if type == 'metric':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_metric(name, data)
        else:
            # Saving on TensorBoardLogger
            logger.experiment.add_scalar(name, data, epoch)
            
    elif type == 'list':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_other(name, data)
        else:
            # Saving on TensorBoardLogger as scalar, with epoch as indice in list
            for i in range(len(data)):
                logger.experiment.add_scalar(name, data[i], i)
                
    elif type == 'hyperparameter':
        if isinstance(logger, pl.loggers.CometLogger):
            # Saving on comet_ml
            logger.experiment.log_parameter(name, data)
        else:
            # Saving on TensorBoardLogger
            logger.log_hyperparams({name: data})
