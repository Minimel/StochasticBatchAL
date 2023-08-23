"""
Author: Mélanie Gaillochet
"""
from comet_ml import Experiment

import os
from datetime import datetime
import numpy as np
import json
import time
from argparse import ArgumentParser
import warnings
from typing import Union, List

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CometLogger

from Data.datamodule import MyDataModule
from Data.dataset import MyDataset
from Models.UNet import UNet
from Models.UNet_LearningLoss import UNetLearningLoss
from Utils.load_utils import get_dict_from_config, NpEncoder, save_to_logger
from Utils.utils import update_config_from_args
from Sampling.sampling import sample_new_indices


############ Training UNEt ifor one AL cycle#############
def train_model(model_config, train_config, data_config, data_dir, logger_config,
                checkpoint_path=None, gpu_devices=1, seed=42):
    """_summary_

    Args:
        model_config (dict): 
        train_config (dict): 
        data_dir (str): path to data folder
        dataset_name (str): name of dataset located in data_dir
        val_indices (list): list of indices of data for validation
    """
    if logger_config['name'] == 'comet':
        # We set comet_ml logger
        logger = CometLogger(
        api_key=logger_config['api_key'],
        workspace=logger_config['workspace'],
        project_name=logger_config['project_name'], 
        experiment_name=logger_config['experiment_name'],
        )
    else:
        logger = True  # Default logger (TensorBoard)

    seed_everything(seed, workers=True)

    # We set up the data module
    AL_data_module = MyDataModule(
                data_dir = data_dir,
                dataset_name = data_config["dataset_name"],
                batch_size = train_config["batch_size"],
                val_batch_size = 1,
                num_workers = train_config["num_workers"],
                # We combine the base data transformations with the transformations for the task during training. 
                # We only keep the base transformations (center_crop and resize) at test time
                train_transform_config = {**data_config["transform"],  **train_config["transform"]},   
                test_transform_config = data_config["transform"],   
                val_indices = data_config["val_indices"],
                train_indices = train_config["train_indices"],
                num_indices = len(train_config["train_indices"]),
                train_loader_length = train_config["num_steps_per_epoch"] * train_config["batch_size"],
    )

    # We set up the UNet for training
    if train_config['additional_params'] is None:
        full_model = UNet(
                        per_device_batch_size=train_config["batch_size"],
                        num_devices = 1,
                        lr=train_config["lr"],
                        weight_decay=train_config["weight_decay"],
                        model_config=model_config,
                        sched_config=train_config["sched"],
                        loss_config=train_config["loss"],
                        val_plot_slice_interval=train_config["val_plot_slice_interval"],
                        seed = seed
                        )
    
    # We initialize and additional module when training with Learning Loss
    elif train_config['additional_params']['module'] == 'LearningLoss': 
        full_model = UNetLearningLoss(
                        per_device_batch_size=train_config["batch_size"],
                        num_devices = 1,
                        lr=train_config["lr"],
                        weight_decay=train_config["weight_decay"],
                        model_config=model_config,
                        sched_config=train_config["sched"],
                        loss_config=train_config["loss"],
                        val_plot_slice_interval=train_config["val_plot_slice_interval"],
                        seed = seed,
                        **train_config['additional_params']
                        )
  
    # We set-up the trainer
    trainer = pl.Trainer(
            deterministic=True,
            max_epochs=train_config["num_epochs"],
            precision=16,
            devices=gpu_devices,
            accelerator='gpu',
            sync_batchnorm=True,
            log_every_n_steps=1,
            check_val_every_n_epoch=1,
            callbacks=[ModelCheckpoint(dirpath=os.path.join(checkpoint_path, '{}labeled'.format(len(train_config["train_indices"]))),
                        monitor='val/loss', mode='min', every_n_epochs=10, save_last=True)],
            logger=logger,
            num_sanity_val_steps=0)
    save_to_logger(trainer.logger, 'hyperparameter', checkpoint_path, 'checkpoint_path')

    # We train our model
    train_start_time = time.time()
    trainer.fit(full_model, AL_data_module)

    # We keep track of runtime
    train_time = time.time() - train_start_time
    save_to_logger(trainer.logger, 'hyperparameter', np.round(train_time/60, 2), 'train time (min)')

    # We evaluate our trained model on the test set
    AL_data_module.setup(stage = 'test')    
    metric_dict = trainer.test(full_model, datamodule=AL_data_module)
    
    # We save the metrics
    results_path = os.path.join(checkpoint_path, '{}labeled'.format(len(train_config["train_indices"])), 'metrics.json')
    with open(results_path, 'w') as file:
        json.dump(metric_dict, file, indent=4, cls=NpEncoder)
    
    return trainer, full_model, AL_data_module, train_time
    

def get_all_train_indices(data_dir, dataset_name, val_indices, type='train'):
    ds_train = MyDataset(data_dir, dataset_name, type=type)

    # We get train and val indices
    _all_indices = np.arange(len(ds_train))
    all_indices = _all_indices.tolist()
    if isinstance(val_indices, list):
        val_indices = val_indices
    elif isinstance(val_indices, int):
        val_indices = np.random.choices(all_indices, size=val_indices, replace=False)
    all_train_indices = np.setdiff1d(all_indices, val_indices)
    
    return all_train_indices


def get_kwargs_sampling(sampling_config, data_dir, data_config, datamodule, seed, device):
    """
    We get the kwargs needed for the given sampling strategy
    """
    strategy = sampling_config['strategy']
    sampling_params = sampling_config['sampling_params']

    kwargs = {}
    kwargs['seed'] = seed
    kwargs['device'] = device
    all_train_indices = get_all_train_indices(data_dir, data_config["dataset_name"], data_config["val_indices"], type='train')
    kwargs['unlabeled_indices'] = [idx for idx in all_train_indices if idx not in datamodule.train_indices]

    if strategy == 'RS':
        pass

    elif strategy == 'Coreset':
        kwargs['labeled_dataloader'] = datamodule.labeled_dataloader()

    elif strategy == 'Entropy':
        pass

    # Dropout and TTA require parameters on the multiple inferences and the uncertainty metric
    elif strategy == 'Dropout':
       kwargs['num_inferences'] = sampling_params['num_inferences']
       kwargs['uncertainty_metric'] = sampling_params['uncertainty_metric']
       kwargs['alpha'] = sampling_params['alpha']
    
    elif strategy == 'TTA':
       kwargs['num_inferences'] = sampling_params['num_inferences']
       kwargs['uncertainty_metric'] = sampling_params['uncertainty_metric']
       kwargs['alpha'] = sampling_params['alpha']
       kwargs['transform_config'] = sampling_params['transform']
       
    elif strategy == 'LearningLoss':
        kwargs['num_subset_indices'] = sampling_params['num_subset_indices']
       
    return kwargs


if __name__ == "__main__":
    parser = ArgumentParser()
    # These are the paths to the data and output folder
    parser.add_argument('--data_dir', default='/home/ar32500@ens.ad.etsmtl.ca/moneta_data/users/melanie/data/', type=str, help='Directory for pre-downloaded imagenet or cache for CIFAR10.')
    parser.add_argument('--output_dir', default='/home/ar32500@ens.ad.etsmtl.ca/moneta_data/users/melanie/output/', type=str, help='Directory for output run')
    
    # These are config files located in src/Config
    parser.add_argument('--data_config',  type=str, default='data_config/data_config_hippocampus.yaml')
    parser.add_argument('--model_config', type=str, default='model_config.yaml')
    parser.add_argument('--train_config', type=str, default='train_config.yaml')
    parser.add_argument('--sampling_config', type=str, default='sampling_config/Entropy_SB.yaml')
    parser.add_argument('--logger_config', type=str, default='logger_config.yaml')

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_gpu', default=1, help='number of GPU devices to use')
    parser.add_argument('--gpu_idx', default=None, type=int, help='otherwise, gpu index, if we want to use a specific gpu')

    # Training hyper-parameters that we should change according to the dataset
    parser.add_argument('--model__out_channels', type=int, help='number of output channels')
    parser.add_argument('--train__train_indices',type=int, nargs='+', help='indices of training data for Segmentation task')
    parser.add_argument('--train__loss__normalize_fct', type=str)
    parser.add_argument('--train__loss__n_classes', type=int)
    parser.add_argument('--train__val_plot_slice_interval', default=1, type=int, help='interval between 2 slices in a volume to be plotted and saved during validation')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    # We set the gpu devices (either a specific gpu or a given number of available gpus)
    gpu_devices = [args.gpu_idx] if args.gpu_idx is not None else args.num_gpu
    print('gpu_devices {}'.format(gpu_devices))

    # We create a checkpoint path
    start_time = datetime.today()
    log_id = '{}_{}h{}min'.format(start_time.date(), start_time.hour, start_time.minute)
    print('args.train__train_indices {}'.format(args.train__train_indices))
    red_init_labels =  str(args.train__train_indices).replace('[', '').replace(']', '').replace(',', '-').replace(' ', '')[:40]
    experiment_name = args.sampling_config.replace('sampling_config/', '').replace('.yaml', '')
    checkpoint_path = os.path.join(args.output_dir, log_id, experiment_name + '_seed{}'.format(args.seed))
    print('checkpoint_path: {}'.format(checkpoint_path))
    
    # We extract the configs from the file names
    train_config = get_dict_from_config(args.train_config)
    sampling_config = get_dict_from_config(args.sampling_config)
    data_config = get_dict_from_config(args.data_config)
    if data_config["transform"] is None:
        data_config["transform"] = {} 
    model_config = get_dict_from_config(args.model_config)
    logger_config = get_dict_from_config(args.logger_config)
    # We update the model and logger config files with the command-line arguments
    model_config = update_config_from_args(model_config, args, 'model')
    logger_config = update_config_from_args(logger_config, args, 'logger')
    
    logger_config['experiment_name'] = experiment_name  # Useful if we use comet logger
    
    # We keep track of initial labelled set (AL cycle 0) 
    initial_labeled_set = args.train__train_indices.copy()

    # We iterate through all AL cycles
    num_al_cycles = train_config["sampling"]["num_cycles"]
    for al_cycle in range(num_al_cycles):                
        # We update the task config with the input parameters and new train indices
        train_config = update_config_from_args(train_config, args, 'train')
        if train_config["transform"] is None:
            train_config["transform"] = {} 
        print(train_config)
        
        print('\n ### al cycles {} with {} samples #### \n'.format(al_cycle, len(train_config["train_indices"])))
        # We train the task model
        trainer, task_model, AL_data_module, train_time = train_model(model_config, train_config, data_config,
                                                                          args.data_dir, logger_config,
                                                                          checkpoint_path, gpu_devices, args.seed)
        # We save initial labelled set and setup (easier to keep track of experiments)
        save_to_logger(trainer.logger, 'hyperparameter', initial_labeled_set, 'initial_labeled_set')
        save_to_logger(trainer.logger, 'hyperparameter', args.logger__setup, 'setup')
        
        # We sample additional indices and add the new samples to the training set
        kwargs = get_kwargs_sampling(sampling_config, args.data_dir, data_config, AL_data_module, args.seed, 'cuda:{}'.format(trainer.device_ids[0]))
        new_indices, sampling_time = sample_new_indices(sampling_config, budget=train_config["sampling"]["budget"], unlabeled_dataloader=AL_data_module.unlabeled_dataloader(), trainer=trainer, **kwargs)

        # We save train and sampling time
        time_dict = {
            'train time (min)': np.round(train_time/60, 2),
            'sampling time (min)': np.round(sampling_time/60, 2)
            }
        results_path = os.path.join(checkpoint_path, '{}labeled'.format(len(train_config["train_indices"])), 'time.json')
        with open(results_path, 'w') as file:
            json.dump(time_dict, file, indent=4, cls=NpEncoder)

        args.train__train_indices += new_indices

