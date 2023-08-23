"""
Author: Mélanie Gaillochet
"""
import os
import random
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from pytorch_lightning import seed_everything

from Data.dataset import MyDataset
from Data.sampler import InfiniteSubsetRandomSampler, SubsetSequentialSampler
from torch.utils.data import SubsetRandomSampler
from Utils.load_utils import get_dict_from_config


class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir: str,
                 dataset_name: str,
                 batch_size: int = 4,
                 val_batch_size: int = 0,
                 num_workers: int = 2,
                 train_transform_config: list = {},
                 test_transform_config: list = {},
                 val_indices: Union[int, list] = None,   # Number or list of indices to use for validation. If we want metrics on validation volumes, the indices must be sequential.
                 train_indices: list = [],   # Indices of samples to use for training. If empty list, we take all indices that are not validation
                 num_indices = 'all',   # If we want to use a subset of training indices for training
                 train_loader_length: int = 0,   # Size/number of samples in training loader (infinite subset)
                 dataset_kwargs = {},   # kwargs for dataset (ie: for CRF)
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.val_indices = val_indices
        self.batch_size = batch_size
        self.val_batch_size = batch_size if val_batch_size == 0 else val_batch_size
        self.num_workers = num_workers
        self.num_indices = num_indices
        self.train_indices = train_indices
        self.train_loader_length = train_loader_length
        
        self.train_transform_config = train_transform_config
        self.test_transform_config = test_transform_config
        
        self.dataset_kwargs = dataset_kwargs
        if self.dataset_kwargs != {}:
            # if we need the partial suffix for CRF based on data base transformations (ie: '_centercrop384_resize256')
            self.dataset_kwargs['partial_suffix'] = ''.join('_{}{}'.format(k.replace('_', ''), v[0]) if (type(v)==str and len(v) == 2) \
                                                            else '' \
                                                                for k, v in self.test_transform_config.items())

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            # We create 2 datasets, one with augmentations (ds_train) and one without (ds_val)
            self.ds_train = MyDataset(self.data_dir, self.dataset_name, type='train', 
                                      kwargs=self.dataset_kwargs, **self.train_transform_config)
            
            self.ds_val = MyDataset(self.data_dir, self.dataset_name, type='train', 
                                      kwargs=self.dataset_kwargs, **self.test_transform_config)
            
            # We get train and val indices
            _all_indices = np.arange(len(self.ds_train))
            all_indices = _all_indices.tolist()
            if isinstance(self.val_indices, list):
                self.val_indices = self.val_indices
            elif isinstance(self.val_indices, int):
                self.val_indices = random.sample(all_indices, self.val_indices)
            all_train_indices = np.setdiff1d(all_indices, self.val_indices)
            
            # If the training indices are not a list of given indices, we take all indices that are not validation
            if len(self.train_indices) == 0:
                self.train_indices = all_train_indices.tolist()
            # We can also take a random subset of indices
            elif self.num_indices != 'all':
                self.train_indices = np.random.choice(self.train_indices, size=self.num_indices, replace=False)

            # We get the rest of indices not used for training
            self.rest_indices = np.setdiff1d(all_train_indices, self.train_indices)
            
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.ds_test = MyDataset(self.data_dir, self.dataset_name, type='test', 
                                     kwargs=self.dataset_kwargs, **self.test_transform_config)

    def train_dataloader(self):
        train_sampler =  InfiniteSubsetRandomSampler(self.ds_train, self.train_indices,
                                                     shuffle=True, length=self.train_loader_length)
        self.train_loader = DataLoader(self.ds_train, 
                          sampler=train_sampler,
                          batch_size=self.batch_size,
                          drop_last=False, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
        return self.train_loader
                
    def val_dataloader(self):
        val_sampler = SubsetSequentialSampler(self.val_indices)
        self.val_loader = DataLoader(self.ds_val, 
                          sampler=val_sampler,
                          batch_size=self.val_batch_size,
                          drop_last=False, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = DataLoader(self.ds_test, 
                          batch_size=1, # We use a batch size of 1 to compute the 3D results more easily
                          drop_last=False, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
        return self.test_loader
    
    def unlabeled_dataloader(self):
        unlabeled_sampler = SubsetSequentialSampler(self.rest_indices)
        self.unlabeled_loader = DataLoader(self.ds_val, 
                          sampler=unlabeled_sampler,
                          batch_size=2*self.batch_size,
                          drop_last=False, 
                          num_workers=self.num_workers,
                          pin_memory=True
                          )
        return self.unlabeled_loader
