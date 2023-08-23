"""
Author: Mélanie Gaillochet
"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from Utils.load_utils import load_single_image, get_dict_from_config
from Utils.utils import natural_keys
from Data.transform import Preprocess


class MyDataset(Dataset):
    def __init__(self, data_folder, dataset_name, type='train', partition=3, kwargs={}, **transform_config):
        self.data_folder = data_folder
        self.dataset_name = dataset_name
        self.type = type
        self.partition = partition
        self.kwargs = kwargs
        self.transform_config = transform_config

        # We select the sample paths
        self.volume_folder_path = os.path.join(self.data_folder, self.dataset_name, 'preprocessed', type, 'data')
        with h5py.File(self.volume_folder_path + '.hdf5', 'r') as hf:
            self.volume_list = list(hf.keys())
        self.volume_list.sort(key=natural_keys)

        self.seg_folder_path = os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'label')
        with h5py.File(self.seg_folder_path + '.hdf5', 'r') as hf:
            self.seg_list = list(hf.keys())
        self.seg_list.sort(key=natural_keys)
        
        # We get the information on slices (patient, slice position, etc)
        slice_info_path =  os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'slice_info.json')
        self.slice_info = get_dict_from_config(slice_info_path)
        
        scan_info_path =  os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'scan_info.json')
        self.scan_info = get_dict_from_config(scan_info_path)
        
        # We get informationt to compute CRF loss
        if 'type' in self.kwargs.keys() and self.kwargs['type'] == 'crf':
            sigma = self.kwargs['sigma']
            partial_suffix = self.kwargs['partial_suffix']
            suffix = partial_suffix + '_sigma{}'.format(sigma)
    
            self.edge_weights_save_path = os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'all_weights_edges{}'.format(suffix))
            with h5py.File(self.edge_weights_save_path + '.hdf5', 'r') as hf:
                self.edge_weights_list = list(hf.keys())
            self.edge_weights_list.sort(key=natural_keys)

            self.start_edges = np.load( os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'start_edges{}_indices.npy'.format(partial_suffix)))
            self.end_edges = np.load( os.path.join(self.data_folder, self.dataset_name,  'preprocessed', type, 'end_edges{}_indices.npy'.format(partial_suffix)))

    def __len__(self):
        """We return the total number of samples"""
        return len(self.volume_list)
    
    def _get_partition(self, slice_position, num_slices):
        """
        We give the relative position of the slice in the volume with regards to the requested volume partition
        """
        _partition = (self.partition * slice_position) / num_slices
        partition = np.ceil(_partition)
        return int(partition)

    def _load_edge_weights(self, idx):
        all_weights_arr = load_single_image(self.edge_weights_save_path, self.edge_weights_list, idx)
        return all_weights_arr

    def __getitem__(self, idx):
        """We generate one sample of data"""
        # We load the volume and segmentation samples
        img = load_single_image(self.volume_folder_path, self.volume_list, idx)
        target = load_single_image(self.seg_folder_path, self.seg_list, idx)
        scan_name = self.slice_info[str(idx)]['scan_name']
        slice_position = self.slice_info[str(idx)]['slice_position']
        partition = self._get_partition(slice_position, self.slice_info[str(idx)]['num_slices'])
        
        # We add channel dimension to target
        target = target[None]

        sample = {'data': torch.from_numpy(img).float(), 
                'label': torch.from_numpy(target).float(),
                'scan_name': scan_name,
                'position': slice_position,
                'num_slices_in_volume': self.slice_info[str(idx)]['num_slices'],
                'partition': partition    
                }
        
        # We apply the transformations
        transf = Preprocess(**self.transform_config)
        transf_data = transf(sample)
        sample['data'] = transf_data['data']
        sample['label'] = transf_data['label']

        if 'type' in self.kwargs.keys() and self.kwargs['type'] == 'crf':
            sample['start_edges'] = self.start_edges
            sample['end_edges'] = self.end_edges
            sample['edge_weights'] = self._load_edge_weights(idx)

        sample["idx"] = idx

        return sample
