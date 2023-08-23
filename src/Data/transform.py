"""
Author: Mélanie Gaillochet
"""
import random
from random import random
import numpy as np
from typing import Any, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

import kornia as K
from kornia.augmentation import RandomGamma, RandomGaussianNoise, CenterCrop, RandomCrop, RandomRotation, \
RandomAffine, RandomVerticalFlip, RandomHorizontalFlip, Resize, AugmentationSequential
 

class Preprocess(nn.Module):
    """Module to perform preprocessing using Kornia on torch tensors.
    It is to be used when creating the Dataset (before data batch is formed)

    Note, intensity transformations are first applied, followed by geometric transformations.
    centre_crop and random_crop are applied before resizing and affine transformation.

    Args:
        data_keys (list): list of keys to be transformed.
            For example, if you want to transform the input and the mask, you should define data_keys = ["input", "mask"].
            The value "input" should always be in data_keys.
        args (dict): dictionary of parameters for the transformations.
            Accepted keys are "translation", "rotation", "center_crop", "random_crop", "scale", "resize", "gauss_noise",  "gamma_correction"
    """

    def __init__(self, data_keys=["input", "mask"], **args): 
        super().__init__()

        self.data_keys = data_keys

        # We check that we have inputed valid keys in args
        _all_transforms = ["translation", "rotation", "center_crop", "random_crop", "scale", "resize", "gauss_noise",  "gamma_correction"]
        valid_keys = set(args.keys()).intersection(_all_transforms)
        assert len(list(valid_keys)) == len(args.keys()) 
        
        ### Intensity transformations
        # Gamma correction
        # Note: gamma correction should be before adding gaussian noise
        gamma = args["gamma_correction"] if "gamma_correction" in args else (1,1)
        p_gamma = 0.5 if "gamma_correction" in args else 0

        # Gaussian noise
        gauss_std = args["gauss_noise"]["std"] if "gauss_noise" in args else 1
        p_gauss_noise = 0.5 if "gauss_noise" in args else 0

        ### Geometric transformations
        # center crop
        center_crop_size = tuple(args["center_crop"]) if "center_crop" in args else 0
        p_center_crop = 1 if "center_crop" in args else 0
        
        # random crop
        random_crop_size = tuple(args["random_crop"]) if "random_crop" in args else 0
        p_random_crop = 0.5 if "random_crop" in args else 0

        # # Affine transform (shift, scale, rotate, shear)
        degrees = args["rotation"] if "rotation" in args else 0
        translate = args["translation"] if "translation" in args else None
        scale = args["scale"] if "scale" in args else None
        p_affine = 1 if ("rotation" in args) or ("translation" in args) or ("scale" in args) else 0

        # resize
        resize = args["resize"] if "resize" in args else 0
        p_resize = 1 if "resize" in args else 0
        
        self.gamma = RandomGamma(gamma=gamma, p=p_gamma, keepdim=True)
        self.gaussian_noise = RandomGaussianNoise(std=gauss_std, p=p_gauss_noise, keepdim=True)
        self.center_crop = CenterCrop(size=center_crop_size, p=p_center_crop, keepdim=True)
        self.random_crop = RandomCrop(size=random_crop_size, p=p_random_crop, keepdim=True)
        self.resize = Resize(size=resize, p=p_resize, keepdim=True)
        self.random_affine = RandomAffine(degrees=degrees, translate=translate, scale=scale, p=p_affine, keepdim=True)

        self.center_crop_nearest = CenterCrop(size=center_crop_size, p=p_center_crop, keepdim=True, resample='nearest')
        self.random_crop_nearest = RandomCrop(size=random_crop_size, p=p_random_crop, keepdim=True, resample='nearest')
        self.resize_nearest = Resize(size=resize, p=p_resize, keepdim=True, resample='nearest')
        self.random_affine_nearest = RandomAffine(degrees=degrees, translate=translate, scale=scale, p=p_affine, keepdim=True, resample='nearest')

    @torch.no_grad()
    def forward(self, sample: dict) -> dict:
        # Extract keys for image and mask from self.data_keys
        if self.data_keys == ["input", "mask"]:
            new_input = self.random_affine(
                            self.resize(
                                self.random_crop(
                                    self.center_crop(
                                        self.gaussian_noise(
                                            self.gamma(
                                                sample['data']
                                            )
                                        )
                                    )
                                )
                            )
                        )
            # We infer geometry params for the mask
            new_label = self.random_affine_nearest(
                            self.resize_nearest(
                                self.random_crop_nearest(
                                    self.center_crop_nearest(
                                        sample['label'],
                                    self.center_crop._params),
                                self.random_crop._params),
                            self.resize._params),
                        self.random_affine._params)
            
            return  {'data': new_input, 'label': new_label}
        
        elif self.data_keys == ["input"]:
            new_input = self.random_affine(
                            self.resize(
                                self.random_crop(
                                    self.center_crop(
                                        self.gaussian_noise(
                                            self.gamma(
                                                sample['data']
                                            )
                                        )
                                    )
                                )
                            )
                        )
            return {'data': new_input}


# run tests of augmentation module
if __name__ == "__main__":
        
    def test_Preprocess():
        """ Test Preprocess class on a tensor """

        ### Test 1 ###
        transform_config = {"rotation": 10, "translation": [0.1,0.1], "scale": [0.9,1.1], 
                            "center_crop": [150, 150], "random_crop": (100, 100), 
                            "resize": [200, 200], 
                            "gauss_noise": {"std": 0.1}, "gamma_correction": [0.9,1.1]}
        preprocess = Preprocess(data_keys=["input", "mask"], **transform_config)
        input = torch.rand(1, 200,200).float()
        label = torch.randint(0, 4,(1, 200,200)).float()
        sample = {'data': input, 'label': label}
        new_sample = preprocess(sample)

        # Assert input and output shape is the same (since we are resizing)
        assert new_sample['data'].shape == (1, 200,200), "Input and output data shape should be the same"
        assert new_sample['label'].shape == (1, 200,200), "Input and output label shape should be the same"
        
        # Assert that new_sample data and label are different from sample data and label
        assert not torch.equal(new_sample['data'], sample['data']), "Input and output data should be different"
        assert not torch.equal(new_sample['label'], sample['label']), "Input and output label should be different"


        ### Test 2 ###
        transform_config = {"center_crop": (150, 150)}
        preprocess = Preprocess(data_keys=["input", "mask"], **transform_config)
        new_sample = preprocess(sample)

        # Assert input and output shape follows center crop
        assert new_sample['data'].shape == (1, 150, 150), "Input and output data shape should be the same"
        assert new_sample['label'].shape == (1, 150, 150), "Input and output label shape should be the same"

    test_Preprocess()
 