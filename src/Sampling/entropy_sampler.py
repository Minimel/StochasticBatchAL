"""
Author: Mélanie Gaillochet
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from Utils.uncertainty_utils import entropy
from Utils.utils import normalize
from Utils.plot_utils import  plot_all_uncertain_samples_from_lists


class EntropySampler:
    """
    Returns 2 lists of indices and corresponding uncertainty (based on mean entropy over pixels), and sampling time
    """

    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        self.model = self.trainer.model
        self.device = kwargs.get('device')

        self.model.to(self.device)

    def get_uncertainty(self, unlabeled_dataloader):
        """ We sample the images with mean entropy on output softmax"""

        sampling_start_time = time.time()

        self.model.eval()

        uncertainty_map_list = []
        indice_list = []
        data_list = []
        logits_list = []
        target_list = []

        # We iterate through the unlabeled dataloader
        for batch_idx, batch in enumerate(unlabeled_dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(self.device)

                logits, _ = self.model(x)

                # We get output probability and prediction
                prob = F.softmax(logits, dim=1)

                # We compute the entropy for each pixels of the image
                cur_uncertainty_map = entropy(prob, dim=1)   # shape (BS, H x W)
                
                # We keep track of the uncertainty (of each pixel and for the entire image)
                uncertainty_map_list.extend(cur_uncertainty_map.detach().cpu().numpy())

                # We keep track of all dataloader results
                indice_list.extend(img_idx.detach().cpu().numpy())
                data_list.extend(x.detach().cpu().numpy())
                logits_list.extend(logits.detach().cpu().numpy())
                target_list.extend(y.detach().cpu().numpy())

        uncertainty_map_array = np.stack(uncertainty_map_list, axis=0)
        mean_uncertainty_list = np.mean(uncertainty_map_array, axis=(1, 2))

        sampling_time = time.time() - sampling_start_time

        # We plot top uncertainty samples and their uncertainty map
        plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_map_list, 
                                              self.budget, self.trainer, title='Entropy of query images',
                                              model_out_channels=self.model.out_channels)

        return indice_list, mean_uncertainty_list, sampling_time
                                      