"""
Author: Mélanie Gaillochet
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils.uncertainty_utils import weighted_jsd
from Utils.plot_utils import plot_all_uncertain_samples_from_lists


def apply_dropout(module):
    """
    This function activates dropout modules. Dropout module should have been defined as nn.Dropout
    """
    if type(module) == nn.Dropout:
        module.train()


class DropoutSampler:
    """
    Returns 2 lists of indices and corresponding uncertainty (based on mean uncertainty over x inferences with dropout), and sampling time
    """

    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        self.model = self.trainer.model
        self.device = kwargs.get('device')

        self.model.to(self.device)

        # We get the parameters for dropout
        self.num_inferences = kwargs.get('num_inferences')
        self.uncertainty_metric = kwargs.get('uncertainty_metric')
        self.alpha = kwargs.get('alpha')

    def get_uncertainty(self, unlabeled_dataloader):
        """ We sample the images with mean entropy on output softmax"""

        sampling_start_time = time.time()

        self.model.eval()

        uncertainty_map_list = []
        indice_list = []
        data_list = []
        logits_list = []
        target_list = []

        for batch_idx, batch in enumerate(unlabeled_dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(self.device)

                # We activate dropout
                self.model.apply(apply_dropout)

                # Compute probability map over multiple inferences with dropout enabled
                prob_dropout_list = [F.softmax(self.model(x)[0], dim=1) for _ in range(self.num_inferences)]
                _prob_dropout_tensor = torch.stack(prob_dropout_list, dim=0)  # (num_inferences, BS, C, H, W)
                prob_dropout_tensor = _prob_dropout_tensor.permute(1, 0, 2, 3, 4) # (BS, num_inferences, C, H, W)

                if self.uncertainty_metric == 'JSD':
                    # Calculate the JSD between the averaged predictions and the original predictions
                    cur_uncertainty_map = weighted_jsd(prob_dropout_tensor, self.alpha)  # shape (BS, H,  W)
                
                # We keep track of the uncertainty (of each pixel and for the entire image)
                uncertainty_map_list.extend(cur_uncertainty_map.detach().cpu().numpy())

                # We get output without dropout
                self.model.eval()
                logits, _ = self.model(x)

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
                                              self.budget, self.trainer, title='{} with Dropout of query images'.format(self.uncertainty_metric), 
                                              model_out_channels=self.model.out_channels)

        return indice_list, mean_uncertainty_list, sampling_time
