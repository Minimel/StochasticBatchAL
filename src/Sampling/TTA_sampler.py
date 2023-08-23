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
from Utils.TTA_utils import make_reverse_aug_prob_list, make_aug_data_list, make_reverse_aug_mask


class TTASampler:
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
        self.transform_config = kwargs.get('transform_config')

    def get_uncertainty(self, unlabeled_dataloader):
        """ We sample the images with mean entropy on output softmax"""

        sampling_start_time = time.time()

        self.model.eval()

        indice_list = []
        uncertainty_map_list = []
        data_list = []
        logits_list = []
        target_list = []
        
        multiple_data_list = []
        multiple_prob_list = []

        for batch_idx, batch in enumerate(unlabeled_dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(self.device)

                all_data_list = [x]
                
                # Debugging
                if batch_idx == (len(unlabeled_dataloader) - 1):
                    print('x.shape', x.shape)

                # We generate T1(x'), T2(x'), T3(x')
                x_aug_list, param_dic_list = make_aug_data_list(x, self.num_inferences - 1, self.transform_config)

                # We combine x, x' and all T(x') in a single tensor
                all_data_list.extend(x_aug_list)
                all_data = torch.cat(all_data_list, dim=0)
                all_logits, _ = self.model(all_data)
                logits, aug_logits = torch.split(all_logits, [len(x), len(x) * (self.num_inferences - 1)], dim=0)

                prob = F.softmax(logits, dim=1)

                # We reverse all S(T(x'))
                prob_TTA_list = make_reverse_aug_prob_list(aug_logits, param_dic_list, model_norm_fct=self.model.out_channels, data_shape=logits.shape)
                _prob_TTA_list = [prob] + prob_TTA_list

                _prob_TTA_tensor = torch.stack(_prob_TTA_list, dim=0)  # (num_inferences, BS, C, H, W)
                prob_TTA_tensor = _prob_TTA_tensor.permute(1, 0, 2, 3, 4) # (BS, num_inferences, C, H, W)
                
                # We compute the mask to compute the JSD
                aug_mask = make_reverse_aug_mask(logits.shape, param_dic_list, device=self.device)

                if self.uncertainty_metric == 'JSD':
                    # Calculate the JSD between the averaged predictions and the original predictions
                    cur_uncertainty_map = weighted_jsd(prob_TTA_tensor * aug_mask, self.alpha)  # shape (BS, H,  W)
                
                # We keep track of the uncertainty (of each pixel and for the entire image)
                uncertainty_map_list.extend(cur_uncertainty_map.detach().cpu().numpy())


                # We keep track of all dataloader results
                indice_list.extend(img_idx.detach().cpu().numpy())
                data_list.extend(x.detach().cpu().numpy())
                logits_list.extend(logits.detach().cpu().numpy())
                target_list.extend(y.detach().cpu().numpy())
                
                _multiple_data_tensor = torch.stack(all_data_list, dim=0)
                multiple_data_tensor = _multiple_data_tensor.permute(1, 0, 2, 3, 4)
                multiple_data_list.extend(multiple_data_tensor.detach().cpu().numpy())
                multiple_prob_list.extend(prob_TTA_tensor.detach().cpu().numpy())
        
    
        uncertainty_map_array = np.stack(uncertainty_map_list, axis=0)
        mean_uncertainty_list = np.mean(uncertainty_map_array, axis=(1, 2))

        sampling_time = time.time() - sampling_start_time

        # We plot top uncertainty samples and their uncertainty map
        plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_map_list, 
                                              self.budget, self.trainer, title='{} with TTA of query images'.format(self.uncertainty_metric), 
                                              model_out_channels=self.model.out_channels, multiple_prob_list=multiple_prob_list, multiple_data_list=multiple_data_list)

        return indice_list, mean_uncertainty_list, sampling_time

