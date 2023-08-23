"""
Author: Mélanie Gaillochet
"""
import time
import numpy as np

import torch
from pytorch_lightning import seed_everything

from Sampling.entropy_sampler import EntropySampler
from Sampling.dropout_sampler import DropoutSampler
from Sampling.learningloss_sampler import LearningLossSampler
from Sampling.TTA_sampler import TTASampler
from Utils.stochasticbatches_utils import generate_random_groups, aggregate_group_uncertainty, select_top_positions_with_highest_uncertainty
from Utils.load_utils import save_to_logger


def sample_new_indices(sampling_config, budget, unlabeled_dataloader, trainer, **kwargs):
    """
    We implement the logic of sample selection for labeling (AL step)
    """
    strategy = sampling_config['strategy']
    sampling_params = sampling_config['sampling_params']
    print('\n sampling strategy {}'.format(strategy))

    seed = kwargs.get('seed')
    unlabeled_indices = kwargs['unlabeled_indices']
    print('Length unlabeled dataloder: {} ({} samples)'.format(len(unlabeled_dataloader), len(unlabeled_indices)))

    position_list = list(range(len(unlabeled_indices)))

    # We save sampling parameters in trainer
    save_to_logger(trainer.logger, 'hyperparameter', strategy, 'sampling_strategy')
    save_to_logger(trainer.logger, 'hyperparameter', sampling_params, 'sampling_config')

    seed_everything(seed)

    uncertainty_based_strategies = ['Entropy', 'Dropout', 'LearningLoss', 'TTA']

    additional_time = 0

    ### For random sampling
    if strategy == 'RS':
        sampling_start_time = time.time()
        query_indices = list(np.random.choice(unlabeled_indices, size=budget, replace=False))
        sampling_time = time.time() - sampling_start_time
        uncertainty_values = []

    ### For uncertainty-based sampling 
    elif strategy in uncertainty_based_strategies:
        if strategy == 'Entropy':
            sampler = EntropySampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'Dropout':
            sampler = DropoutSampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'TTA':
            sampler = TTASampler(budget, trainer, **kwargs)
            unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)

        elif strategy == 'LearningLoss':
            sampler = LearningLossSampler(budget, trainer, **kwargs)
            # We will compute the uncertainty of all data-points in the unlabeled dataloader or on a subset of it
            num_subset_indices = kwargs.get('num_subset_indices')
            if num_subset_indices == 'all':
                unlabeled_indice_list, uncertainty_list, sampling_time = sampler.get_uncertainty(unlabeled_dataloader)
            else:
                subset_indices = list(np.random.choice(unlabeled_dataloader.sampler.indices, size=num_subset_indices, replace=False))
                subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices,
                                                                            generator=torch.Generator())
                subset_dataloader = torch.utils.data.DataLoader(unlabeled_dataloader.dataset, sampler=subset_sampler,
                                                                batch_size=unlabeled_dataloader.batch_size,
                                                                drop_last=False, num_workers=1)
                unlabeled_indice_list, uncertainty_list = sampler.get_uncertainty(subset_dataloader)

        if 'RandomPacks' in sampling_params.keys():
            print('uncertainty_list {}'.format(sorted(np.unique(uncertainty_list))))
            additional_start_time = time.time()

            # We extract parameters
            num_groups = sampling_params['RandomPacks']['num_groups']
            resampling = sampling_params['RandomPacks']['resampling']
            aggregation = sampling_params['RandomPacks']['aggregation']

            position_group_list = generate_random_groups(position_list, num_groups, group_size=budget, resample=resampling)
            print("len(position_group_list): {}".format(position_group_list))
            aggregated_uncertainty_list = aggregate_group_uncertainty(position_group_list, uncertainty_list, aggregation=aggregation)
            position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_group_list, aggregated_uncertainty_list, num_groups=1)

            # We get the corresponding indices and uncertainty values for the selected position
            # Since we only want one group of indices, we take the first element of the position_with_highest_uncertainty
            query_indices = [[unlabeled_indice_list[i] for i in indices] for indices in position_with_highest_uncertainty][0]
            uncertainty_values = [[uncertainty_list[i] for i in indices] for indices in position_with_highest_uncertainty][0]

            additional_time = time.time() - additional_start_time

        else:
            print('uncertainty_list {}'.format(sorted(np.unique(uncertainty_list))))
            additional_start_time = time.time()
            
            # Index in ascending order
            position_with_highest_uncertainty = select_top_positions_with_highest_uncertainty(position_list, uncertainty_list, num_groups=budget)
            query_indices = [unlabeled_indice_list[i] for i in position_with_highest_uncertainty]
            uncertainty_values = [uncertainty_list[i] for i in position_with_highest_uncertainty]
            
            additional_time = time.time() - additional_start_time
            
        save_to_logger(trainer.logger, 'metric', np.round(additional_time/60, 2), 'sampling additional time (min)', trainer.current_epoch)
        save_to_logger(trainer.logger, 'list', uncertainty_values, 'uncertainty_values', None)

    # We keep track of runtime
    save_to_logger(trainer.logger, 'metric', np.round(sampling_time/60, 2), 'sampling time (min)', trainer.current_epoch)
    save_to_logger(trainer.logger, 'metric', np.round((sampling_time + additional_time)/60), 'sampling total time (min)', trainer.current_epoch)
    save_to_logger(trainer.logger, 'list', query_indices, 'query_indices')

    return query_indices, sampling_time + additional_time
