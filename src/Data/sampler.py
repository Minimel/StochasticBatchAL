"""
Author: Mélanie Gaillochet
"""
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class _InfiniteSubsetRandomIterator(Iterator):
    def __init__(self, data_source, indices, shuffle=True):
        self.data_source = data_source
        self.indices = indices
        self.shuffle = shuffle
        if self.shuffle:
            permuted_indices = np.random.permutation(self.indices).tolist()
            self.iterator = iter(permuted_indices)
        else:
            self.iterator = iter(self.indices)

    def __next__(self):
        try:
            idx = next(self.iterator)
        except StopIteration:
            if self.shuffle:
                permuted_indices = np.random.permutation(self.indices).tolist()
                self.iterator = iter(permuted_indices)
            else:
                self.iterator = iter(self.indices)
            idx = next(self.iterator)
        return idx


class InfiniteSubsetRandomSampler(Sampler):
    """
    This is a sampler that randomly selects data indices from a list of indices, in an infinite loop
    """

    def __init__(self, data_source, indices, shuffle=True, length=0):
        self.data_source = data_source
        self.indices = indices
        self.shuffle = shuffle
        self.length =  length

    def __iter__(self):
        return _InfiniteSubsetRandomIterator(self.data_source, self.indices,
                                             shuffle=self.shuffle)

    def __len__(self):
        if self.length == 0:
            return len(self.data_source)
        else:
            return self.length


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

