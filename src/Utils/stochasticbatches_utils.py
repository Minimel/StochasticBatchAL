"""
Author: Mélanie Gaillochet
"""
from contextlib import AbstractContextManager
import numpy as np
from typing import Any, List, Tuple

def aggregate_group_uncertainty(groups: List[List[int]], uncertainty_values: List[float], aggregation: str) -> List[float]:
    """
    Function that takes in a list of groups of indices and a list of uncertainty values and returns a list of aggregated uncertainty values per group.
    """
    if aggregation not in ['mean', 'max', 'sum']:
        raise ValueError("Invalid aggregation method. Supported methods are 'mean', 'max', and 'sum'.")

    aggregated_uncertainty = []

    for group in groups:
        group_uncertainties = [uncertainty_values[idx] for idx in group]

        if aggregation == 'mean':
            aggregated_value = np.mean(group_uncertainties)
        elif aggregation == 'max':
            aggregated_value = np.max(group_uncertainties)
        else:  # aggregation == 'sum'
            aggregated_value = np.sum(group_uncertainties)

        aggregated_uncertainty.append(aggregated_value)

    return aggregated_uncertainty


def generate_random_groups(indices: List[int], num_groups: int, group_size: int, resample: bool = False) -> List[List[int]]:
    """
    Function that takes in a list of indices and returns a list of groups of indices of the specified size.
    If resampling=False and num_groups*group_size > len(indices), there will be fewer groups (all of size group_size).
    """
    if resample:
        # Resampling allowed: we can use the same index in multiple groups
        groups = [np.random.choice(indices, group_size, replace=False).tolist() for _ in range(num_groups)]
    else:
        # Resampling not allowed: shuffle the indices and split into groups
        shuffled_indices = np.random.permutation(indices)
        groups = [shuffled_indices[i:i+group_size].tolist() for i in range(0, len(shuffled_indices), group_size)]

        # In case there are not enough indices to form num_groups, there will be less groups
        if len(groups[-1]) < group_size:
            groups.pop()

    return groups


def select_top_positions_with_highest_uncertainty(positions: List[List[int]], aggregated_uncertainty: List[float], num_groups: int) -> List[List[int]]:
    """
    Function to select the list positions with the highest aggregated uncertainty. Can return multiple groups.
    Args:
        positions: List of positions (can be list of list, or single list)
        aggregated_uncertainty: List of aggregated uncertainty values for each group (list of same length as groups)
        num_groups: Number of groups to select
    """
    sorted_uncertainty_indices = np.argsort(aggregated_uncertainty)[::-1]  # Sort in descending order
    top_groups = [positions[i] for i in sorted_uncertainty_indices[:num_groups]]
    return top_groups


if __name__ == '__main__':
    """ We make tests for the functions in this file"""
    import unittest
    import torch
    import numpy as np
    import monai.metrics as monai_metrics

    class Test_aggregate_group_uncertainty(unittest.TestCase):
        def setUp(self):
            self.indices = list(range(10))
            self.uncertainty_values = list(np.array(range(10)) /10)

        def test_metrics(self):
            groups = [[1] * 10] + [self.indices] + [[2, 5] * 5]

            aggregated_values = aggregate_group_uncertainty(groups, self.uncertainty_values, aggregation='mean')
            expected_mean_values = [0.1, 0.45, 0.35]
            assert aggregated_values == expected_mean_values, "The lists are not equal in mean."

            aggregated_values = aggregate_group_uncertainty(groups, self.uncertainty_values, aggregation='max')
            expected_max_values = [0.1, 0.9, 0.5]
            assert aggregated_values == expected_max_values, "The lists are not equal in max."
    
            aggregated_values = aggregate_group_uncertainty(groups, self.uncertainty_values, aggregation='sum')
            expected_sum_values = [1, 4.5, 3.5]
            assert aggregated_values == expected_sum_values, "The lists are not equal in sum."

    class Test_generate_random_groups(unittest.TestCase):
        def setUp(self):
            self.indices = list(range(100))

        def test_metrics(self):

            groups = generate_random_groups(self.indices, num_groups=10, group_size=10, resample=False)
            flat_groups = [item for sublist in groups for item in sublist]
            assert len(groups) == 10, "The number of groups is not correct."
            assert len(flat_groups) == 100, "The number of values is not correct."
            assert len(np.unique(flat_groups)) == 100, "There are repeated values."
            assert np.unique([len(sublist) for sublist in groups]) == 10, "The group sizes are not correct."

            # With resampling
            groups = generate_random_groups(self.indices, num_groups=10, group_size=15, resample=True)
            flat_groups = [item for sublist in groups for item in sublist]
            assert len(groups) == 10, "The number of groups is not correct."
            assert len(flat_groups) == 150, "The number of values is not correct."
            assert np.unique([len(sublist) for sublist in groups]) == 15, "The group sizes are not correct."

            # Not enough indices to form num_groups
            groups = generate_random_groups(self.indices, num_groups=10, group_size=15, resample=False)
            flat_groups = [item for sublist in groups for item in sublist]
            assert len(groups) == 6, "The number of groups is not correct."
            assert len(flat_groups) == 90, "The number of values is not correct."
            assert len(np.unique(flat_groups)) == 90, "There are repeated values."
            assert np.unique([len(sublist) for sublist in groups]) == 15, "The group sizes are not correct."

    class Test_select_top_groups_with_highest_uncertainty(unittest.TestCase):
        def SetUp(self):
            pass
        
        def subTest(self):
            # groups as a list of lists
            groups = [[1] * 10] + [[2, 5] * 5] + [[3, 4] * 5]
            aggregated_uncertainty = [0.1, 0.45, 0.35]
            top_groups = select_top_positions_with_highest_uncertainty(groups, aggregated_uncertainty, num_groups=2)
            assert len(top_groups) == 2, "The number of groups is not correct."
            assert top_groups[0] == [[2, 5] * 5], "The first group is not correct."
            assert top_groups[1] == [[3, 4] * 5], "The second group is not correct."

            # groups as a list of indices
            groups = [5, 3, 4, 7, 8, 0]
            aggregated_uncertainty = [0.1, 0.45, 0.35, 0.2, 0.6, 0.4]
            top_groups = select_top_positions_with_highest_uncertainty(groups, aggregated_uncertainty, num_groups=3)
            assert len(top_groups) == 3, "The number of groups is not correct."
            assert top_groups[0] == 8, "The first group is not correct."
            assert top_groups[1] == 3, "The first group is not correct."

    unittest.main()
