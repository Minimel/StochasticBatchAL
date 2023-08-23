"""
Author: Mélanie Gaillochet
"""
from contextlib import AbstractContextManager
from typing import Any
import numpy as np

import torch
import torch.nn.functional as F


def entropy(p_input: torch.tensor, dim:int=1):
    """
    We compute the entropy along the given dimension
    args:
        input: tensor of normalized probabilities about the given dimension. Shape (N, C, H, W)
        dim: dimension along which to compute the entropy, default=1
    """
    entropy = -torch.sum(p_input * torch.log(p_input + 1e-8), dim=dim)
    return entropy


def weighted_jsd(p_distributions: torch.tensor, alpha: float) -> torch.tensor:
    """
    We computed the weighted JSD of the given list of tensors
    JS divergence JSD(p1, .., pn) = H(sum_i_to_n [w_i * p_i]) - sum_i_to_n [w_i * H(p_i)], where w_i is the weight given to each probability
                                  = Entropy of average prob. - Average of entropy
    Args:
        distributions: tensor of probabilities of shape (BS, #inferences, C, H, W), where #inferences=#samples to compute JSD over, C=#classes, H,W=image dimensions
        alpha: weight given to the average probability distribution
    """
    # Calculate the entropy of the average probability distribution
    avg_distribution = torch.mean(p_distributions, dim=1) # (BS, C, H, W)
    entropy_avg_prob = entropy(avg_distribution, dim=1)   # (BS, H, W)

    # Calculate the average of entropy for the individual distributions
    _entropies = [entropy(d, dim=1) for d in p_distributions]
    entropies = torch.stack(_entropies, dim=0)  # (BS, #inferences, H, W)
    avg_entropy = torch.mean(entropies, dim=1)  # (BS, H, W)

    # Compute the JSD by combining the two entropy terms with the alpha weight
    jsd_alpha = alpha * entropy_avg_prob + (1 - alpha) * avg_entropy # (BS, H, W)

    return jsd_alpha 


if __name__ == '__main__':
    """ We make tests for the functions in this file"""
    import unittest
    import torch
    import numpy as np
    import monai.metrics as monai_metrics

    class Test_entropy(unittest.TestCase):
        def setUp(self) -> None:
            self.input = torch.zeros(2, 3, 2, 2)
            self.input[:, 0, 0, 0] = 0.4
            self.input[:, 1, 0, 0] = 0.2
            self.input[:, 2, 0, 0] = 0.1
            
            self.input[:, 0, 0, 1] = 0.5
            self.input[:, 1, 0, 1] = 0.5
            self.input[:, 2, 0, 1] = 0

            self.input[:, 0, 1, 0] = 0.8
            self.input[:, 1, 1, 0] = 0.1
            self.input[:, 2, 1, 0] = 0.1

            self.input[:, 0, 1, 1] = 0.3
            self.input[:, 1, 1, 1] = 0.3
            self.input[:, 2, 1, 1] = 0.4

            print(self.input)

        def test_metrics(self) -> None:
            expected_entropy = torch.tensor([[[0.9187, 0.6931], [0.6390, 1.0889]], [[0.9187, 0.6931], [0.6390, 1.0889]]])

            entropy = entropy(self.input, dim=1)
            
            for i in range(len(expected_entropy.flatten())):
                self.assertAlmostEqual(entropy.flatten()[i].item(), expected_entropy.flatten()[i].item(), places=4)

    class Test_weighted_jsd(unittest.TestCase):
        def setUp(self) -> None:
            self.input = torch.zeros(2, 3, 2, 2)
            self.input[:, 0, 0, 0] = 0.4
            self.input[:, 1, 0, 0] = 0.2
            self.input[:, 2, 0, 0] = 0.1
            
            self.input[:, 0, 0, 1] = 0.5
            self.input[:, 1, 0, 1] = 0.5
            self.input[:, 2, 0, 1] = 0

            self.input[:, 0, 1, 0] = 0.8
            self.input[:, 1, 1, 0] = 0.1
            self.input[:, 2, 1, 0] = 0.1

            self.input[:, 0, 1, 1] = 0.3
            self.input[:, 1, 1, 1] = 0.3
            self.input[:, 2, 1, 1] = 0.4

        def subTest(self) -> None:
            pass
            

    unittest.main()