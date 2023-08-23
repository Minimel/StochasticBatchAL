"""
Author: Mélanie Gaillochet
"""
import os
import random
import re

from flatten_dict import flatten
from flatten_dict import unflatten
import numpy as np
import torch


def _atoi(text):
    """
    We return the string as type int if it represents a number (or the string itself otherwise)
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    We return the list of string and digits as different entries,
    following the order in which they appear in the string
    """
    return [_atoi(c) for c in re.split('(\\d+)', text)]


def default0():
    """ Function to be called by defaultdic to default values to 0 """
    return 0


def defaultinf():
    """ Function to be called by defaultdic to default values to inf """
    return np.inf


def to_onehot(input, n_classes):
    """
    We do a one hot encoding of each label in 3D.
    (ie: instead of having a dimension of size 1 with values 0-k,
    we have 3 axes, all with values 0 or 1)
    :param input: tensor
    :param n_classes:
    :return:
    """
    assert torch.is_tensor(input)

    # We get (bs, l, h, w, n_channels), where n_channels is now > 1
    one_hot = torch.nn.functional.one_hot(input.to(torch.int64), n_classes)

    # We permute axes to put # channels as 2nd dim
    if len(one_hot.shape) == 5:
        one_hot = one_hot.permute(0, 4, 1, 2, 3)
    elif len(one_hot.shape) == 4:
        one_hot = one_hot.permute(0, 3, 1, 2)
    return one_hot


def normalize(normalize_fct, x):
    """We apply sigmoid or softmax on the logits to get probabilities
    Args:
        normalize_fct (str): 'softmax' or 'sigmoid'
        x (tensor): logits. Must be 4D or 5D (B, C, H, W) or (B, C, L, H, W)
    """
    assert len(x.shape) == 4 or len(x.shape) == 5

    if normalize_fct == 'softmax':
        fct = torch.nn.Softmax(dim=1)

    elif normalize_fct == 'sigmoid':
        fct = torch.nn.Sigmoid()

    return fct(x)


def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FixedSeed:
    def __init__(self, seed):
        self.seed = seed
        self.random_state = None
        self.np_state = None
        self.torch_state = None

    def __enter__(self):
        self.random_state = random.getstate()
        self.np_state = np.random.get_state()
        self.torch_state = torch.random.get_rng_state()
        fix_all_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        random.setstate(self.random_state)
        np.random.set_state(self.np_state)  
        torch.random.set_rng_state(self.torch_state) 


def apply_seeded_transform(sample, transform_class, transform_config, data_keys=['input', 'mask'], seed=42, cuda=True):
    """
        We apply seeded transformations, with given seed
    Args:
        sample (dict): dict with keys 'data' and 'label'
        transform_class (class): class of the transform to apply (ie: Preprocess)
        transform_config (dict): 
        interpolation_mode (str): 'bilinear' or 'nearest'
        seed (int): seeding the transformation

    Returns:
        dictionary with keys 'data' and 'label', with values being the transformed versions on input
    """
    with FixedSeed(seed):
        transf = transform_class(data_keys, **transform_config).cuda() if cuda else transform_class(data_keys, **transform_config)
        return  transf(sample)


def update_config_from_args(config, args, prefix):
    """
    We update the given config with the values given by the args
    Args:
        config (list): config that we would like to update
        args (parser arguments): input arguments whose values starting with given prefix we would like to use
                                Must be in the form <prefix>/<config_var_name-separated-by-/-if-leveled>/ (ie: ssl/sched/step_size) 
                                ! Must also not contain 'config' in the name !
        prefix (str): all parser arguments starting with the prefix + '/' will be updated.

    Returns:
        config: updated_config
    """
    # We extract the names of variables to update
    var_to_update_list = [name for name in vars(args) if prefix +'__' in name]# and 'config' not in name)]
    
    updated_config = flatten(config)  # We convert dictionary to list of tuples (tuples incorporating level information)
    for name in var_to_update_list:
        new_val = getattr(args, name)
        if new_val is not None:   # if the values given is not null, we will update the dictionary
            variable = name.replace(prefix + '__', '', 1)  # We remove the prefix
            level_tuple = tuple(variable.split('__'))   # We create a tuple with all sublevels of config
            updated_config[level_tuple] = new_val
    updated_config = unflatten(updated_config)  # We convert back to a dictionary
    
    return updated_config



if __name__ == '__main__':
    """ We make tests for the functions in this file"""

    import sys
    import os
    import pytorch_lightning as pl

    src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(src_folder))
    sys.path.insert(0, os.path.join(src_folder, 'Data'))

    # Test for to_one_hot
    def minitest_to_onehot():
        input = torch.tensor([[[0, 1, 2, 3], [3, 2, 1, 0]], 
                             [[0, 1, 1, 1], [3, 2, 2, 1]], 
                             [[2, 3, 2, 3], [1, 1, 1, 1]]]).float()
        output = to_onehot(input, 4)

        assert output.shape == (3, 4, 2, 4), "output.shape should be (3, 4, 2, 4)"
        assert torch.equal(output[:,0, :, : ], torch.tensor([[[1, 0, 0, 0], [0, 0, 0, 1]], 
                                                             [[1, 0, 0, 0], [0, 0, 0, 0]], 
                                                             [[0, 0, 0, 0], [0, 0, 0, 0]]])), "Wrong output[:, 0, :, :] "
        assert torch.equal(output[:, 1, :, : ], torch.tensor([[[0, 1, 0, 0], [0, 0, 1, 0]], 
                                                              [[0, 1, 1, 1], [0, 0, 0, 1]], 
                                                              [[0, 0, 0, 0], [1, 1, 1, 1]]])), "Wrong output[:, 1, :, :] "
        assert torch.equal(output[:, 2, :, : ], torch.tensor([[[0, 0, 1, 0], [0, 1, 0, 0]], 
                                                              [[0, 0, 0, 0], [0, 1, 1, 0]], 
                                                              [[1, 0, 1, 0], [0, 0, 0, 0]]])), "Wrong output[:, 2, :, :] "
        assert torch.equal(output[:, 3, :, : ], torch.tensor([[[0, 0, 0, 1], [1, 0, 0, 0]], 
                                                              [[0, 0, 0, 0], [1, 0, 0, 0]], 
                                                              [[0, 1, 0, 1], [0, 0, 0, 0]]])), "Wrong output[:, 3, :, :] "
        

    def mini_test_normalize():
        input = torch.rand(3, 4, 5, 6).float()

        # We test for softmax
        norm_input = normalize(normalize_fct='softmax', x=input)
        _sum_v = torch.sum(torch.exp(input), dim=1) #shape (3, 5, 6)
        sum_v = _sum_v[:, None, :, :].repeat(1, 4, 1, 1)
        assert torch.allclose(norm_input, torch.exp(input) / sum_v), "Wrong softmax normalization"
        assert torch.allclose(torch.sum(norm_input, dim=1), torch.ones(3, 5, 6)), "Wrong softmax normalization"

        # We test for sigmoid
        norm_input = normalize(normalize_fct='sigmoid', x=input)
        assert torch.allclose(norm_input, torch.exp(input) / (1 + torch.exp(input))), "Wrong sigmoid normalization"


    def minitest_FixedSeed():
        fix_all_seed(0)
        start_random_state = random.getstate()
        start_np_state = np.random.get_state()
        start_torch_state = torch.random.get_rng_state()

        with FixedSeed(1):
            temp_random_state = random.getstate()
            temp_np_state = np.random.get_state()
            temp_torch_state = torch.random.get_rng_state()
        
        assert start_random_state == random.getstate(), "random state should not have changed"
        for i in range(len(start_np_state)):
            try:
                assert (start_np_state[i] == np.random.get_state()[i]).all(), "np state should not have changed"
            except AttributeError:
                assert start_np_state[i] == np.random.get_state()[i], "np state should not have changed"
        assert (start_torch_state == torch.random.get_rng_state()).all(), "torch state should not have changed"

        assert temp_random_state != random.getstate(), "random state should have changed"
        changed_np_state = False
        for i in range(len(temp_np_state)):
            try:
                if (temp_np_state[i] != np.random.get_state()[i]).all():
                    changed_np_state = True
                    break
            except AttributeError:
                if temp_np_state[i] != np.random.get_state()[i]:
                    changed_np_state = True
                    break
        assert changed_np_state is True, "np state should have changed"
        assert not (temp_torch_state == torch.random.get_rng_state()).all(), "torch state should have changed"

    def minitest_update_config_from_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_config__model__name', type=str, default='unet')
        parser.add_argument('--model_config__model__out_channels', type=int, default=4)   
        parser.add_argument('--model_config__model__activation__layer', type=str, default='relu')   
        parser.add_argument('--model_config__model__activation__sigmoid', type=str, default='softmax')   

        config = {'model': {'name': None, 'in_channels': 1, 'out_channels': 1, 'activation': {'layer': None, 'output': 'sigmoid'}},
                  'loss': {'name': 'bce', 'reduction': 'mean'}
                }
        parser = pl.Trainer.add_argparse_args(parser)
        args = parser.parse_args()

        updated_config = update_config_from_args(config, args, 'model_config')

        assert updated_config['model']['name'] == 'unet', "updated_config['model']['name'] should be 'unet'"
        assert updated_config['model']['in_channels'] == 1, "updated_config['model']['in_channels'] should be 1"
        assert updated_config['model']['out_channels'] == 4, "updated_config['model']['out_channels'] should be 4"
        assert updated_config['model']['activation']['layer'] == 'relu', "updated_config['model']['activation']['layer'] should be 'relu'"
        assert updated_config['model']['activation']['output'] == 'sigmoid', "updated_config['model']['activation']['output'] should be 'sigmoid'"
        assert updated_config['loss']['name'] == 'bce', "updated_config['loss']['name'] should be 'bce'"
        assert updated_config['loss']['reduction'] == 'mean', "updated_config['loss']['reduction'] should be 'mean'"    


    minitest_to_onehot()
    mini_test_normalize()
    minitest_FixedSeed()
    minitest_update_config_from_args()