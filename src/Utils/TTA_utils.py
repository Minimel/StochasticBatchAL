"""
Author: Mélanie Gaillochet
"""
import random
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from Utils.utils import normalize


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        cur_device = torch.device(tensor.get_device() if tensor.get_device()>-1 else "cpu")
        noise = torch.randn(tensor.size(), device=cur_device)
        return tensor + noise * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _random_train_transforms(inputs, targets=None, augmentation_params={}):
    """
    We apply given transformations with a given probability to the input data and target (if given)
    Args:
        inputs (tensor of shape (...., H, W)): input volume or image to augment
        targets (tensor of shape (...., H, W), optional): associated segmentation mask to augment. Defaults to None.
        augmentation_params (dict, optional): dictionary of parameters for the given augmentations. Defaults to {}.
            (ie: {'rotation': {'min': -15, 'max': 90}, 'center_crop': {'value': 224}, 'gaussian_noise': {'mean': 0, 'std': 0.01}})

    Returns:
        image (tensor): augmented image
        segmentation (tensor): augmented target (None if input is None)
        param_dic (dic): dictionary of parameters for augmentations. key is augmentation to be applied, value is dic with parameters of augmentations
            (ie: {'rotation': {'angle': 37}, 'center_crop': {'output_size': 224}, 'gaussian_noise': {'mean': 0, 'std': 0.01}})
    """

    param_dic = {}
    if 'hflip' in augmentation_params.keys() and random.random() > 0.5:
        param_dic['hflip'] = 1
    if 'rot90' in augmentation_params.keys():
        random_value = random.randint(0, 3)
        param_dic['rot90'] = {'angle': random_value * 90}
    if 'rotation' in augmentation_params.keys():
        random_value = random.randint(
            -augmentation_params['rotation'], augmentation_params['rotation'])
        param_dic['rotation'] = {'angle': random_value}
    if 'center_crop' in augmentation_params.keys():
        param_dic['center_crop'] = {'output_size': augmentation_params['center_crop']['value']}
    if 'gauss_noise' in augmentation_params.keys():
        param_dic['gauss_noise'] = {'mean': 0, 
                                    'std': augmentation_params['gauss_noise']['std']}

    image, segmentation = _train_transforms(inputs, targets, param_dic)

    return image, segmentation, param_dic


def _train_transforms(image, segmentation=None, param_dic={}):
    # Flip
    if ('hflip' in param_dic.keys()) and (param_dic['hflip'] == 1):
        image = TF.hflip(image)
        segmentation = TF.hflip(
            segmentation) if segmentation is not None else None
        
    # Rotation 90 degrees
    if 'rot90' in param_dic.keys():
        angle = param_dic['rot90']['angle']
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(
            segmentation, angle) if segmentation is not None else None

    # Rotation
    if 'rotation' in param_dic.keys():
        angle = param_dic['rotation']['angle']
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(
            segmentation, angle) if segmentation is not None else None
        
    if ('hflip' in param_dic.keys()) and (param_dic['hflip'] == -1):
        image = TF.hflip(image)
        segmentation = TF.hflip(
            segmentation) if segmentation is not None else None

    # Center Crop
    if 'center_crop' in param_dic.keys():
        output_size = param_dic['center_crop']['output_size']
        image = TF.center_crop(image, output_size=output_size)
        segmentation = TF.center_crop(
            segmentation, output_size=output_size) if segmentation is not None else None

    # Gaussian noise (only on the image)
    if 'gauss_noise' in param_dic.keys():
        mean_gaussian = param_dic['gauss_noise']['mean']
        std_gaussian = param_dic['gauss_noise']['std']
        transform = AddGaussianNoise(mean_gaussian, std_gaussian)
        image = transform(image)

    return image, segmentation


def _reverse_geometry_transform(image, param_dic):

    reverse_param_dic = {}

    # Flip
    if ('hflip' in param_dic.keys()):
        reverse_param_dic['hflip'] = - param_dic['hflip']

    # Rotation 90 degrees
    if 'rot90' in param_dic.keys():
         reverse_param_dic['rot90'] =  {'angle': -1 * param_dic['rot90']['angle']}

    # Rotation
    if 'rotation' in param_dic.keys():
        reverse_param_dic['rotation'] = {'angle': -1 * param_dic['rotation']['angle']}

    # Center Crop
    if 'center_crop' in param_dic.keys():
        reverse_param_dic['center_crop'] = {'output_size': param_dic['center_crop']['output_size']}

    image, _ = _train_transforms(image, param_dic=reverse_param_dic)
    return image


def make_aug_data_list(input, num_augmentations, augmentation_params):
    """ We generate T1(x'), T2(x'), T3(x')
    Args:
        unsup_data (tensor): unsupervised data which we want to augment
    Returns:
        (list of tensors) aug_data: list of augmented data (length=self.consistency_num_augmentations, each with shape (BS, C, H, W, ..)) 
        (list of dics) param_dic_list: list of augmentation parameters (length=self.consistency_num_augmentations)
    """
    aug_data_list = []
    param_dic_list = []
    with torch.no_grad():
        for i in range(num_augmentations):
            # We augment the data
            transformed_data, _, param_dic = _random_train_transforms(input, targets=None, augmentation_params=augmentation_params)
            aug_data_list.append(transformed_data)
            param_dic_list.append(param_dic)
    return aug_data_list, param_dic_list


def make_reverse_aug_prob_list(aug_scores, param_dic_list, model_norm_fct, data_shape):
    """ We generate T1^(-1)[S(T1(x'))], T2^(-1)[S(T2(x'))], T3^(-1)[S(T3(x'))], ...

    Args:
        aug_scores (list of tensors): output logits of all T(x')
        param_dic_list (list of dics): list of all parameters for applied transforms
        data_shape (tuple): data shape

    Returns:
        (list of tensors) cur_prob_list: list of probability tensors after inverse transform
    """
    cur_prob_list = []
    for i in range(len(param_dic_list)):
        # We do the inverse transformation on the output
        rev_aug_scores = _reverse_geometry_transform(
            aug_scores[i * data_shape[0]:(i+1) * data_shape[0]], param_dic_list[i])
        # We get output probability and prediction
        rev_aug_prob = F.softmax(rev_aug_scores, dim=1)
        # We keep track of the output probabilities
        cur_prob_list.append(rev_aug_prob)
    return cur_prob_list


def make_reverse_aug_mask(data_shape, param_dic_list, device='cpu'):
    """ We generate a mask that covers only pixels which appear in all transformed versions of x' 
    (which do not disappear because of a rotation, for example)

    Args:
        data_shape (tupe): shape of data x' (ie: BS, C, H, W, ...)
        param_dic_list (list of dics): list of augmentation parameters (length=self.consistency_num_augmentations)

    Returns:
        (tensor) mask: True-False mask
    """
    _mask = torch.ones(data_shape, device=device)
    for i in range(len(param_dic_list)):
        identity = torch.ones(data_shape, device=device)
        rev_identity = _reverse_geometry_transform(identity, param_dic_list[i])
        # We update the mask (will be 1 whenever both tensor have value 1)
        _mask = torch.logical_and(_mask, rev_identity)
    mask = torch.unsqueeze(_mask, 1)
    mask = mask.repeat(1, len(param_dic_list) + 1, 1, 1, 1)
    return mask

