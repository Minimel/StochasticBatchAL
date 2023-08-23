"""
Author: Mélanie Gaillochet
"""
import torch
import torch.nn as nn

from Utils.utils import normalize


def mean_dice_per_channel(predictions, onehot_target, eps=1e-9,
                          global_dice=False,
                          reduction='mean'):
    """
    We compute the dice, averaged for each channel
    :pa
    """

    dice_dic = {}
    for c in range(1, onehot_target.shape[1]):
        # We select only the predictions and target for the given class
        _selected_idx = torch.tensor([c])
        selected_idx = _selected_idx.to(predictions.get_device())
        pred = torch.index_select(predictions, 1, selected_idx)
        tg = torch.index_select(onehot_target, 1, selected_idx)

        # For each channel, we compute the mean dice
        dice = compute_dice(pred, tg, eps=eps, global_dice=global_dice,
                            reduction=reduction)
        dice_dic['dice_{}'.format(c)] = dice

    return dice_dic


def compute_dice(pred, tg, eps=1e-9, global_dice=False, reduction='mean'):
    """
    We compute the dice for a 3d image
    :param pred: normalized tensor (3d) [BS, x, y, z]
    :param target: tensor (3d) [BS, x, y, z]
    :param eps:
    :param normalize_fct:
    :param weighted:
    :return:
    """
    # if we compute the global dice then we will sum over the batch dim,
    # otherwise no
    if global_dice:
        dim = list(range(0, len(pred.shape)))
    else:
        dim = list(range(1, len(pred.shape)))

    intersect = torch.sum(pred * tg, dim=dim)
    union = pred.sum(dim=dim) + tg.sum(dim=dim)
    dice = (2. * intersect + eps) / (union + eps)

    if reduction == 'mean':
        # We average over the number of samples in the batch
        dice = dice.mean()

    return dice


class DiceLoss(nn.Module):
    """
    This loss is based on the mean dice computed over all channels
    """

    def __init__(self, **kwargs):
        super(DiceLoss, self).__init__()
        # print('Using {} normalization function'.format(normalize_fct))
        self.normalize_fct = kwargs.get('normalize_fct')
        self.reduction = kwargs.get('reduction', 'mean')
        self.global_dice = False

    def forward(self, logits, onehot_target, eps=1e-9):
        pred = normalize(self.normalize_fct, logits)

        dice_dic = mean_dice_per_channel(pred, onehot_target,
                                         global_dice=self.global_dice,
                                         reduction=self.reduction, eps=eps)
        mean_dice = sum(dice_dic.values()) / len(dice_dic)

        loss = 1 - mean_dice

        return loss


class CrossEntropyLoss(nn.Module):
    """
    This is nn.CrossEntropyLoss except that the prediction and target must have the same shape
    """

    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.normalize_fct = kwargs.get('normalize_fct')
        self.CE_weights = kwargs.get('CE_weights', None)
        self.reduction = kwargs.get('reduction', 'mean')
        self.device = kwargs.get('device', 'cpu')

    def forward(self, logits, onehot_target):
        argmax_target = torch.argmax(onehot_target, dim=1)

        if self.normalize_fct == 'softmax':
            loss_fct = nn.CrossEntropyLoss(weight=self.CE_weights, reduction='none')                
            loss = loss_fct(logits, argmax_target)

        elif self.normalize_fct == 'sigmoid':
            if self.CE_weights is not None:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.CE_weights[1], reduction='none')
            else:
                loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            loss = loss_fct(logits, onehot_target.type_as(logits))

        if self.reduction == 'mean':
            loss = torch.mean(loss)

        return loss
    

class WCE_DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        """
        combination of Weight Binary Cross Entropy and Binary Dice Loss
        Args:
            @param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient
            @param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
            @param alpha: weight between WBCE('Weight Binary Cross Entropy') and binary dice, apply on WBCE
        Shapes:
            output: A tensor of shape [N, *] without sigmoid activation function applied
            target: A tensor of shape same with output
        """
        super(WCE_DiceLoss, self).__init__()
        self.normalize_fct = kwargs.get('normalize_fct')
        self.reduction = kwargs.get('reduction', 'mean')
        self.alpha_CE = kwargs.get('alpha_CE', 0.5)
        weighted_CE = kwargs.get('weighted_CE', True)
        n_classes = kwargs.get('n_classes', 4)
        device = kwargs.get('device', 'cpu')

        assert 0 <= self.alpha_CE <= 1, '`alpha` should in [0,1]'

        if weighted_CE:
            # Assume background is first channels and gets prob 0.1.
            # The rest gets prob 0.9/# other channels (not background)
            self.CE_weights = torch.ones((n_classes), device=device) * (1 - 0.1) / (n_classes - 1)
            self.CE_weights[0] = 0.1
            self.CE_weights = self.CE_weights.to(device)
            kwargs['CE_weights'] = self.CE_weights
            self.wce = CrossEntropyLoss(**kwargs)
            print('CE_weightss {}'.format(self.CE_weights))
        else:
            self.wce = CrossEntropyLoss(**kwargs)
        self.dice = DiceLoss(**kwargs)

        self.wce_loss = None
        self.dice_loss = None

    def forward(self, logits, onehot_target):
        # We compute the CE
        self.dice_loss = self.dice(logits, onehot_target)
        self.wce_loss = self.wce(logits, onehot_target)
        
        # If reduction is none, we will take the mean over pixels (loss will be size BS) to match shape of dice_loss
        if self.reduction == 'none':
            self.wce_loss = torch.mean(self.wce_loss, dim=tuple(range(1, len(self.wce_loss.shape))))

        loss = self.alpha_CE * self.wce_loss + (1 - self.alpha_CE) * self.dice_loss
        return loss
