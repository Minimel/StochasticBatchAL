"""
Author: Mélanie Gaillochet
"""
from monai import metrics as monai_metrics

from torch.optim import lr_scheduler


############################# METRICS #############################
dice_metric = monai_metrics.DiceMetric(include_background=False, reduction='none', get_not_nans=False, ignore_empty=False)
iou_metric = monai_metrics.MeanIoU(include_background=False, reduction='none', get_not_nans=False, ignore_empty=False)
hausdorff_metric = monai_metrics.HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, directed=False, reduction='none', get_not_nans=False)
hausdorff95_metric = monai_metrics.HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, directed=False, reduction='none', get_not_nans=False)
assd_metric = monai_metrics.SurfaceDistanceMetric(include_background=False, symmetric=False, distance_metric='euclidean', reduction='none', get_not_nans=False)

metrics = {
    'dice': dice_metric,
    'iou': iou_metric,
    'hausdorff': hausdorff_metric,
    'hausdorff95': hausdorff95_metric,
    'assd': assd_metric

}


"""from https://github.com/jizongFox/deepclustering2/blob/master/deepclustering2/schedulers/warmup_scheduler.py"""
class GradualWarmupScheduler(lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.0:
            raise ValueError("multiplier should be greater than 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


if __name__ == '__main__':
    """ We make tests for the functions in this file"""
    import unittest
    import torch
    import numpy as np
    import monai.metrics as monai_metrics

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

    class TestMonaiMetrics(unittest.TestCase):

        def setUp(self):
            self.ground_truth = torch.zeros(2, 2, 128, 128)
            self.prediction = torch.zeros(2, 2, 128, 128)

            # Add some foreground pixels
            self.ground_truth[:, 1, 64, 64] = 1
            self.ground_truth[:, 1, 64, 65] = 1
            self.ground_truth[:, 0, :, :] = 1 - self.ground_truth[:, 1, :, :]

            self.prediction[:, 1, 64, 65] = 1
            self.prediction[:, 0, :, :] = 1 - self.prediction[:, 1, :, :]

            self.metrics = {    
                'dice': dice_metric,
                'iou': iou_metric,
                'hausdorff': hausdorff_metric,
                'hausdorff95': hausdorff95_metric,
                'assd': assd_metric}

        def test_metrics(self):
            expected_results = {
                'dice': [2 / 3],
                'iou': [1/2],
                'hausdorff': [1.0],
                'hausdorff95': [1.0],
                'assd': [0.5]
            }

            for metric_name, metric_instance in self.metrics.items():
                _result = metric_instance(self.ground_truth, self.prediction)
                result = torch.mean(_result, dim=0)

                if metric_name == 'hausdorff95':
                    assert expected_results[metric_name] <= expected_result  # Hausdorff95 is not deterministic. We just check that it is less than Hausdorff

                else:
                    expected_result = expected_results[metric_name]
                    for i, r in enumerate(result):
                        self.assertAlmostEqual(r.item(), expected_result[i], places=4)

    class TestMonaiMetrics_multiclass(unittest.TestCase):

        def setUp(self):
            self.out_channels = 3
            self.ground_truth = torch.tensor([[[1, 2, 2], [1, 2, 2], [1, 0, 0]], [[1, 2, 2], [1, 2, 2], [1, 0, 0]]])
            self.prediction = torch.tensor([[[2, 2, 0], [1, 1, 0], [1, 1, 0]], [[2, 2, 0], [1, 1, 0], [1, 1, 0]]])
            
            self.onehot_ground_truth = to_onehot(self.ground_truth.squeeze(1), self.out_channels)
            self.onehot_prediction = to_onehot(self.prediction.squeeze(1), self.out_channels)

            self.metrics = {    
                'dice': dice_metric,
                'hausdorff': hausdorff_metric,
                }

        def test_metrics(self):

            expected_results = {
                'dice': [4/7, 1/3],
                'hausdorff': [1.0, np.sqrt(2)]
            }

            for metric_name, metric_instance in self.metrics.items():
                _result = metric_instance(self.onehot_ground_truth, self.onehot_prediction)
                result = torch.mean(_result, dim=0)

                expected_result = expected_results[metric_name]
                for i, r in enumerate(result):
                    self.assertAlmostEqual(r.item(), expected_result[i], places=4)

    class TestMonaiMetrics_backgroundtarget(unittest.TestCase):

        def setUp(self):
            self.out_channels = 3
            self.ground_truth = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
            self.prediction = torch.tensor([[[2, 2, 0], [1, 1, 0], [1, 1, 0]]])
            
            self.onehot_ground_truth = to_onehot(self.ground_truth.squeeze(1), self.out_channels)
            self.onehot_prediction = to_onehot(self.prediction.squeeze(1), self.out_channels)

            self.metrics = {    
                'dice': dice_metric,
                'hausdorff': hausdorff_metric,
                }

        def test_metrics(self):

            expected_results = {
                'dice': [0, 0],
                'hausdorff': [np.inf, np.inf]
            }

            for metric_name, metric_instance in self.metrics.items():
                _result = metric_instance(self.onehot_ground_truth, self.onehot_prediction)
                result = torch.mean(_result, dim=0)

                expected_result = expected_results[metric_name]
                for i, r in enumerate(result):
                    self.assertAlmostEqual(r.item(), expected_result[i], places=4)

    class TestMonaiMetrics_emptyclass(unittest.TestCase):

        def setUp(self):
            self.out_channels = 3
            self.ground_truth = torch.tensor([[[0, 2, 2], [0, 2, 2], [0, 0, 0]]])
            self.prediction = torch.tensor([[[2, 2, 0], [1, 1, 0], [1, 1, 0]]])
            
            self.onehot_ground_truth = to_onehot(self.ground_truth.squeeze(1), self.out_channels)
            self.onehot_prediction = to_onehot(self.prediction.squeeze(1), self.out_channels)

            self.metrics = {    
                'dice': dice_metric,
                'hausdorff': hausdorff_metric,
                }

        def test_metrics(self):

            expected_results = {
                'dice': [0, 1/3],
                'hausdorff': [np.inf, np.sqrt(2)]
            }

            for metric_name, metric_instance in self.metrics.items():
                _result = metric_instance(self.onehot_ground_truth, self.onehot_prediction)
                result = torch.mean(_result, dim=0)

                expected_result = expected_results[metric_name]
                for i, r in enumerate(result):
                    self.assertAlmostEqual(r.item(), expected_result[i], places=4)

    class TestMonaiMetrics_emptyclass2(unittest.TestCase):

        def setUp(self):
            self.out_channels = 3
            self.ground_truth = torch.tensor([[[0, 0, 0], [2, 2, 0], [2, 2, 0]]])
            self.prediction = torch.tensor([[[0, 0, 2], [1, 1, 0], [1, 1, 1]]])
            
            self.onehot_ground_truth = to_onehot(self.ground_truth.squeeze(1), self.out_channels)
            print(self.onehot_ground_truth)
            self.onehot_prediction = to_onehot(self.prediction.squeeze(1), self.out_channels)
            print(self.onehot_prediction)

            self.metrics = {    
                'dice': dice_metric,
                'hausdorff': hausdorff_metric,
                }

        def test_metrics(self):

            expected_results = {
                'dice': [0, 0],
                'hausdorff': [np.inf, np.sqrt(8)]
            }

            for metric_name, metric_instance in self.metrics.items():
                _result = metric_instance(self.onehot_ground_truth, self.onehot_prediction)
                result = torch.mean(_result, dim=0)
                print('{}: {}'.format(metric_name, result))

                expected_result = expected_results[metric_name]
                for i, r in enumerate(result):
                    self.assertAlmostEqual(r.item(), expected_result[i], places=4)


    unittest.main()
