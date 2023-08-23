"""
Author: Mélanie Gaillochet
"""
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
import torch.optim as optim
from pytorch_lightning_spells.lr_schedulers import MultiStageScheduler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

from Models.unet_modules import decoderUNet, encoderbottleneckUNet
from Utils.loss_utils import WCE_DiceLoss
from Utils.plot_utils import plot_data_pred_volume
from Utils.training_utils import (GradualWarmupScheduler, assd_metric,
                                  dice_metric, hausdorff95_metric,
                                  hausdorff_metric, iou_metric)
from Utils.utils import to_onehot


class UNet(pl.LightningModule):
    def __init__(self, 
                 per_device_batch_size: int = 1,
                 num_devices: int = 1,
                 lr: float = 1e-6, #1e-3,
                 weight_decay: float = 5e-4, #1e-4,
                 model_config: dict = {},
                 sched_config: dict = {},
                 loss_config: dict = {},
                 val_plot_slice_interval: int = 1,
                 seed = 42
                 ):
        """
        This modified UNet differs from the original one with the use of
        leaky relu (instead of ReLU) and the addition of residual connections.
        The idea is to help deal with fine-grained details
        :param in_channels: # of input channels (ie: 3 if  image in RGB)
        :param out_channels: # of output channels (# segmentation classes)
        """
        super().__init__()
        self.save_hyperparameters()
        self.per_device_batch_size = per_device_batch_size
        self.num_devices = num_devices
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.sched_config = sched_config

        self.in_channels = model_config["in_channels"]
        self.out_channels = model_config["out_channels"]
        self.channel_list = model_config["channel_list"] # channel_list=[64, 128, 256, 512]
        
        self.encoderbottleneck = encoderbottleneckUNet(model_config)
        self.decoder = decoderUNet(model_config)
        
        if loss_config != {}:
            self.model_loss = WCE_DiceLoss(**loss_config)
            
        # For 3D validation and test (Note that train_volume_list and test_volume_list are loaded by the datamodule)
        self.val_data_list, self.test_data_list = [], []
        self.val_target_list, self.test_target_list = [], []
        self.val_logits_list, self.test_logits_list = [], []
        self.val_dice_list, self.test_dice_list = [], []
        self.val_per_class_dice_list, self.test_per_class_dice_list = [], []
        self.val_slice, self.test_slice = 0, 0

        # Plot params
        self.log_metric_freq = 5 
        self.log_img_freq = 80  # Must be multiple of self.log_metric_freq
        self.val_plot_slice_interval = val_plot_slice_interval
        self.plot_type = 'contour' if self.out_channels == 2 else 'image'

    def forward(self, x):
        # Encoding
        center, [enc_1, enc_2, enc_3] = self.encoderbottleneck(x)
        out, [dec_1, dec_2, dec_3] = self.decoder(center, enc_1, enc_2, enc_3)  
        return out, [enc_1, enc_2, enc_3, center, dec_1, dec_2, dec_3]    
    
    def _compute_loss(self, logits, y):
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        loss = self.model_loss(logits, onehot_target)
        return loss
    
    def _compute_metrics(self, logits, y, pred_dim=1):
        pred = torch.argmax(logits, dim=pred_dim)
        onehot_pred = to_onehot(pred.squeeze(1), self.out_channels)
        onehot_target = to_onehot(y.squeeze(1), self.out_channels)
        _dice = dice_metric(onehot_pred, onehot_target)
        dice = torch.mean(_dice)
        per_class_dice = torch.mean(_dice, dim=0)
        hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target))
        per_class_hausdorff95 = torch.mean(hausdorff95_metric(onehot_pred, onehot_target), dim=0)
        return 100*dice, 100*per_class_dice, hausdorff95, per_class_hausdorff95

    def _training_step(self, batch, batch_idx):
        """ Contains all computations necessary to produce a loss value to train the model"""
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, per_class_dice, _, _ = self._compute_metrics(logits, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('train/loss', loss)
            self.log('train/dice', dice)
            for i in range(per_class_dice.shape[0]):
                self.log('train/dice_{}'.format(i), per_class_dice[i])
        return loss
    
    def _validation_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, per_class_dice, _, _ = self._compute_metrics(logits, y)

        if self.current_epoch % self.log_metric_freq == 0:
            self.log('val/loss', loss)
            self.log('val/dice', dice)
            for i in range(per_class_dice.shape[0]):
                self.log('val/dice_{}'.format(i), per_class_dice[i])

            # For 3D val resuts
            train_slice_info = self.trainer.datamodule.ds_train.slice_info
            self.val_data_list.append(x)
            self.val_target_list.append(y)
            self.val_logits_list.append(logits)
            self.val_dice_list.append(dice)
            self.val_per_class_dice_list.append(per_class_dice)
            self.val_slice += 1

            # We consider that all slices of a volume have been processed when we reach the last sample of the dataloader
            # or when the next sample comes from another volume/ has a different cur_scan_name (note that the slices must be sequential)
            cur_img_idx = img_idx.item()
            cur_scan_name = train_slice_info[str(cur_img_idx)]['scan_name']
            if (batch_idx + 1 == len(self.trainer.datamodule.val_loader)) or \
                (cur_scan_name != train_slice_info[str(cur_img_idx + 1)]['scan_name']):
                # We verify that we have the correct number of slices
                assert len(self.val_data_list) == self.trainer.datamodule.ds_train.scan_info[cur_scan_name]['num_slices']
                _vol_data = torch.stack(self.val_data_list, dim=0)
                _vol_target = torch.stack(self.val_target_list, dim=0)
                _vol_logits = torch.stack(self.val_logits_list, dim=0)
                vol_data = _vol_data.permute(1, 2, 0, 3, 4) # shape (BS=1, C, num_slices, H, W)
                vol_target = _vol_target.permute(1, 2, 0, 3, 4)
                vol_logits = _vol_logits.permute(1, 2, 0, 3, 4) 

                dice3d, per_class_dice3d, hausdorff953d, per_class_hausdorff953d = self._compute_metrics(vol_logits, vol_target)
                
                self.log('val3d/dice', dice3d)
                for i in range(per_class_dice3d.shape[0]):
                    self.log('val3d/dice_{}'.format(i), per_class_dice3d[i])
                    self.log('val3d/hausdorff95_{}'.format(i), per_class_hausdorff953d[i])
                self.log('val3d/hausdorff95', hausdorff953d)
                        
                self.val_data_list = []
                self.val_target_list = []
                self.val_logits_list = []
                self.val_dice_list = []
                self.val_per_class_dice_list = []
                self.val_slice = 0
            
        return loss
    
    def _test_step(self, batch, batch_idx):
        x, y, img_idx = batch['data'], batch['label'], batch['idx']
        logits, _ = self(x)
        loss = self._compute_loss(logits, y)
        dice, per_class_dice, hausdorff95, per_class_hausdorff95 = self._compute_metrics(logits, y)
        self.log('test/loss', loss)
        self.log('test/dice', dice)
        for i in range(per_class_dice.shape[0]):
            self.log('test/dice_{}'.format(i), per_class_dice[i])
        self.log('test/hausdorff95', hausdorff95)
        
        # For 3D test results
        test_slice_info = self.trainer.datamodule.ds_test.slice_info
        self.test_data_list.append(x)
        self.test_target_list.append(y)
        self.test_logits_list.append(logits)
        self.test_dice_list.append(dice)
        self.test_per_class_dice_list.append(per_class_dice)
        self.test_slice += 1
        
        # We consider that all slices of a volume have been processed when we reach the last sample of the dataloader
        # or when the next sample comes from another volume (note that the slices must be sequential)
        cur_img_idx = img_idx.item()
        cur_scan_name = test_slice_info[str(cur_img_idx)]['scan_name']
        if (batch_idx + 1 == len(self.trainer.datamodule.test_loader)) or \
            (cur_scan_name != test_slice_info[str(cur_img_idx + 1)]['scan_name']):
            assert len(self.test_data_list) == self.trainer.datamodule.ds_test.scan_info[cur_scan_name]['num_slices']
            _vol_data = torch.stack(self.test_data_list, dim=0)
            _vol_target = torch.stack(self.test_target_list, dim=0)
            _vol_logits = torch.stack(self.test_logits_list, dim=0)
            vol_data = _vol_data.permute(1, 2, 0, 3, 4) # shape (BS=1, num_slices, C, H, W)
            vol_target = _vol_target.permute(1, 2, 0, 3, 4)
            vol_logits= _vol_logits.permute(1, 2, 0, 3, 4)
            
            dice3d, per_class_dice3d, hausdorff953d, per_class_hausdorff953d = self._compute_metrics(vol_logits, vol_target)
            
            self.log('test3d/dice', dice3d)
            for i in range(per_class_dice3d.shape[0]):
                self.log('test3d/dice_{}'.format(i), per_class_dice3d[i])
                self.log('test3d/hausdorff95_{}'.format(i), per_class_hausdorff953d[i])
            self.log('test3d/hausdorff95', hausdorff953d)

            for slice in range(len(self.test_data_list)):
                title = 'test_vol{}_dice{:.2f}'.format(cur_scan_name, 
                                                    np.round(dice3d.detach().cpu().numpy(), 3)) 
                cur_vol_data, cur_vol_target, cur_vol_logits = vol_data[0, :, :, :, :], vol_target[0, :, :, :, :], vol_logits[0, :, :, :, :]
                subplot_title = '{:.2f}: '.format(self.test_dice_list[slice].item()) + ' & '.join([str(np.round(i.item(), 2)) for i in self.test_per_class_dice_list[slice]])
                plt = plot_data_pred_volume(cur_vol_data, cur_vol_target, cur_vol_logits, slice, plot_type=self.plot_type, title=subplot_title, vmin=0, vmax=self.out_channels - 1)
                
                # We save the figure on the logger
                if isinstance(self.logger, pl.loggers.CometLogger):
                    # Saving on comet_ml
                    self.logger.experiment.log_figure(figure=plt, figure_name=title, step=slice)
                else:
                    # Saving on TensorBoardLogger
                    self.logger.experiment.add_image(title + '_data', cur_vol_data[0, slice, :, :].detach().cpu().numpy(), slice, dataformats="WD")
                    self.logger.experiment.add_image(title + '_target', cur_vol_target[0, slice, :, :].detach().cpu().numpy(), slice, dataformats="WD")
                    self.logger.experiment.add_image(title + '_prediction', torch.argmax(cur_vol_logits[:, slice, :, :], dim=0).detach().cpu().numpy(), slice, dataformats="WD")
                plt.close()
                
            self.test_data_list = []
            self.test_target_list = []
            self.test_logits_list = []
            self.test_dice_list = []
            self.test_per_class_dice_list = []
            self.test_slice = 0
                
        return loss

    def training_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.train():
                return  self._training_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._training_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.validate():
                return  self._validation_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._validation_step(batch, batch_idx)
        
    def test_step(self, batch, batch_idx):
        # If logger is comet_ml
        if isinstance(self.logger, pl.loggers.CometLogger):
            with self.logger.experiment.test():
                return  self._test_step(batch, batch_idx)
        # If logger is the default TensorBoardLogger or another logger
        else:
            return self._test_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        base_sched = CosineAnnealingLR(optimizer, T_max=self.sched_config["max_epoch"] - self.sched_config["warmup_max"], 
                                                        eta_min=1e-7)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=self.sched_config['multiplier'], 
                                           total_epoch=self.sched_config["warmup_max"],
                                           after_scheduler=base_sched)
        scheduler = MultiStageScheduler(schedulers=[scheduler], 
                                        start_at_epochs=[0])

        scheduler = {
            "scheduler": scheduler,
            "interval": self.sched_config["update_interval"],
            "frequency": self.sched_config["update_freq"],
        }

        return [optimizer], [scheduler]
    
    

