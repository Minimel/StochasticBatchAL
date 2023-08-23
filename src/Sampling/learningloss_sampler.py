"""
Reference:
- Learning Loss for Active Learning, Yoo et al. (2019) (https://arxiv.org/abs/1905.03677)
- Github Repo: https://github.com/Mephisto405/Learning-Loss-for-Active-Learning (as of May 31st 2021)
"""
import time
import torch

from Utils.plot_utils import plot_all_uncertain_samples_from_lists


class LearningLossSampler:
    """
    Sampler for learning loss method
    """

    def __init__(self, budget, trainer, **kwargs):
        self.budget = budget
        self.trainer = trainer
        self.model = self.trainer.model
        self.normalize_fct = self.model.model_loss.normalize_fct
        self.loss_module = self.trainer.model.loss_module

        self.device = kwargs.get('device')

        self.model.to(self.device)
        self.loss_module.to(self.device)

    def get_uncertainty(self, unlabeled_dataloader):

        sampling_start_time = time.time()

        self.model.eval()
        self.loss_module.eval()

        # We make the list of indices from the dataloader
        indice_list = []
        uncertainty_list = []
        data_list = []
        logits_list = []
        target_list = []

        for batch_idx, batch in enumerate(unlabeled_dataloader):
            with torch.no_grad():
                x, y, img_idx = batch['data'], batch['label'], batch['idx']
                x = x.to(self.device)

                # We get the uncertainty based on predicted loss
                logits, [enc1, enc2, enc3, center, _, _, _] = self.model(x)
                features = [enc1, enc2, enc3, center]
                pred_loss = self.loss_module(features)

                # We keep track of the uncertainty (single value per image)
                uncertainty_list.extend(pred_loss.flatten().tolist())

                # We keep track of all dataloader results
                indice_list.extend(img_idx.detach().cpu().numpy())
                data_list.extend(x.detach().cpu().numpy())
                logits_list.extend(logits.detach().cpu().numpy())
                target_list.extend(y.detach().cpu().numpy())

        sampling_time = time.time() - sampling_start_time

        # We plot top uncertainty samples and their uncertainty map
        plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_list, 
                                              self.budget, self.trainer, title='Query images', 
                                              model_out_channels=self.model.out_channels)

        return indice_list, uncertainty_list, sampling_time
