import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values
from funcs.module_funcs import setup_optimizer, setup_scheduler
from models.optimizers import Lamb
import numpy as np

# manual backpropagation
class PreBiasMimicTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities, sensitive_groups):
        super(PreBiasMimicTrainer, self).__init__(args, backbone=backbone, modalities=modalities)

        self.sensitive_groups = sensitive_groups
        self.criterion = {
                'bin': nn.BCEWithLogitsLoss(reduction='none'),
                'multi': nn.MSELoss(reduction='none')
                }
        print('PreBiasMimicTrainer')
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):

        opt_tmp = Lamb(self.backbone.parameters(), lr=self.args.lr)
        opt = Lamb(self.backbone.to_logits.parameters(), lr=self.args.lr)
        scheduler_tmp = setup_scheduler(self.args, opt_tmp, milestones=self.args.milestones)
        scheduler = setup_scheduler(self.args, opt, milestones=self.args.milestones)
        # if scheduler is None:
        #     return [{'tmp_opt': opt_tmp, 'opt': opt}]
        # return [{'tmp_opt': opt_tmp, 'opt': opt}], [{'scheduler_tmp': scheduler_tmp, 'scheduler': scheduler}]
        if scheduler is None:
            return [opt_tmp, opt]
        return [opt_tmp, opt], [scheduler_tmp, scheduler]

    def training_epoch_end(self, outputs):
        feat = torch.cat([output['feat'] for output in outputs])
        label_ocean = torch.cat([output['label_ocean'] for output in outputs])
        bs = 128
        total_samples = feat.shape[0]
        num_batches = total_samples // bs
        for batch_idx in range(num_batches):
            all_idx = torch.randperm(total_samples).cuda()

            feat[all_idx] = feat.clone()
            label_ocean[all_idx] = label_ocean.clone()
            feat_batch = feat[batch_idx * bs: (batch_idx + 1) * bs]
            label_ocean_batch = label_ocean[batch_idx * bs: (batch_idx + 1) * bs]
            pred_ocean_batch = self.backbone.to_logits(feat_batch)
            loss_ocean_batch = self.criterion['multi'](pred_ocean_batch, label_ocean_batch)
            loss_ocean_batch = loss_ocean_batch.mean()
            self.manual_backward(loss_ocean_batch)
            self.optimizers()[1].step()
            self.optimizers()[1].zero_grad()
        # step scheduler
        if self.args.scheduler != 'constant':
            self.lr_schedulers()[0].step()
            self.lr_schedulers()[1].step()

    def shared_step(self, batch, mode):

        x, label_ocean, label_sen_dict = batch

        sample_weights = torch.ones_like(label_ocean[:, 0])

        modalities_x = {modality: x[modality] for modality in self.modalities}
        pred_ocean = self.backbone(modalities_x)

        y = label_ocean[:, self.args.target_personality].unsqueeze(1)


        loss_mse = self.mse_loss(pred_ocean, y)
        self.metrics[mode].update(pred_ocean, y)


        loss = loss_mse * sample_weights
        loss = loss.mean()


        metric = self.metrics[mode].compute()
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
        }

        self.log_out(log_data, mode)

        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
               'label_ocean': y}  # , 'mse_wo_reduction': mse_wo_reduction}

        return ret

    # def on_train_start(self):
    #     self.
    def training_step(self, batch, batch_idx):

        tmp_opt = self.optimizers()[0]
        x, label_ocean, label_sen_dict, label_bm_target_sen = batch

        modalities_x = {modality: x[modality] for modality in self.modalities}
        feat = self.backbone.extract_features(modalities_x)
        pred_ocean = self.backbone.tmp_to_logits(feat)
        y = label_ocean[:, self.args.target_personality].unsqueeze(1)
        # pred_ocean = pred_ocean[:, self.args.target_personality].unsqueeze(1)
        # binary y
        labels_bin = get_binary_ocean_values(y, target_personality=self.args.target_personality)

        multi = torch.ones_like(labels_bin)

        multi[labels_bin == -1] = 0
        labels_bin[labels_bin == -1] = 0
        loss = self.criterion['bin'](pred_ocean, labels_bin)
        loss = loss * multi
        div = torch.sum(multi)
        loss = torch.sum(loss / div)

        self.manual_backward(loss)
        tmp_opt.step()
        tmp_opt.zero_grad()



        prefix = 'train'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'feat': feat.detach(),
               'label_ocean': y}  # , 'mse_wo_reduction': mse_wo_reduction}

        return ret


