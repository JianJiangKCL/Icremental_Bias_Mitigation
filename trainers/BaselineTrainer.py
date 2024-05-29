import torch.nn as nn
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap
import wandb
from einops import rearrange


class BaselineTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities, sensitive_groups):
        super(BaselineTrainer, self).__init__(args, backbone=backbone, modalities=modalities)
        self.target_sensitive_group = args.target_sensitive_group
        self.sensitive_groups = sensitive_groups
        print('baseline trainer')

    def shared_step(self, batch, mode):
        x, label_ocean, label_sen_dict = batch

        if self.args.arch == 'perceiver':
            modalities_x = {modality: x[modality] for modality in self.modalities}
            pred_ocean = self.backbone(modalities_x)
            # loss = self.mse_loss(pred_ocean, label_ocean)
        elif self.args.arch == 'infomax':
            modalities_x = {modality: rearrange(x[modality], 'b d () -> b  d') for modality in self.modalities}
            lld, nce, pred_ocean, pn_dic, H, _ = self.backbone(modalities_x)
            # alpha defaut 0.3; sigma default 0.1
            loss_info = self.args.alpha * nce - self.args.sigma * lld
            # loss = self.mse_loss(pred_ocean, label_ocean)


        if self.args.target_personality is not None:
            loss_mse = self.mse_loss(pred_ocean[:, self.args.target_personality], label_ocean[:, self.args.target_personality])
            self.metrics[mode].update(pred_ocean[:, self.args.target_personality], label_ocean[:, self.args.target_personality])

        else:
            loss_mse = self.mse_loss(pred_ocean, label_ocean)
            self.metrics[mode].update(pred_ocean, label_ocean)

        if self.args.arch == 'perceiver':
            loss = loss_mse
        elif self.args.arch == 'infomax':
            loss = loss_mse + loss_info


        metric = self.metrics[mode].compute()
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
        }
        
        self.log_out(log_data, mode)
        # compute mse without reduction
        # with torch.no_grad():
        #     mse_wo_reduction = self.mse_wo_reduction(pred_ocean, label_ocean)

        prefix = '' if mode == 'train' else f'{mode}_'

        if self.args.target_personality is not None:
            ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict,
                   'pred_ocean': pred_ocean[:, self.args.target_personality],
                   'label_ocean': label_ocean[:, self.args.target_personality]}
        else:
            ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
                   'label_ocean': label_ocean}

        return ret

    # def shared_epoch_end(self, outputs, mode):
    #     local_rank = os.getenv("LOCAL_RANK", 0)
    #     metric = self.metrics[mode].compute()
    #     if local_rank == 0:
    #         wandb.log({f'{mode}_mse': metric})
    #         log_DIR(outputs, mode)
    #         log_gap(outputs, mode)
    #
    #     self.metrics[mode].reset()


