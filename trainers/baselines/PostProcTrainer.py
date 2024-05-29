import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap, DIR_metric_three_way
import wandb
from einops import rearrange
from models.losses import log_DIR_v2, log_gap, log_MSE_sensitive, log_MSE_personality, separate_binary_label_group
from models.baselines.postp_claib_eq_odds import PostModel
import numpy as np


class PostProcTrainer(TrainerABC):
    def __init__(self, args, backbone, modalities, sensitive_groups):
        super(PostProcTrainer, self).__init__(args, backbone=backbone, modalities=modalities)
        self.target_sensitive_group = args.target_sensitive_group
        self.sensitive_groups = sensitive_groups
        self.backbone = backbone
        self.is_postprocess = False
        self.is_incremental = False
        print('postProc trainer')

    def set_postprocess(self, is_postprocess):
        self.is_postprocess = is_postprocess

    def set_incremental(self, is_incremental):
        self.is_incremental = is_incremental

    def set_target_sensitive_group(self, next_target_sensitive_group):
        self.target_sensitive_group = next_target_sensitive_group

    def shared_step(self, batch, mode):

        x, label_ocean, label_sen_dict = batch
        modalities_x = {modality: x[modality] for modality in self.modalities}
        y = label_ocean[:, self.args.target_personality].unsqueeze(1)
       
        pred_ocean = self.backbone.forward(modalities_x)
        pred_ocean = pred_ocean[:, self.args.target_personality].unsqueeze(1)

        loss = self.mse_loss(pred_ocean, y)
        self.metrics[mode].update(pred_ocean, y)
        metric = self.metrics[mode].compute()
        log_data = {
            f'{mode}_loss': loss,
            f'{mode}_metric': metric,
        }
        
        self.log_out(log_data, mode)

        prefix = '' if mode == 'train' else f'{mode}_'
        ret = {f'{prefix}loss': loss, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
               'label_ocean': y} #, 'mse_wo_reduction': mse_wo_reduction}

        return ret


    # def post_process(self):

    def shared_epoch_end(self, outputs, mode):
        if not self.is_postprocess:
            local_rank = os.getenv("LOCAL_RANK", 0)
            metric = self.metrics[mode].compute()
            if local_rank == 0:
                # if mode == 'val' or mode == 'test':
                wandb.log({f'{mode}_mse': metric})  # this is the same as the loss_ocean
                log_MSE_personality(outputs, mode, self.args.target_personality)
                for sensitive_group in self.sensitive_groups:
                    log_DIR(outputs, sensitive_group, mode, self.args.target_personality)
                    log_MSE_sensitive(outputs, sensitive_group, mode)

            self.metrics[mode].reset()
            if self.classification_metrics is not None:
                self.classification_metrics[mode].reset()
        else:
            if not self.is_incremental:
                pred_ocean = torch.cat([output['pred_ocean'] for output in outputs])
                # binary_pred_ocean = get_binary_ocean_values(pred_ocean, STE=False)
                pred_ocean = pred_ocean.cpu()
                label_ocean = torch.cat([output['label_ocean'] for output in outputs])
                log_prefix = f'first_{self.target_sensitive_group}'

            else:
                if mode == 'val':
                    pred_ocean = self.last_val_pred_ocean
                elif mode == 'test':
                    pred_ocean = self.last_test_pred_ocean
                log_prefix = f'second_{self.target_sensitive_group}'
            # label remains unchanged forever
            label_ocean = torch.cat([output['label_ocean'] for output in outputs])
            label_ocean = label_ocean.cpu()
            sensitive_labels = torch.cat([output['label_sen_dict'][self.target_sensitive_group] for output in outputs])

            binary_label_ocean = get_binary_ocean_values(label_ocean, STE=False,
                                                         target_personality=self.args.target_personality)
            # next_target_personality = self.sensitive_groups.copy()
            # next_target_personality.remove(self.args.target_sensitive_group)
            preds_0, preds_1 = separate_binary_label_group(pred_ocean, sensitive_labels)
            labels_0, labels_1 = separate_binary_label_group(binary_label_ocean, sensitive_labels)
            tn_rate = 0
            tp_rate = 1
            post_model_0 = PostModel(preds_0, labels_0, target_personality=self.args.target_personality)
            # post_model_0.init()
            post_model_1 = PostModel(preds_1, labels_1, target_personality=self.args.target_personality)
            # post_model_1.init()
            if mode == 'val':
                _, _, mix_rates = PostModel.calib_eq_odds(post_model_0, post_model_1, tp_rate, tn_rate)
                self.mix_rates = mix_rates
                if not self.is_incremental:
                    # for next stage validation
                    calib_eq_odds_group_0_val_model, calib_eq_odds_group_1_val_model = PostModel.calib_eq_odds(
                        post_model_0,
                        post_model_1,
                        tp_rate, tn_rate,
                        self.mix_rates)
                    all_calib_pred = np.concatenate(
                        (calib_eq_odds_group_0_val_model.pred, calib_eq_odds_group_1_val_model.pred))
                    all_calib_pred = torch.from_numpy(all_calib_pred).float()
                    self.last_val_pred_ocean = all_calib_pred

            elif mode == 'test':
                calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model = PostModel.calib_eq_odds(
                    post_model_0,
                    post_model_1,
                    tp_rate, tn_rate,
                    self.mix_rates)
                all_calib_pred = np.concatenate((calib_eq_odds_group_0_test_model.pred, calib_eq_odds_group_1_test_model.pred))
                all_calib_pred = torch.from_numpy(all_calib_pred).float()
                if not self.is_incremental:
                    self.last_test_pred_ocean = all_calib_pred

                binary_all_calib_pred = get_binary_ocean_values(all_calib_pred, STE=False, target_personality=self.args.target_personality)
                # new mse results; all_calib_pred, label_ocean are both [B,1]
                MSE = F.mse_loss(all_calib_pred, label_ocean, reduction='mean')
                wandb.log({f'{log_prefix}_{mode}_mse': MSE})
                wandb.log({f'{log_prefix}_{mode}_mse_{self.args.target_personality}': MSE})
                # log_MSE_personality(outputs, mode, target_personality=self.args.target_personality, log_prefix=log_prefix)
                for sensitive_group in self.sensitive_groups:
                    print(f'---------------------{sensitive_group}---------------------')
                    sensitive_labels = torch.cat([output['label_sen_dict'][sensitive_group] for output in outputs])
                    log_DIR_v2(binary_all_calib_pred, sensitive_labels, sensitive_group, mode, target_personality=self.args.target_personality, log_prefix=log_prefix)


                # # new DIR results
                # for sensitive_group in self.sensitive_groups:
                #     metric_name = ['O', 'C', 'E', 'A', 'N']
                #     if self.args.target_sensitive_group == 'ethnicity':
                #         DIRs, SPDs, list_p_privileged = DIR_metric_three_way(binary_all_calib_pred, sensitive_labels, self.args.target_personality)
                #     else:
                #         DIRs, SPDs, list_p_privileged = DIR_metric(binary_all_calib_pred, sensitive_labels, self.args.target_personality)
                #     t =1
                #
                #     k=1
                #
                #     wandb.log({f'{self.args.target_sensitive_group}_{mode}_DIR_{self.args.target_personality}': DIRs})









