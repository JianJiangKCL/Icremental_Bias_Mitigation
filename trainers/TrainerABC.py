import pytorch_lightning as pl
from pytorch_lightning.utilities import AttributeDict
from funcs.module_funcs import setup_optimizer, setup_scheduler
from models.perceiver import Perceiver
import os
import wandb
from models.losses import log_DIR, log_gap, log_MSE_sensitive, log_MSE_personality
import torchmetrics
import torch.nn as nn


class TrainerABC(pl.LightningModule):
    args: AttributeDict
    backbone: Perceiver
    def __init__(self, args, backbone=None, modalities=None):
        super(TrainerABC, self).__init__()
        self.train_metric = torchmetrics.MeanSquaredError()
        self.val_metric = torchmetrics.MeanSquaredError()
        self.test_metric = torchmetrics.MeanSquaredError()
        self.classification_metrics = None
        self.metrics = {'train': self.train_metric, 'val': self.val_metric, 'test': self.test_metric}
        self.mse_loss = nn.MSELoss()
        self.mse_wo_reduction = nn.MSELoss(reduction='none')
        self.modalities = sorted(modalities)
        self.args = args
        self.backbone = backbone

    def configure_optimizers(self):
        opt = setup_optimizer(self.args, self.backbone)
        scheduler = setup_scheduler(self.args, opt, milestones=self.args.milestones)
        if scheduler is None:
            return opt
        return [opt], [scheduler]

    def shared_step(self, batch, mode):
        loss = None
        ret = {'loss': loss}
        # the returns are dict so cannot be stacked directly
        return ret

    def shared_epoch_end(self, outputs, mode):
        local_rank = os.getenv("LOCAL_RANK", 0)
        metric = self.metrics[mode].compute()
        if local_rank == 0:
            # if mode == 'val' or mode == 'test':
            wandb.log({f'{mode}_mse': metric})  # this is the same as the loss_ocean
            if self.args.target_personality is None:
                log_MSE_personality(outputs, mode)
            else:
                wandb.log({f'{mode}_mse_{self.args.target_personality}': metric})
            for sensitive_group in self.sensitive_groups:
                log_DIR(outputs, sensitive_group, mode, self.args.target_personality)
                # log_gap(outputs, sensitive_group, mode, target_personality=self.args.target_personality)
                log_MSE_sensitive(outputs, sensitive_group, mode)


        self.metrics[mode].reset()
        if self.classification_metrics is not None:
            self.classification_metrics[mode].reset()

    def training_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'train')

        # must return loss
        return ret

    def training_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'train')

    def validation_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'val')
        return ret

    def validation_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'val')
        # calculate per mse


    def test_step(self, batch, batch_idx):
        ret = self.shared_step(batch, 'test')
        return ret

    def test_epoch_end(self, outputs):
        self.shared_epoch_end(outputs, 'test')

    def log_out(self, log_data, mode):
        if mode == 'train':
            log_data['lr'] = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict(log_data, prog_bar=not self.args.disable_tqdm, sync_dist=False if mode == 'train' else True,
                      on_step=True if mode == 'train' else False, on_epoch=False if mode == 'train' else True)

