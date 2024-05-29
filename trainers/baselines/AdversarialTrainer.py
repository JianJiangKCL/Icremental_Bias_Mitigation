import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import os
from trainers.TrainerABC import TrainerABC
import torch
from models.losses import get_binary_ocean_values, DIR_metric, log_DIR, log_gap
import wandb
from einops import rearrange
from models.baselines.adversial_debias import adversary_model
from models.optimizers import Lamb
from models.losses import log_DIR, log_gap, log_MSE_sensitive, log_MSE_personality


class AdversarialTrainer(TrainerABC):
	def __init__(self, args, backbone, adv_model, modalities, sensitive_groups):
		super(AdversarialTrainer, self).__init__(args, backbone=backbone, modalities=modalities)
		self.clf_model = backbone
		self.adv_model = adv_model
		self.target_sensitive_group = args.target_sensitive_group
		self.sensitive_groups = sensitive_groups
		self.adv_loss = F.binary_cross_entropy_with_logits
		self.adversary_loss_weight = 0.1
		# for manual optimization
		self.automatic_optimization = False

		print('baseline trainer')

	def configure_optimizers(self):
		self.clf_opt = Lamb(self.clf_model.parameters(), lr=self.args.lr)
		self.adv_opt = torch.optim.Adam(self.adv_model.parameters(), lr=0.001, weight_decay=1e-5)
		return [self.clf_opt, self.adv_opt]

	def training_step(self, batch, batch_idx):
		mode = 'train'
		x, label_ocean, label_sen_dict = batch
		modalities_x = {modality: x[modality] for modality in self.modalities}
		y = label_ocean[:, self.args.target_personality].unsqueeze(1)
		y_s = label_sen_dict[self.args.target_sensitive_group].float().unsqueeze(1)

		pred_ocean = self.clf_model.forward(modalities_x)[:, self.args.target_personality].unsqueeze(1)

		loss_mse = self.mse_loss(pred_ocean, y)
		loss_mse.backward(retain_graph=True)
		# tmp = self.clf_model.parameters()
		# for par in tmp:
		# 	if par.grad is None:
		# 		print('-----------------')
		# 		print(par)\
		# if par.grad is not None then add to clf_grad

		clf_grad = [
			torch.clone(par.grad.detach()) if par.grad is not None else torch.zeros_like(par) for par in self.clf_model.parameters()
		]
		self.clf_opt.zero_grad()
		self.adv_opt.zero_grad()
		self.metrics[mode].update(pred_ocean, y)

		pred_protected_attributes_labels, pred_protected_attributes_logits = self.adv_model.forward(pred_ocean, y)
		loss_adv = self.adv_loss(pred_protected_attributes_logits, y_s, reduction='mean')
		loss_adv.backward()
		adv_grad = [
			torch.clone(par.grad.detach()) for par in self.clf_model.parameters()
		]
		for i, par in enumerate(self.clf_model.parameters()):
			# Normalization
			unit_adversary_grad = adv_grad[i] / (torch.norm(adv_grad[i]) + torch.finfo(float).tiny)
			# projection proj_{dW_LA}(dW_LP)
			proj = torch.sum(torch.inner(unit_adversary_grad, clf_grad[i]))
			# integrating into the CLF gradient
			par.grad = clf_grad[i] - (proj * unit_adversary_grad) - (self.adversary_loss_weight * adv_grad[i])

		self.clf_opt.step()
		self.adv_opt.step()
		self.clf_opt.zero_grad()
		self.adv_opt.zero_grad()
			
		metric = self.metrics[mode].compute()
		log_data = {
			f'{mode}_loss': loss_mse,
			f'{mode}_metric': metric,
		}

		self.log_out(log_data, mode)

		prefix = '' if mode == 'train' else f'{mode}_'
		ret = {f'{prefix}loss': loss_mse, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
		       'label_ocean': y}  # , 'mse_wo_reduction': mse_wo_reduction}

		return ret

	def shared_step(self, batch, mode):
		x, label_ocean, label_sen_dict = batch
		modalities_x = {modality: x[modality] for modality in self.modalities}
		y = label_ocean[:, self.args.target_personality].unsqueeze(1)

		pred_ocean = self.clf_model.forward(modalities_x)
		pred_ocean = pred_ocean[:, self.args.target_personality].unsqueeze(1)

		loss_mse = self.mse_loss(pred_ocean, y)
		self.metrics[mode].update(pred_ocean, y)
		self.log_out({f'{mode}_loss': loss_mse}, mode)

		prefix = '' if mode == 'train' else f'{mode}_'
		ret = {f'{prefix}loss': loss_mse, 'label_sen_dict': label_sen_dict, 'pred_ocean': pred_ocean,
		       'label_ocean': y}
		return ret

	def shared_epoch_end(self, outputs, mode):
		local_rank = os.getenv("LOCAL_RANK", 0)
		metric = self.metrics[mode].compute()
		if local_rank == 0:
			# if mode == 'val' or mode == 'test':
			wandb.log({f'{mode}_mse': metric})  # this is the same as the loss_ocean
			# the mse is only for the target personality.
			wandb.log({f'{mode}_mse_{self.args.target_personality}': metric})
			# log_MSE_personality(outputs, mode, self.args.target_personality)
			for sensitive_group in self.sensitive_groups:
				log_DIR(outputs, sensitive_group, mode, self.args.target_personality)
				log_MSE_sensitive(outputs, sensitive_group, mode)

		self.metrics[mode].reset()
		if self.classification_metrics is not None:
			self.classification_metrics[mode].reset()


