import torch
import torch.utils.data
import random
from scipy.optimize import linprog
import numpy as np

import torch.nn as nn
import pandas as pd
from models.losses import get_binary_ocean_values
from einops import rearrange
from models.multi_modality_perceiver import MultiModalityPerceiver
from typing import Dict


# class SamplingDataset():

# can only handle one sensitive attribute


class BiasMimicDatasetPreprocessor(nn.Module):
	# only support single task setting
	# learn the transform on the training set and apply it on the train set and test set
	def __init__(self):
		super().__init__()

	# creat new copy of the dataset, as the distribution is changed
	def preprocess_dataset_(self, train_ds, modalities, attribute, target_personality):

		print("----Preprocessing dataset for sensitive attributes: ", attribute)
		X_train_dict, y_train, S_train_dict = train_ds.get_all_data()

		# binary y_train and set to float
		y_train = get_binary_ocean_values(y_train, STE=False)

		y_train = y_train[:, target_personality]
		self.targets = y_train
		# make a copy of y_train
		# self.bm_st = y_train.clone()
		self.bm_st = self.get_targets_bin()

		S_train = S_train_dict[attribute]
		self.sensitive_targets = S_train

		self.set_dro_info()
		self.bias_mimick()
		self.print_new_distro()
		# pass bm_st to the dataset
		train_ds.init_bias_mimic(self.bm_st)

	def set_dro_info(self):
		num_targets = 2
		num_biases = 2

		self.groups_idx = torch.zeros((len(self.targets)))
		for i, t, b in zip(torch.arange((len(self.targets))), self.targets, self.sensitive_targets):
			idx = t + (b * num_targets)
			self.groups_idx[i] = idx

		self.n_groups = num_targets * num_biases

	def solve_linear_program(self, target_distro, target_prime_distro):
		num_biases = len(torch.unique(self.sensitive_targets))
		obj = [-1] * num_biases

		lhs_ineq = []
		for bias in range(num_biases):
			ineq = [0] * num_biases
			ineq[bias] = 1
			lhs_ineq.append(ineq)

		rhs_ineq = target_prime_distro

		lhs_eq = []
		target_distro = [x / sum(target_distro) for x in target_distro]
		for prob, bias in zip(target_distro, range(num_biases - 1)):
			eq = [-prob] * num_biases
			eq[bias] = 1 - prob
			lhs_eq.append(eq)

		rhs_eq = [0] * (num_biases - 1)

		bnd = [(0, float("inf")) for _ in range(num_biases)]

		opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq,
		              A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd,
		              method="revised simplex")

		sol = opt.x
		sol = [int(x) for x in sol]
		sol = [x if x > 0 else 1 for x in sol]
		return sol

	def get_target_distro(self, target):
		num_biases = 2
		target_distro = []
		for bias in range(num_biases):
			target_distro.append(torch.sum(torch.logical_and(self.targets == target, self.sensitive_targets == bias)))

		return target_distro

	def get_kept_indices(self, target, target_prime, target_prime_new_distro):

		to_keep_indices = []
		for bias, bias_distro in enumerate(target_prime_new_distro):
			tmp = torch.logical_and(self.targets == target_prime, self.sensitive_targets == bias)
			indices_bias = list(torch.arange(len(self.targets))[tmp].numpy())
			to_keep_indices.extend(random.sample(indices_bias, bias_distro))

		return to_keep_indices

	def bias_mimick(self):

		num_targets = 2
		num_biases = 2

		for target in range(num_targets):
			target_distro = self.get_target_distro(target)
			to_keep_indices = []
			for target_prime in range(num_targets):

				if target_prime == target:
					indices_target = list(torch.arange(len(self.targets))[self.targets == target])
					to_keep_indices.extend(indices_target)
				else:
					target_prime_distro = self.get_target_distro(target_prime)
					target_prime_new_distro = self.solve_linear_program(target_distro, target_prime_distro)
					to_keep_indices.extend(self.get_kept_indices(target, target_prime, target_prime_new_distro))

			full_idxs = torch.arange((len(self.targets)))
			to_select = torch.ones((len(self.targets)))
			to_select[to_keep_indices] = 0
			full_idxs = full_idxs[to_select.bool()]

			self.bm_st[full_idxs, target] = -1

	def get_bm_st_distro(self, target_bin, num_targets, num_biases):

		distro = []
		for target in range(num_targets):
			target_distro = []
			for bias in range(num_biases):
				count = torch.logical_and(self.targets == target, self.sensitive_targets == bias)
				count = torch.logical_and(count, self.bm_st[:, target_bin] != -1)
				target_distro.append(torch.sum(count))
			distro.append(target_distro)

		return distro

	def print_new_distro(self):

		num_targets = 2
		num_biases = 2

		print('===================================')
		print("Binary Labels Distribution: ")
		for target_idx in range(num_targets):

			print(f'Binary Target {target_idx}')
			print('---------------------------')
			target_distro = self.get_bm_st_distro(target_idx, num_targets, num_biases)
			for target, distro in enumerate(target_distro):
				print(f"Target {target}: {[x.item() for x in distro]}")

		print('===================================')
		print("Normal Label Distribution: ")
		for target in range(num_targets):
			target_distro = self.get_target_distro(target)
			print(f"Target {target}: {[x.item() for x in target_distro]}")

		print('===================================')

	def get_targets_bin(self):
		num_targets = len(torch.unique(self.targets))
		targets_ = torch.zeros((len(self.targets), num_targets))
		targets_[torch.arange((len(self.targets, ))), self.targets] = 1
		return targets_



