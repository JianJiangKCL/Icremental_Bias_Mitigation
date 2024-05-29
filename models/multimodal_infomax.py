import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F



class SubNet(nn.Module):
	'''
	The subnetwork that is used in TFN for video and audio in the pre-fusion stage
	'''

	def __init__(self, in_size, hidden_size, n_class, dropout, modal_name='text'):
		'''
		Args:
			in_size: input dimension
			hidden_size: hidden layer dimension
			dropout: dropout probability
		Output:
			(return value in forward) a tensor of shape (batch_size, hidden_size)
		'''
		super(SubNet, self).__init__()
		# self.norm = nn.BatchNorm1d(in_size)
		self.drop = nn.Dropout(p=dropout)
		self.linear_1 = nn.Linear(in_size, hidden_size)
		self.linear_2 = nn.Linear(hidden_size, hidden_size)
		self.linear_3 = nn.Linear(hidden_size, n_class)

	def forward(self, x):
		'''
		Args:
			x: tensor of shape (batch_size, in_size)
		'''
		# normed = self.norm(x)
		dropped = self.drop(x)
		y_1 = torch.tanh(self.linear_1(dropped))
		# fusion = self.linear_2(y_1)
		y_2 = torch.tanh(self.linear_2(y_1))
		y_3 = self.linear_3(y_2)
		return y_2, y_3


class CPC(nn.Module):
	"""
		Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

		Args:
			x_size (int): embedding size of input modality representation x
			y_size (int): embedding size of input modality representation y
	"""

	def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
		super().__init__()
		self.x_size = x_size
		self.y_size = y_size
		self.layers = n_layers
		self.activation = getattr(nn, activation)
		if n_layers == 1:
			self.net = nn.Linear(
				in_features=y_size,
				out_features=x_size
			)
		else:
			net = []
			for i in range(n_layers):
				if i == 0:
					net.append(nn.Linear(self.y_size, self.x_size))
					net.append(self.activation())
				else:
					net.append(nn.Linear(self.x_size, self.x_size))
			self.net = nn.Sequential(*net)

	def forward(self, x, y):
		"""Calulate the score
		"""
		# import ipdb;ipdb.set_trace()
		x_pred = self.net(y)  # bs, emb_size

		# normalize to unit sphere
		x_pred = x_pred / x_pred.norm(dim=1, keepdim=True)
		x = x / x.norm(dim=1, keepdim=True)

		pos = torch.sum(x * x_pred, dim=-1)  # bs
		neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)  # bs
		nce = -(pos - neg).mean()
		return nce


class MMILB(nn.Module):
	"""Compute the Modality Mutual Information Lower Bound (MMILB) given bimodal representations.
	Args:
		x_size (int): embedding size of input modality representation x
		y_size (int): embedding size of input modality representation y
		mid_activation(int): the activation function in the middle layer of MLP
		last_activation(int): the activation function in the last layer of MLP that outputs logvar
	"""
	def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
		super(MMILB, self).__init__()
		try:
			self.mid_activation = getattr(nn, mid_activation)
			self.last_activation = getattr(nn, last_activation)
		except:
			raise ValueError("Error: CLUB activation function not found in torch library")
		self.mlp_mu = nn.Sequential(
			nn.Linear(x_size, y_size),
			self.mid_activation(),
			nn.Linear(y_size, y_size)
		)
		self.mlp_logvar = nn.Sequential(
			nn.Linear(x_size, y_size),
			self.mid_activation(),
			nn.Linear(y_size, y_size),
		)
		self.entropy_prj = nn.Sequential(
			nn.Linear(y_size, y_size // 4),
			nn.Tanh()
		)
		k=1

	def forward(self, x, y, labels=None, mem=None):
		""" Forward lld (gaussian prior) and entropy estimation, partially refers the implementation
		of https://github.com/Linear95/CLUB/blob/master/MI_DA/MNISTModel_DANN.py
			Args:
				x (Tensor): x in above equation, shape (bs, x_size)
				y (Tensor): y in above equation, shape (bs, y_size)
		"""
		mu, logvar = self.mlp_mu(x), self.mlp_logvar(x) # (bs, hidden_size)
		batch_size = mu.size(0)

		positive = -(mu - y)**2/2./torch.exp(logvar)
		lld = torch.mean(torch.sum(positive,-1))

		# For Gaussian Distribution Estimation
		pos_y = neg_y = None
		H = 0.0
		sample_dict = {'pos':None, 'neg':None}

		if labels is not None:
			# store pos and neg samples
			y = self.entropy_prj(y)
			pos_y = y[labels.squeeze() > 0]
			neg_y = y[labels.squeeze() < 0]

			sample_dict['pos'] = pos_y
			sample_dict['neg'] = neg_y

			# estimate entropy
			if mem is not None and mem.get('pos', None) is not None:
				pos_history = mem['pos']
				neg_history = mem['neg']

				# Diagonal setting
				# pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
				# neg_all = torch.cat(neg_history + [neg_y], dim=0)
				# mu_pos = pos_all.mean(dim=0)
				# mu_neg = neg_all.mean(dim=0)

				# sigma_pos = torch.mean(pos_all ** 2, dim = 0) - mu_pos ** 2 # (embed)
				# sigma_neg = torch.mean(neg_all ** 2, dim = 0) - mu_neg ** 2 # (embed)
				# H = 0.25 * (torch.sum(torch.log(sigma_pos)) + torch.sum(torch.log(sigma_neg)))

				# compute the entire co-variance matrix
				pos_all = torch.cat(pos_history + [pos_y], dim=0) # n_pos, emb
				neg_all = torch.cat(neg_history + [neg_y], dim=0)
				mu_pos = pos_all.mean(dim=0)
				mu_neg = neg_all.mean(dim=0)
				sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
				sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
				a = 17.0795
				H = 0.25 * (torch.logdet(sigma_pos) + torch.logdet(sigma_neg))

		return lld, sample_dict, H


class MMIM(nn.Module):
	def __init__(self, args):
		"""Construct MultiMoldal InfoMax model.
		Args:
			args (dict): a dict stores training and model configurations
		"""
		# Base Encoders
		super().__init__()
		self.args = args
		# self.add_va = args.add_va # visual_audion
		d_tout = 512
		d_vout = 552
		d_aout = 59


		# For MI maximization
		self.mi_tv = MMILB(
			x_size=d_tout,#args.d_tout,
			y_size=d_vout,
			mid_activation='ReLU',
			last_activation='Tanh'
		)

		self.mi_ta = MMILB(
			x_size=d_tout,
			y_size=d_aout,
			mid_activation='ReLU',
			last_activation='Tanh'
		)

		dim_sum = d_aout + d_vout + d_tout

		# CPC MI bound
		self.cpc_zt = CPC(
			x_size=d_tout,  # to be predicted
			y_size=args.d_prjh,
			n_layers=args.cpc_layers,
			activation='Tanh'
		)
		self.cpc_zv = CPC(
			x_size=d_vout,
			y_size=args.d_prjh,
			n_layers=args.cpc_layers,
			activation='Tanh'
		)
		self.cpc_za = CPC(
			x_size=d_aout,
			y_size=args.d_prjh,
			n_layers=args.cpc_layers,
			activation='Tanh'
		)

		# Trimodal Settings
		self.fusion_prj = SubNet(
			in_size=dim_sum,
			hidden_size=args.d_prjh,
			n_class=args.num_outputs,
			dropout=args.dropout_prj
		)

	def forward(self, multi_modality_data, y=None,
				mem=None):
		"""
		text, audio, and vision should have dimension [batch_size, seq_len, n_features]
		For Bert input, the length of text is "seq_len + 2"
		"""

		text = multi_modality_data['text']
		acoustic = multi_modality_data['audio']
		visual = multi_modality_data['facebody']


		if y is not None:
			lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual, labels=y, mem=mem['tv'])
			lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic, labels=y, mem=mem['ta'])
			# for ablation use

		else:
			lld_tv, tv_pn, H_tv = self.mi_tv(x=text, y=visual)
			lld_ta, ta_pn, H_ta = self.mi_ta(x=text, y=acoustic)


		# Linear proj and pred
		fusion, preds = self.fusion_prj(torch.cat([text, acoustic, visual], dim=1))

		nce_t = self.cpc_zt(text, fusion)
		nce_v = self.cpc_zv(visual, fusion)
		nce_a = self.cpc_za(acoustic, fusion)

		nce = nce_t + nce_v + nce_a

		pn_dic = {'tv': tv_pn, 'ta': ta_pn, }
		lld = lld_tv + lld_ta
		H = H_tv + H_ta

		return lld, nce, preds, pn_dic, H, fusion

	# def extract_features(self, multi_modality_data, y=None, mem=None):
	# 	"""
	# 	text, audio, and vision should have dimension [batch_size, seq_len, n_features]
	# 	For Bert input, the length of text is "seq_len + 2"
	# 	"""
	#
	# 	return self.forward(multi_modality_data, y, mem)
