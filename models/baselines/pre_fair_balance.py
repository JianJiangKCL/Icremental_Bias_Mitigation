from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from datasets.data_module import MultiTaskDataset
import numpy as np
import torch.nn as nn
import pandas as pd
from models.losses import get_binary_ocean_values
from einops import rearrange
from models.multi_modality_perceiver import MultiModalityPerceiver
from typing import Dict
# class MultiModalityPerceiverWithSampleWeight(MultiModalityPerceiver):


# Reproduced for Fairer Machine Learning Software on Multiple Sensitive Attributes With Data Preprocessing
# adapted from https://github.com/hil-se/FairBalance
def FairBalance(X, y, A):
	# X: independent variables (2-d pd.DataFrame)
	# y: the dependent variable (1-d np.array)
	# A: the name of the sensitive attributes (list of string)
	groups_class = {}
	group_weight = {}
	for i in range(len(y)):
		key_class = tuple([X[a][i] for a in A] + [y[i]])
		key = key_class[:-1]
		if key not in group_weight:
			group_weight[key] = 0
		group_weight[key] += 1
		if key_class not in groups_class:
			groups_class[key_class] = []
		groups_class[key_class].append(i)
	sample_weight = np.array([1.0]*len(y))
	for key in groups_class:
		weight = group_weight[key[:-1]]/len(groups_class[key])
		for i in groups_class[key]:
			sample_weight[i] = weight
	# Rescale the total weights to len(y)
	sample_weight = sample_weight * len(y) / sum(sample_weight)
	return sample_weight


class FairBalanceDatasetPreprocessor(nn.Module):
	# only support single task setting
	# learn the transform on the training set and apply it on the train set and test set
	def __init__(self):
		super().__init__()

	def preprocess_dataset_(self, datasets:Dict, modalities, A, target_personality):
		# datasets: a dictionary of datasets; trainval, test
		# modalities are already normalized
		# A are sensitive attribute names
		# A = ["sex", "race"]
		self.A = A
		print("----Preprocessing dataset for sensitive attributes: ", A)
		train_ds, test_ds = datasets["train_val"], datasets["test"]
		X_train_dict, y_train, S_train_dict = train_ds.get_all_data()

		# binary y_train and set to float
		y_train = get_binary_ocean_values(y_train, STE=False)

		y_train = y_train.numpy()[:, target_personality].astype(float)
		# stack all modalities
		X_train_tmp = np.hstack([X_train_dict[modality] for modality in modalities])
		# the length of the first dimension of X_train_tmp is the number of samples
		columns = [str(i) for i in range(X_train_tmp.shape[1])]
		# convert to dataframe, with column dtype as string
		X_train = pd.DataFrame(X_train_tmp, columns=columns)
		# add sensitive attributes to X_train
		for a in A:
			X_train[a] = S_train_dict[a].numpy().astype(float)

		# data_preprocess; treat;  self.preprocessor.fit_transform
		self.data_preprocess(X_train)
		sample_weight = self.treat(X_train, y_train)
		# learn the transform
		X_train_processed = self.preprocessor.fit_transform(X_train)

		# convert X_train_processed back to dict
		X_train_processed_dict = {}
		for modality in X_train_dict.keys():
			# the sensitive attributes are not included in X_train_processed
			X_train_processed_dict[modality] = X_train_processed[:, :X_train_dict[modality].shape[1]].astype(dtype=np.float32)

		# transform test set
		X_test_dict, y_test, S_test_dict = test_ds.get_all_data()
		X_test_tmp = np.hstack([X_test_dict[modality] for modality in modalities])
		columns = [str(i) for i in range(X_test_tmp.shape[1])]
		X_test = pd.DataFrame(X_test_tmp, columns=columns)
		for a in A:
			X_test[a] = S_test_dict[a].numpy().astype(float)

		# apply the learned transform
		X_test_processed = self.preprocessor.transform(X_test)
		X_test_processed_dict = {}
		for modality in X_test_dict.keys():
			# the sensitive attributes are not included in X_test_processed
			X_test_processed_dict[modality] = X_test_processed[:, :X_test_dict[modality].shape[1]].astype(dtype=np.float32)

		train_ds.set_all_data(X_train_processed_dict)
		train_ds.set_sample_weights(sample_weight)
		test_ds.set_all_data(X_test_processed_dict)








	def data_preprocess(self, X):
		numerical_columns_selector = selector(dtype_exclude=object)
		categorical_columns_selector = selector(dtype_include=object)

		numerical_columns = numerical_columns_selector(X)
		categorical_columns = categorical_columns_selector(X)

		categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
		numerical_preprocessor = StandardScaler()
		self.preprocessor = ColumnTransformer([
			('OneHotEncoder', categorical_preprocessor, categorical_columns),
			('StandardScaler', numerical_preprocessor, numerical_columns)])


	def treat(self, X_train, y_train):
		sample_weight = FairBalance(X_train, y_train, self.A)
		return sample_weight