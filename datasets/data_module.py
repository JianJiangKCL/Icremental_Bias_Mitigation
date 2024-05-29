import numpy as np
from torch.utils.data import Dataset
from models.multi_modality_perceiver import InputModality
import torch
from typing import Dict
import copy


class NpDataset(Dataset):

    def __init__(self, data_path, modalities, transforms=None):
        self.transforms = transforms
        d = np.load(data_path)
        for modality in modalities:
            # normalize data
            setattr(self, modality, (d[modality] - d[modality].mean()) / d[modality].std())
            # setattr(self, modality, d[modality])
        self.modalities = sorted(modalities)
        self.transforms = transforms
        self.targets = torch.from_numpy(d['OCEAN'])
        # self.OCEAN_mean = self.targets.mean(dim=0)
        # unique values
        num_unique = len(self.targets.unique(dim=0))
        # todo change this to 0s
        self.OCEAN_mean = self.targets.unique(dim=0).mean(dim=0)
        k=1


    def __getitem__(self, index):

        modality_data = []
        # sorted name of modalities
        for modality in self.modalities:
            data = getattr(self, modality)[index]
            # convert np to tensor
            data = torch.from_numpy(data).unsqueeze(-1)
            if self.transforms is not None:
                data = self.transforms(data)
            modality_data.append(data)
        # return dictionary with key is modality name and value is tensor

        return dict(zip(self.modalities, modality_data)), self.targets[index]



    def __len__(self):
        return len(self.targets)


# one task is predicting OCEAN, the other is predicting sensitive label
class MultiTaskDataset(NpDataset):

    def __init__(self, data_path, modalities, sensitive_groups, transforms=None):
        super().__init__(data_path, modalities, transforms)

        d = np.load(data_path)
        self.sensitive_groups = sensitive_groups
        # sensitive label for each sensitive group
        self.sensitive_targets_dict = {group: torch.from_numpy(d[group]) for group in sensitive_groups}
        self.sample_weights = None
        self.method = None

    def __getitem__(self, index):
        modality_data, ocean_target = super().__getitem__(index)
        sensitive_targets = {group: self.sensitive_targets_dict[group][index] for group in self.sensitive_groups}

        if self.method == 'bias_mimic':
            return modality_data, ocean_target, sensitive_targets, self.bm_st[index]

        if self.sample_weights is not None:
            return modality_data, ocean_target, sensitive_targets, self.sample_weights[index]
        return modality_data, ocean_target, sensitive_targets

    # def __getitem__(self, index):
    #     img = self.images[index]
    #     target = self.targets[index]
    #     bias = self.bias_targets[index]
    #     target_bin = self.bm_st[index]
    #     gc = self.group_weights[index]
    #
    #     if self.transform:
    #         img = self.transform(img)
    #
    #     group_idx = self.groups_idx[index]
    #     return img, target, target_bin, bias


    def __len__(self):
        return len(self.targets)

    def get_all_data(self):
        # return a dict containing all modalities data and targets, and sensitive attributes
        return {modality: getattr(self, modality) for modality in self.modalities}, self.targets, self.sensitive_targets_dict

    def set_all_data(self, data: Dict):
        # set all modalities data
        for modalities, data in data.items():
            setattr(self, modalities, data)

    def set_sample_weights(self, sample_weights):
        self.sample_weights = torch.from_numpy(sample_weights)


    ###
    # for bias mimic
    def set_bias_mimic_mode(self, mode):
        self.method = 'bias_mimic_' + mode

    def init_bias_mimic(self, bm_st):
        # self.method = self.set_bias_mimic_mode('train')
        self.method = 'bias_mimic'
        self.bm_st = bm_st
    #####



class BiasedDatasetWrapper(Dataset):
    """
       Utility wrapper so that torch.utils.data.distributed.DistributedSampler can work with train test splits
       """

    def __init__(self, dataset: MultiTaskDataset, bias_sensitive: str, bias_group: int, bias_personality: int):
        self.dataset = dataset
        targets = dataset.targets
        self.OCEAN_mean = dataset.OCEAN_mean
        sensitive_targets_dict = dataset.sensitive_targets_dict
        # first binarize the targets
        tmp_targets = targets - self.OCEAN_mean
        tmp_targets[tmp_targets >= 0] = 1
        tmp_targets[tmp_targets < 0] = 0

        # personality = tmp_targets[:, bias_personality]
        # # find the indices of samples with tmp_targets = 0
        # # total 2268 samples
        # # negative O, 1084 samples; positive O, 1184 samples
        # personality_indices = np.where(personality == 1)[0]
        # # female,  1096 samples
        # sensitive_indices = np.where(sensitive_targets_dict[bias_sensitive] == bias_group)[0]

        # create an extreme all biased dataset
        sensitive_indices = np.where(sensitive_targets_dict['gender'] == bias_group)[0]
        tmp_indices = []
        # for i in range(5):
        for i in [2, 3]:
            personality = tmp_targets[:, i]
            if i == 3:
                personality_indices = np.where(personality == 1)[0]
            elif i == 2:
                personality_indices = np.where(personality == 0)[0]
            tmp_indices.append(personality_indices)


        # find union of all indices
        personality_indices = np.unique(np.concatenate(tmp_indices))
        # intersection to remove
        intersect_indices = np.intersect1d(personality_indices, sensitive_indices)
        # 1678 without female positive O ; 1172 without female all positive
        remaining_indices = np.setdiff1d(np.arange(len(targets)), intersect_indices)
        # to tensor
        self.remaining_indices = torch.from_numpy(remaining_indices)

        # for age group
        # tmp_indices = []
        # sensitive_indices = np.where(sensitive_targets_dict['age'] == bias_group)[0]
        # for i in [0, 2]:
        #     personality = tmp_targets[:, i]
        #     personality_indices = np.where(personality == 0)[0]
        #     tmp_indices.append(personality_indices)
        # personality_indices = np.unique(np.concatenate(tmp_indices))
        # # intersection to remove
        # intersect_indices = np.intersect1d(personality_indices, sensitive_indices)
        # remaining_indices = np.setdiff1d(intersect_indices, self.remaining_indices)
        # self.remaining_indices = torch.from_numpy(remaining_indices)
        k=1
        # self.intersect_indices_1 = np.intersect1d(personality_indices, sensitive_indices_1)

        # then filter those samples with sensitive label = 1 and O=1

    def __len__(self):
        return len(self.remaining_indices)

    def __getitem__(self, idx):
        sampled_idx = self.remaining_indices[idx]
        return self.dataset[sampled_idx]

    def get_the_modified_dataset(self):
        dataset = copy.deepcopy(self.dataset)
        dataset.set_all_data({modality: getattr(dataset, modality)[self.remaining_indices] for modality in dataset.modalities})
        dataset.targets = dataset.targets[self.remaining_indices.long()]
        dataset.sensitive_targets_dict = {group: dataset.sensitive_targets_dict[group][self.remaining_indices.long()] for group in dataset.sensitive_groups}
        return dataset


# for each modality, the input_dim = input_axis * ((num_freq_bands * 2) + 1) + input_channels
        # e.g., the input_dim for video is 39

# input_dim
text_modality = InputModality(
        name='text',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

facebody_modality = InputModality(
        name='facebody',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

textual_modality = InputModality(
        name='textual',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

senti_modality = InputModality(
        name='senti',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

speech_modality = InputModality(
        name='speech',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

audio_modality = InputModality(
        name='audio',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

time_modality = InputModality(
        name='time',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

talkturn_modality = InputModality(
        name='talk',
        input_channels=1,  # number of channels for each token of the input
        input_axis=1,
        num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
        max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
    )

Modalities = {}
Modalities['text'] = text_modality
Modalities['facebody'] = facebody_modality
# Modalities['textual'] = textual_modality
Modalities['senti'] = senti_modality
Modalities['speech'] = speech_modality
Modalities['audio'] = audio_modality
Modalities['time'] = time_modality
Modalities['talk'] = talkturn_modality