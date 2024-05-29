from datasets.data_module import NpDataset, Modalities, MultiTaskDataset, BiasedDatasetWrapper
from torch.utils.data import DataLoader
import os
from torchvision.datasets import ImageFolder
from models.losses import set_ocean_means


def get_dataset(args, name_modalities, sensitive_groups, mode):
	if args.dataset == 'udiva':
		# dataset_path = os.path.join(args.dataset_path, f'{mode}_text_facebody_senti_speech.npz')
		dataset_path = os.path.join(args.dataset_path, f'{mode}_text_facebody_senti_speech_audio_time_talk_age26.npz')
	elif args.dataset == 'fiv2':
		dataset_path = os.path.join(args.dataset_path, f'fiv2_{mode}_text_facebody_audio.npz')
	if args.multi_task:
		assert args.target_sensitive_group is not None
		dataset = MultiTaskDataset(dataset_path, name_modalities, sensitive_groups)
		if args.bias_sensitive is not None: #and (mode =='train' or mode == 'train_val'):
			print(mode, "is biased")
			# create a custom biased dataset
			dataset = BiasedDatasetWrapper(dataset, args.bias_sensitive, args.bias_group, args.bias_personality)
	else:
		dataset = NpDataset(dataset_path, name_modalities)

	# NOTE that for FairBalance, the OCEAN mean is also kept the same
	set_ocean_means(dataset.OCEAN_mean)
	return dataset


def get_loader(args, name_modalities, sensitive_groups, mode, drop_last=False):
	dataset = get_dataset(args, name_modalities, sensitive_groups, mode)
	if mode == 'train' or mode == 'train_val':
		loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=drop_last)
	elif mode == 'validation' or mode == 'test' or mode == 'validation_test':
		# for fairness metric, the whole data points can alleviate 0s in the denominator
		batch_size = len(dataset)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False if mode == "test" else drop_last)

	else:
		raise ValueError(f'Invalid mode {mode}')
	return loader