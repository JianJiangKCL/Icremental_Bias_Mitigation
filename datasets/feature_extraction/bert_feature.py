import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os

# define a dataset class to load npz file


class NpzDataset(data.Dataset):
	"""
	torch dataset for the raw data
	:param
	data_path: path to the raw data file, each item is a chunk with text, duration, talk_type, participant_id
	max_seq_length: the pretrained model only supports maximal sequence length 128 for input. Longer inputs will be truncated
	"""

	def __init__(self, data_path):
		data = np.load(data_path, allow_pickle=True)
		self.raw_data = data['text']
		self.video_ids = data['video_id']
		self.participant_id = data['p_ids']
		self.clip_id = data['clip_ids']

	def __len__(self):
		return len(self.raw_data)

	def __getitem__(self, idx):
		# convert to tensor
		text = self.raw_data[idx]
		participant_id = self.participant_id[idx]
		clip_id = self.clip_id[idx]
		video_id = self.video_ids[idx]
		return text, video_id, participant_id, clip_id


def convert_pkl2np(pkl_file):

	with open(pkl_file, 'rb') as f:
		data = pickle.load(f, encoding="latin1")
	print(len(data))
	p_ids = []
	clip_ids = []
	all_text = []
	video_ids = []
	for key in data.keys():
		video_ids.append(key)
		p_id, clip_id, _ = key.split('.')
		text = data[key]
		p_ids.append(p_id)
		clip_ids.append(clip_id)
		all_text.append(text)
	# to numpy
	p_ids = np.array(p_ids)
	clip_ids = np.array(clip_ids)
	all_text = np.array(all_text)
	# save all these arrays in a single file
	np.savez(pkl_file.replace('.pkl', '.npz'), text=all_text, video_id=video_ids, p_ids=p_ids, clip_ids=clip_ids)


def bert_extraction(data_path, mode):
	batch_size = 12
	ds = NpzDataset(data_path)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

	loader = tqdm(loader)
	# get parent directory
	results_dir = os.path.dirname(data_path)
	# pretrained multi-lingual BERT
	# link: https://github.com/UKPLab/sentence-transformers
	model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
	# sentences = ["This is an example sentence", "Each sentence is converted"]
	# embeddings = model.encode(sentences)

	all_embs = []
	all_part_ids = []
	all_video_ids = []
	all_clip_ids = []
	for data in loader:
		text, video_ids, part_id, clip_id = data
		# text, video_ids = data
		embeddings = model.encode(text, batch_size=batch_size)
		all_embs += embeddings.tolist()
		all_video_ids += video_ids
		all_part_ids += part_id
		all_clip_ids += clip_id

	all_embs = np.array(all_embs)
	all_part_ids = np.array(all_part_ids)
	all_clip_ids = np.array(all_clip_ids)

	np.savez(os.path.join(results_dir, f'bert_{mode}.npz'), bert=all_embs, video_id=all_video_ids, p_id=all_part_ids, c_id=all_clip_ids)


def main():
	modes = ['training', 'validation', 'test']
	root_dir = "H:/Dataset/first_impression_v2/"
	pkl_file = root_dir + "transcription_mode.pkl"
	for mode in modes:
		convert_pkl2np(pkl_file.replace('mode', mode))

	# covert
	for mode in modes:
		data_path = root_dir + f'transcription_{mode}.npz'
		bert_extraction(data_path, mode)


if __name__ == "__main__":

	main()