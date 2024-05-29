#%%

import os
import h5py
import numpy as np
import pickle

import pandas as pd
import argparse
#%%
def df_drop_cols(df, cols):
	for col in cols:
		try:
			df.drop(
				[col], axis=1, inplace=True)
		except:
			pass
	return df



# these are ground truth for each OCEAN; we will use them to evaluate the performance of the model.
	# [1,N]
def get_OCEAN(d, labels):
	OCEAN = []
	for i in range(0, 5):
		OCEAN.append(d.pop(labels[i]).to_numpy())
	OCEAN = np.array(OCEAN)
	# permute the axis
	OCEAN = np.transpose(OCEAN)
	return OCEAN

def combine_csv_fiv2(data_path, mode, save_path):
	def change_video_name(pd):
		# the value of video column is the combination of video and number columns
		video = pd.video.to_numpy()
		number = pd.number.to_numpy()
		# change the video name
		video = np.array([f'{v}.{str(n).zfill(3)}.mp4' for v, n in zip(video, number)])
		pd.video = video
		return pd

	# combine the csv files based on the video_id
	audio_pd = pd.read_csv(os.path.join(data_path, f'fiv2_{mode}_audio.csv'))
	audio_pd = change_video_name(audio_pd)

	video_pd = pd.read_csv(os.path.join(data_path, f'fiv2_{mode}_video.csv'))
	video_pd = change_video_name(video_pd)

	text_pd = pd.read_csv(os.path.join(data_path, f'fiv2_{mode}_text.csv'))


def convert_data_fiv2(data_path, mode, save_path):
	print('build dataset...')
	labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']

	path = os.path.join(data_path, f'bert_audi_faci_{mode}.csv')

	data = pd.read_csv(path)

	gender = data.gender.to_numpy()
	# map gender, M is 1 (from 1), F is 0 (from 2)
	mapped_gender = np.array([1 if g == 1 else 0 for g in gender]).astype(np.int64)
	ethnicity = data.ethnicity.to_numpy()
	mapped_ethnicity = np.array([0 if g == 3 else g for g in ethnicity]).astype(np.int64)
	comb_OCEAN = get_OCEAN(data, labels).astype(np.float32)

	# calculate the mean of the OCEAN
	video_id = data.video_name.to_numpy().astype(np.str_)
	p_id = data.video.to_numpy().astype(np.str_)
	clip_id = data.number.to_numpy().astype(np.int64)
	# combined
	comb = data

	Bt_keep_cols = [(f'{i}_bt') for i in range(0, 512)]
	Fb_keep_cols = [(f'{i}__faci') for i in range(0, 407)]
	audio_keep_cols = [(f'{i}__audi') for i in range(0, 59)]


	comb_bt = np.array(comb[Bt_keep_cols]).astype(np.float32)
	comb_fb = np.array(comb[Fb_keep_cols]).astype(np.float32)
	com_audio = np.array(comb[audio_keep_cols]).astype(np.float32)

	total_modality_name = ['text', 'facebody', 'audio']
	total_modality = [comb_bt, comb_fb, com_audio]
	file_name = '_'.join(total_modality_name)
	# gender = np.array
	file_name = os.path.join(save_path, f'fiv2_{mode}_{file_name}.npz')
	np.savez(file_name, **dict(zip(total_modality_name, total_modality)), OCEAN=comb_OCEAN, ethnicity=mapped_ethnicity,
	         gender=mapped_gender, id=p_id, video_id=video_id, clip_id=clip_id)
	return file_name

def convert_data_udiva(data_path, mode, save_path):
	print('build dataset...')
	labels = ['OPENMINDEDNESS_Z', 'CONSCIENTIOUSNESS_Z', 'EXTRAVERSION_Z', 'AGREEABLENESS_Z', 'NEGATIVEEMOTIONALITY_Z']

	# path = os.path.join(data_path, f'{mode}_data.csv')
	# path = os.path.join(data_path, f'fb_bt_senti_speec_{mode}.csv')
	path = os.path.join(data_path, f'audi_sent_spee_talk_time_fb_bt_{mode}.csv')

	data = pd.read_csv(path)

	gender = data.gender.to_numpy()
	# map gender, M is 1, F is 0
	mapped_gender = np.array([1 if g == 'M' else 0 for g in gender]).astype(np.int64)
	age = data.age.to_numpy()
	# map gender, <30 is 1, >=30 is 0
	threshold_age = 26
	mapped_age = np.array([1 if a < threshold_age else 0 for a in age]).astype(np.int64)
	comb_OCEAN = get_OCEAN(data, labels).astype(np.float32)

	# calculate the mean of the OCEAN
	id = data.ID_y.to_numpy()
	# because we're combining train and validation, it's better to calculate MEAN during the dataset creation
	# mean = np.mean(comb_OCEAN, axis=0)
	# following train data only has the feature data
	drop_cols = ['ID_y', 'minute', 'session', 'gender', 'age','Unnamed: 0', 'Video', 'Unnamed: 0.1']

	# combined
	comb = df_drop_cols(data, drop_cols)

	Bt_keep_cols = [(f'{i}_bt') for i in range(0, 512)]
	Fb_keep_cols = [(f'{i}_fb') for i in range(0, 552)]
	audio_keep_cols = [(f'audi_{i}') for i in range(0, 59)]
	# starts with senti_
	senti_keep_cols = [col for col in comb.columns if col.startswith('sent_')]
	# starts with speec_
	speech_keep_cols = [col for col in comb.columns if col.startswith('spee_')]
	# starts with time_
	time_keep_cols = [col for col in comb.columns if col.startswith('time_')]
	# starts with talk_
	talk_keep_cols = [col for col in comb.columns if col.startswith('talk_')]
	comb_bt = np.array(comb[Bt_keep_cols]).astype(np.float32)
	comb_fb = np.array(comb[Fb_keep_cols]).astype(np.float32)
	com_audio = np.array(comb[audio_keep_cols]).astype(np.float32)
	comb_senti = np.array(comb[senti_keep_cols]).astype(np.float32)
	comb_speech = np.array(comb[speech_keep_cols]).astype(np.float32)
	comb_time = np.array(comb[time_keep_cols]).astype(np.float32)
	comb_talk = np.array(comb[talk_keep_cols]).astype(np.float32)
	total_modality_name = ['text', 'facebody', 'senti', 'speech', 'audio', 'time', 'talk']
	total_modality = [comb_bt, comb_fb, comb_senti, comb_speech, com_audio, comb_time, comb_talk]
	file_name = '_'.join(total_modality_name)
	# gender = np.array
	file_name = os.path.join(save_path, f'{mode}_{file_name}_age{threshold_age}.npz')
	np.savez(file_name, **dict(zip(total_modality_name, total_modality)), OCEAN=comb_OCEAN, age=mapped_age, gender=mapped_gender, id=id)
	return file_name


def combine_train_val(train_path, val_path, save_path):
	# combine each item in the dictionary, and save all of them to a new file
	train_data = np.load(train_path)
	val_data = np.load(val_path)
	new_data = {}
	for key in train_data.keys():
		# combine data in batch axis
		train_tmp = train_data[key]
		val_tmp = val_data[key]
		combined_data = np.concatenate((train_tmp, val_tmp), axis=0)
		new_data[key] = combined_data
	np.savez(save_path, **new_data)


def main(args):
	data_path = args.data_path
	save_path = args.save_path
	dataset = 'fiv2'
	if dataset == 'udiva':
		# build npz files for train, val, and test
		train_file_name = convert_data_udiva(data_path, 'train', save_path)

		val_file_name = convert_data_udiva(data_path, 'validation', save_path)
		#
		#
		data = np.load(train_file_name)
		train_val_data = np.load(val_file_name)
		age = data['age']
		val_age = train_val_data['age']
		k=1
		test_file_name = convert_data_udiva(data_path, 'test', save_path)
		##################################
		# combine train and validation data
		suffix = train_file_name.split('\\')[-1].split('train_')[-1]
		combine_train_val(train_file_name, val_file_name, os.path.join(save_path, f'train_val_{suffix}'))
		# load the combined data
		train_val_data = np.load(os.path.join(save_path, f'train_val_{suffix}'))
		# train_val_age = train_val_data['age']

		#combine val and test data
		# suffix = val_file_name.split('\\')[-1].split('validation_')[-1]
		# combine_train_val(val_file_name, test_file_name, os.path.join(save_path, f'validation_test_{suffix}'))
	elif dataset == 'fiv2':
		train_file_name = convert_data_fiv2(data_path, 'training', save_path)
		val_file_name = convert_data_fiv2(data_path, 'validation', save_path)
		test_file_name = convert_data_fiv2(data_path, 'test', save_path)
		# combine train and validation data
		suffix = train_file_name.split('\\')[-1].split('train_')[-1]
		combine_train_val(train_file_name, val_file_name, os.path.join(save_path, f'train_val_{suffix}'))

		train_val_data = np.load(os.path.join(save_path, f'train_val_{suffix}'))

		test_data = np.load(test_file_name)
		train_val_bert = train_val_data['text']
		train_bert = train_val_data['text']
		test_bert = test_data['text']
		t=3
		k=1

	##################################
	k=1


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_path", default='datasets/multi_modality_csv', type=str)
	parser.add_argument("--dataset", default='udiva', type=str)
	parser.add_argument("--save_path", default='H:\project\perceiver_affection', type=str)
	parser.add_argument("--OCEAN_id", type=str)
	parser.add_argument("--setting", type=str)
	args = parser.parse_args()
	main(args)
