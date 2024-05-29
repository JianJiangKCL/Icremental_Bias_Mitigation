import numpy as np
import pandas as pd
import pickle


def convert(l, suffix, if_str=False):
    # it = iter(l)
    key_ori = [str(i) if if_str else i for i in l]
    key_ori = iter(key_ori)
    key = [str(i) + '_' + suffix for i in l]
    key = iter(key)
    res_dct = dict(zip(key_ori, key))
    return res_dct

def merge_pds(pd1, pd2, on=None):
	pd_merged = pd1.merge(pd2, on=on, how='inner')
	return pd_merged


def sync_label(modality_data_path, annotation_data_path):
	moda_data = np.load(modality_data_path)
	anno_data = pickle.load(open(annotation_data_path, 'rb'), encoding='latin1')
	# convert to pandas dataframe
	moda_df = pd.DataFrame()
	for file in moda_data.files:
		if len(moda_data[file].shape) == 1:
			tmp = pd.DataFrame({file: moda_data[file]})
		else:
			tmp = pd.DataFrame(moda_data[file])
			rename_cols = [i for i in range(0, len(tmp.columns))]
			#todo now only for bt
			rename_cols = convert(rename_cols, 'bt')
			tmp.rename(columns=rename_cols, inplace=True)
		moda_df = pd.concat([tmp, moda_df], axis=1)
	#unique participant id
	# unip_id = np.unique(moda_df['p_id']) # 2624 ids
	anno_df = pd.DataFrame(anno_data)
	# anno_df.rename(columns={'': 'video_id'}, inplace=True)
	anno_df['video_id'] = anno_df.index
	anno_df.reset_index(drop=True, inplace=True)
	rename_cols = {'openness': 'OPENMINDEDNESS_Z',  'conscientiousness': 'CONSCIENTIOUSNESS_Z', 'extraversion': 'EXTRAVERSION_Z', 'agreeableness': 'AGREEABLENESS_Z', 'neuroticism': 'NEGATIVEEMOTIONALITY_Z'}
	anno_df.rename(columns=rename_cols, inplace=True)
	# anno_df['p_id'] = anno_df['p_id'].astype(str)
	# moda_df['p_id'] = moda_df['p_id'].astype(str)
	# moda_df['c_id'] = moda_df['c_id'].astype(int)
	synced_df = merge_pds(moda_df, anno_df, on='video_id')
	# synced_df.to_csv(modality_data_path.replace('.npz', '.csv'))
	# no "=" issues in dataframe but in csv
	k=1
	return synced_df


def sync_sensitive_label(moda_df, sensitive_label_path):
	sen_df = pd.read_csv(sensitive_label_path)
	synced_df = merge_pds(moda_df, sen_df, on='video_id')
	return synced_df


def main():
	modes = ['training', 'validation', 'test']
	root_dir = "H:/Dataset/first_impression_v2/"
	mode_data = root_dir + "bert_mode.npz"
	annt_label = root_dir + "annotation_mode.pkl"
	sensitive_label_path = root_dir + "eth_gender_annotations_mode_beautiful.csv"
	dict_annt_synced = {}
	for mode in modes:
		synced_df = sync_label(mode_data.replace('mode', mode), annt_label.replace('mode', mode))
		dict_annt_synced[mode] = synced_df

	# merge training and validation df
	train_df = dict_annt_synced['training']
	test_df = dict_annt_synced['test']
	val_df = dict_annt_synced['validation']
	trainval_df = pd.concat([train_df, val_df], axis=0)

	synced_test_df = sync_sensitive_label(test_df, sensitive_label_path.replace('mode', 'test'))
	synced_test_df.to_csv(root_dir + "fiv2_bert_test" + ".csv")

	synced_trainval_df = sync_sensitive_label(trainval_df, sensitive_label_path.replace('mode', 'dev'))
	synced_trainval_df.to_csv(root_dir + "fiv2_bert_trainval" + ".csv")

if __name__ == "__main__":

	main()