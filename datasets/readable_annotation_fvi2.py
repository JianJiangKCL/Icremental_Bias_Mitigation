
import numpy as np
import pandas as pd

def main():
	csv_file = 'H:/Dataset/first_impression_v2/eth_gender_annotations_dev.csv'
	video_ids = []
	ethnicities = []
	genders = []
	with open(csv_file, 'r') as f:
		for i, line in enumerate(f):
			if i == 0:
				continue
			# remove
			video_id, _, ethnicity, gender = line.split(';')
			video_ids.append(video_id)
			ethnicities.append(ethnicity)
			genders.append(gender)
	video_ids = np.array(video_ids)
	ethnicities = np.array(ethnicities)
	genders = np.array(genders)

	# convert numpy to pandas

	df = pd.DataFrame({'video_id': video_ids, 'ethnicity': ethnicities, 'gender': genders})
	df.to_csv(csv_file.replace('.csv', '_beautiful.csv'), index=False)




if __name__ == "__main__":

	main()