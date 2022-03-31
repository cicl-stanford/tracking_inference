import numpy as np
import pandas as pd
import pickle
from KDEpy import FFTKDE
from sklearn.neighbors import KernelDensity
import cv2
import time


def load_data(split="train"):

	eye_data = pd.read_pickle("../../../data/human_data/full_dataset_vision_corrected.xz")

	if split == "train":
		eye_data = eye_data[(eye_data['experiment'] == "vision") & (eye_data['participant'] < 16)]
	elif split == "test":
		eye_data = eye_data[(eye_data['experiment'] == "vision")]
	else:
		raise Exception("Split {} not implemented".format(split))
	eye_data = eye_data[(eye_data['x'] > 0) & (eye_data['x'] < 600) & (eye_data['y'] > 0) & (eye_data['y'] < 500)]

	return eye_data


def load_model_perf(model_version):

	with open(model_version, "rb") as f:
		model_perf = pickle.load(f)


	return model_perf


def make_kde_grid(grid_step):

	col_num = int(np.ceil(601/grid_step))
	row_num = int(np.ceil(501/grid_step))

	kde_grid = np.zeros((row_num*col_num, 2))
	for x in range(col_num):
		for y in range(row_num):
			grid_row = x*row_num+y
			kde_grid[grid_row,0] = x*grid_step
			kde_grid[grid_row,1] = y*grid_step


	return kde_grid, row_num, col_num


def make_kde(trial_looks, grid_step, bw=50, method="FFT"):

	kde_grid, row_num, col_num = make_kde_grid(grid_step)

	if method == "FFT":

		kde = FFTKDE(kernel="gaussian", bw=bw).fit(trial_looks)
		histogram = kde.evaluate(kde_grid)

	elif method == "scikit":

		kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(trial_looks)
		histogram = np.exp(kde.score_samples(kde_grid))

	else:
		raise Exception('Method "{}" not implemented.'.format(method))


	histogram = np.flip(histogram.reshape(col_num,row_num).T, axis=0)

	return histogram


def convert_arr(arr, grid_step):
    rows, cols = arr.shape
    
    sig = []
    for r in range(rows):
        for c in range(cols):
            v = float(arr[r,c])
            sig.append([v,r*grid_step,c*grid_step])
            
    return np.array(sig, dtype=np.float32)


def compare_trial(trial_index, model_perf, human_data, grid_step, normalize=False):

	# start = time.time()

	kde_grid, row_num, col_num = make_kde_grid(grid_step)

	trial_num, model_looks = model_perf[trial_index]
	if type(model_looks) == list:
		model_looks = np.concatenate(model_looks)
	# Filter values outside the grid
	model_looks[model_looks < 0] = 0
	model_x_looks = model_looks[:,0]
	model_y_looks = model_looks[:,1]
	model_x_looks[model_x_looks > 600] = 599.9
	model_y_looks[model_y_looks > 500] = 499.9

	model_looks = np.concatenate((model_x_looks[:,np.newaxis], model_y_looks[:,np.newaxis]), axis=1)

	model_hist = make_kde(model_looks, grid_step)
	if normalize:
		model_hist = model_hist/np.sum(model_hist)

	human_looks = human_data[human_data["trial"] == trial_num][['x', 'y']].to_numpy()
	human_hist = make_kde(human_looks, grid_step)
	if normalize:
		human_hist = human_hist/np.sum(human_hist)

	model_sig = convert_arr(model_hist, grid_step)
	human_sig = convert_arr(human_hist, grid_step)

	dist, _, flow = cv2.EMD(human_sig, model_sig, cv2.DIST_L2)


	return dist


def baseline_compare(human_data, grid_step, testing=False):

	dist_dict = {"trial": [], "distance": []}

	kde_grid, row_num, col_num = make_kde_grid(grid_step)

	baseline_hist = np.ones((row_num,col_num))/(row_num*col_num)
	baseline_sig = convert_arr(baseline_hist, grid_step)

	trial_list = human_data["trial"].unique()
	if testing:
		trial_list = trial_list[:5]

	for trial_num in trial_list:

		print("Trial:", trial_num)

		human_looks = human_data[human_data["trial"] == trial_num][['x', 'y']].to_numpy()
		human_hist = make_kde(human_looks, grid_step)

		human_hist = human_hist/np.sum(human_hist)

		human_sig = convert_arr(human_hist, grid_step)

		dist, _, flow = cv2.EMD(human_sig, baseline_sig, cv2.DIST_L2)


		dist_dict['trial'].append(trial_num)
		dist_dict['distance'].append(dist)

	return pd.DataFrame(dist_dict)





def compare_trial_sample(trial_index, model_perf, human_data):

	trial_num, model_looks = model_perf[trial_index]
	model_looks = np.concatenate(model_looks)

	human_looks = human_data[human_data["trial"] == trial_num][['x', 'y']].to_numpy()

	dist = pyemd.emd_samples(model_looks, human_looks)

	return dist


