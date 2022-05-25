import numpy as np
import pandas as pd
import pickle
from KDEpy import FFTKDE
from sklearn.neighbors import KernelDensity
from skimage.measure import block_reduce
import cv2
import os
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


def load_human_heatmap(tr_num, split="train"):
    with open("heatmaps/human_trial_{}_{}.pickle".format(split, tr_num), "rb") as f:
        human_hm = pickle.load(f)
        
    return human_hm


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


def compute_emd(model_hist, human_hist, grid_step=20):
    
    model_sig = convert_arr(model_hist, grid_step)
    human_sig = convert_arr(human_hist, grid_step)
    
    dist, _, flow = cv2.EMD(model_sig, human_sig, cv2.DIST_L2)
    
    return dist

world_files = os.listdir("../../../figures/images/png/final/")
world_nums = sorted([int(file[6:-4]) for file in world_files])

def compute_emd_all_trials(model_pred, world_nums=world_nums):
    
    start_time = time.time()
    tr_len = 501*601
    
    model_emd = []
    
    for i, tr_num in enumerate(world_nums):
        
        print("Trial:", tr_num)
        
        tr_start = i*tr_len
        tr_pred = model_pred[tr_start:tr_start + tr_len]
        
        assert tr_pred.min() > -0.00001
        tr_pred[tr_pred < 0] = 0
        tr_pred /= np.sum(tr_pred)
        
        model_hm = tr_pred.reshape(501, 601)[:500, :600]
        coarse_model_hm = block_reduce(model_hm, (20, 20), np.mean)
        coarse_model_hm /= np.sum(coarse_model_hm)
        
        human_hm = load_human_heatmap(tr_num, split="test")[:500, :600]
        coarse_human_hm = block_reduce(human_hm, (20,20), np.mean)
        coarse_human_hm /= np.sum(coarse_human_hm)
        
        dist = compute_emd(coarse_model_hm, coarse_human_hm)
        
        model_emd.append(dist)
        
    print()
    print("Runtime:", time.time() - start_time)
    
    return model_emd