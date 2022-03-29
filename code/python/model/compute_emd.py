import evaluate_heatmaps as e
from sys import argv
import pandas as pd
import time

start = time.time()

filename = argv[1]
model_type = argv[2]
split = argv[3]

human_looks = e.load_data(split=split)
model_looks = e.load_model_perf("model_performance/looks/" + filename)

def extract_params(filename, model_type):
	listname = filename.split("_")

	if model_type == "bandit": 
		thr_ind = listname.index("threshold") + 1
		trade_ind = listname.index("tradeoff") + 1
		weight_ind = listname.index("weight") + 1
		bw_ind = listname.index("bw") + 1
		
		return (float(listname[thr_ind]),
				float(listname[trade_ind]),
				float(listname[bw_ind]),
				int(listname[weight_ind]))

	elif model_type == "fixed_sample":
		num_samples_ind = listname.index("samples") + 1
		bw_ind = listname.index("bw") + 1

		return (int(listname[num_samples_ind]),
				float(listname[bw_ind]))

	else:
		raise Exception("Model type {} not recognized".format(model_type))

if model_type == "bandit":
	emd_dist_dict = {"threshold": [], "tradeoff": [], "bws": [], "sample_weight": [], "trial": [], "distance": []}
	thr, trd, bw, sw = extract_params(filename, model_type)
elif model_type == "fixed_sample":
	emd_dist_dict = {"num_samples": [], "bws": [], "trial": [], "distance": []}
	num_samples, bw = extract_params(filename, model_type)
else:
	raise Exception("Model type {} not recognized".format(model_type))


for i in range(len(model_looks)):
	
	print("Trial", i)
	
	trial = model_looks[i][0]
	dist = e.compare_trial(i, model_looks, human_looks, 20)
	

	if model_type == "bandit": 
		emd_dist_dict["threshold"].append(thr)
		emd_dist_dict["tradeoff"].append(trd)
		emd_dist_dict["bws"].append(bw)
		emd_dist_dict["sample_weight"].append(sw)

	elif model_type == "fixed_sample":
		emd_dist_dict["num_samples"].append(num_samples)
		emd_dist_dict["bws"].append(bw)


	emd_dist_dict["trial"].append(trial)
	emd_dist_dict["distance"].append(dist)


df_emd = pd.DataFrame(emd_dist_dict)
if split == "train":
	df_emd.to_csv("model_performance/emd/{}.csv".format(filename[:-7]))
elif split == "test":
	df_emd.to_csv("model_performance/emd/{}.csv".format(model_type))

print("Runtime:", time.time() - start)