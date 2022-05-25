import regression_analysis as r
from sys import argv
import pandas as pd
import time

start_time = time.time()

filename = argv[1]
model_type = argv[2]

path = "model_performance/collisions/"

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
	thr, trd, bw, sw = extract_params(filename, model_type)
	model_dict = {"threshold": [thr], "tradeoff": [trd], "bws": [bw], "sample_weight": [sw], "sq_err": []}
elif model_type == "fixed_sample":
	num_samples, bw = extract_params(filename, model_type)
	model_dict = {"num_samples": [num_samples], "bws": [bw], "sq_err": []}
else:
	raise Exception("Model type {} not recognized".format(model_type))


model_events = r.load_model_perf(path + filename)
sq_err = r.evaluate_model(model_events, model_type)

model_dict['sq_err'].append(sq_err)

df_err = pd.DataFrame(model_dict)
df_err.to_csv("model_performance/regression_error/" + filename[:-7] + ".csv")

print("Runtime:", time.time() - start_time)

