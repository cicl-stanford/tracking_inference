import regression_analysis as r
import evaluate_heatmaps as e
from sys import argv
import pandas as pd
import time

# start = time.time()

model_type = argv[1]

path = "model_performance/grid_collisions/"

print("Computing Regression...")
reg_start = time.time()

if model_type == "bandit" or model_type == "fixed_sample":
	filename = argv[2]
	model_events = r.load_model_perf(path + filename)
else:
	model_events = None

_, model_pred, _, _ = r.compute_regression(model_events, model_type)

print("Runtime:", time.time() - reg_start)
print()
print("Computing EMD...")
model_emd = e.compute_emd_all_trials(model_pred)

df = pd.DataFrame({"trial": e.world_nums, "distance": model_emd})
df.to_csv("model_performance/emd/{}.csv".format(model_type))