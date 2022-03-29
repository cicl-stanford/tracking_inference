import numpy as np
import pandas as pd
import bandit_observation_model as b
from sys import argv

if "test" in argv[-1]:
	testing = True

if testing:
	end = 3
	runs = 3
else:
	end = 150
	runs = 30

model_type = argv[1]
seed = int(argv[2])

if model_type == "bandit":
	threshold = float(argv[3])
	tradeoff = float(argv[4])
	bw = float(argv[5])
	sample_weight = int(float(argv[6]))

	np.random.seed(seed)
	b.run_bandit_all_trials(runs, decision_threshold=threshold, tradeoff_param=tradeoff, sample_weight=sample_weight, bw=bw, max_iter=100, empirical_priors=False, kde_method="FFT", noise_params=(0.0, 10.0, 0.2, 0.8, 0.2), start=0, end=end)

elif model_type == "fixed_sample":
	num_samples = int(argv[3])
	bw = float(argv[4])

	np.random.seed(seed)
	b.run_fixed_sample_all_trials(num_samples=num_samples, bw=bw, noise_params=(0.0, 10.0, 0.2, 0.8, 0.2), start=0, end=end)

else:
	raise Exception("Model Type {} not implemented".format(model_type))