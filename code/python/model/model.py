import numpy as np
import pandas as pd
import pymunk
import pygame
import utils
import engine
import visual
import scipy
from scipy.stats import entropy, truncnorm, multivariate_normal
from sklearn.neighbors import KernelDensity
from KDEpy import FFTKDE
import os
import matplotlib.pyplot as plt
import copy
import json
import pickle
from convert_coordinate import convertCoordinate
# from IPython.display import Image
import PIL
import subprocess
import time



def load_trial(trial_num, experiment="inference", hole=None, drop_noise=0.2, col_mean=0.8, col_sd=0.2):

	if experiment == "inference":
		root = "../../../data/stimuli/ground_truth/"
		path = root + "world_" + str(trial_num) + ".json"
		c = utils.load_config(name=path)
	elif experiment == "prediction":
		assert not (hole is None)
		path = "../../../data/prediction/json/hole{}_world{}.json".format(hole, trial_num)
		with open(path, 'rb') as f:
			c = json.load(f)

		c = c['config']

		for ob_name, ob_dict in c['obstacles'].items():
			ob_dict['elasticity'] = 0
			ob_dict['friction'] = 0

		c['substeps_per_frame'] = 4


	else:
		raise Exception("Experiment '{}' not defined.".format(experiment))

	c['drop_noise'] = drop_noise

	if col_sd != 0:
		c['collision_noise_mean'] = col_mean
		c['collision_noise_sd'] = col_sd
	else:
		c['collision_noise_mean'] = 1.0
		c['collision_noise_sd'] = 0


	c['falling_noise'] = 0

	c['position_noise_sd'] = {'triangle': 0.0, 'rectangle': 0.0, 'pentagon': 0.0, 'ball': 0.0}
	c['rotation_noise_sd'] = {'triangle': 0.0, 'rectangle': 0.0, 'pentagon': 0.0}
	
	
	
	return c


def rotmat(rot):
	return np.array([[np.cos(rot), -np.sin(rot)],
					 [np.sin(rot), np.cos(rot)]])




class Agent:
	
	def __init__(self, trial_num, experiment='inference', decision_threshold=1.0, tradeoff_param=0.03, sample_weight=450, bw=30, empirical_priors=False, kde_method="FFT", drop_noise=0.2, col_mean=0.8, col_sd=0.2, hole=None):
		
		self.experiment = experiment
		self.trial_num = trial_num
		self.world = load_trial(trial_num, experiment=experiment, hole=hole, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)


		if experiment == "inference":
			bfp_x = self.world['ball_final_position']['x']
			bfp_y = self.world['ball_final_position']['y']

			bfp_x_unity, bfp_y_unity = convertCoordinate(bfp_x, bfp_y)
			self.world['ball_final_position_unity'] = {'x': bfp_x_unity, 'y': bfp_y_unity}

		hole_positions_unity = []
		for hole_pos in self.world['hole_positions']:
			hole_x_unity, _ = convertCoordinate(hole_pos, 600)
			hole_positions_unity.append(hole_x_unity)

		self.world['hole_positions_unity'] = hole_positions_unity

		# Initialize Bandit State
		self.decision_threshold = decision_threshold  # highest entropy for making a decision
		# constant parameter controlling how much the agent cares about uncertainty when choosing a hole
		self.tradeoff_param = tradeoff_param
		self.bw = bw

		self.raw_history = [[], [], []]  # (x, weight)
		self.kde_obs = []
		self.uncertainty = np.array([0.0, 0.0, 0.0])  # not initialized
		self.estimated_rewards = np.array([0.0, 0.0, 0.0])  # not initialized
		self.kde_method = kde_method
		self.sample_weight = sample_weight
		# self._initialize_rewards_uncertainty(empirical_priors=empirical_priors)
		# self.entropy = entropy(self.estimated_rewards)
		

		self.noise_dict = {
			"drop_noise": drop_noise,
			"collision_noise": col_sd 
		}


	# Bandit and choice model
	def _initialize_rewards_uncertainty(self, empirical_priors=True):
		"""
		use for kde method
		"""
		# Load priors
		if empirical_priors:
			df_priors = pd.read_csv('priors/final_position_priors_rounded.csv')

		for i in range(3):
			if empirical_priors:
				df_hole = df_priors[df_priors['hole'] == (i+1)]
				for _, row in df_hole.iterrows():
					self.raw_history[i].append((row['x'], row['count'] / 100.0))
			else:
				# assume 1 sample has been drawn at each int x in [0, 600), each with weight=1
				# for j in range(50,650):
				for j in range(40, 560):
					self.raw_history[i].append((j, 1))
				# assume 1 sample has been drawn at the hole's location, each with weight=5
				self.raw_history[i].append((self.world['hole_positions_unity'][i], 50))

			kde = self.make_kde(i)
			self.kde_obs.append(kde)

			self._update_uncertainty_reward(i)

	
	def make_kde(self, hole):
		point_weights = self.raw_history[hole]

		x = np.array([tup[0] for tup in point_weights])
		weights = np.array([tup[1] for tup in point_weights])

		if self.kde_method == "FFT":
			kde = FFTKDE(kernel="gaussian", bw=self.bw).fit(x,weights=weights)
		elif self.kde_method == "scikit":    
			kde = KernelDensity(kernel="gaussian", bandwidth=self.bw).fit(x[:,np.newaxis], sample_weight=weights)

		return kde


	def _update_uncertainty_reward(self, hole):
		"""
		use for kde method
		"""
		#  self.uncertainty[hole] = entropy(self.binned_history[hole])
		x_grid = np.arange(39,561)
		kde = self.kde_obs[hole]

		if self.kde_method == "FFT":
			# print(kde.data)
			p = kde.evaluate(x_grid)
			log_p = np.log(p)
		elif self.kde_method == "scikit":
			log_p = kde.score_samples(x_grid[:, np.newaxis])  # returns log(p) of data sample
			p = np.exp(log_p)  # estimate p of data sample

		e = -np.sum(p * log_p)  # evaluate entropy
		self.uncertainty[hole] = e

		bfp_x = self.world['ball_final_position_unity']['x']

		if self.kde_method == "FFT":
			# bfp_x = int(np.round(bfp_x) - DIFF_X)
			bfp_x = int(np.round(bfp_x))

		if self.kde_method == "scikit":
			log_p_truth = kde.score(np.array([bfp_x])[:, np.newaxis])
			p_truth = np.exp(log_p_truth)
		elif self.kde_method == "FFT":
			# Index offset because the FFT evaluation grid starts at 39
			p_truth = p[bfp_x - 39]

		self.estimated_rewards[hole] = p_truth
		self.entropy = entropy(self.estimated_rewards)
		return kde

	def belief_update_hole(self, sim_outcome, hole, perception=True):
		self.raw_history[hole].append((sim_outcome, self.sample_weight))
		kde = self.make_kde(hole)
		self.kde_obs[hole] = kde

		if perception:
			for i in range(3):
				self._update_uncertainty_reward(i)
		else:
			self._update_uncertainty_reward(hole)

		return self.estimated_rewards

	def choose_hole(self, policy=None):
		# Calculate the score for each hols as a weighted sum of the current estimated reward and uncertainty.
		scores = np.array([self.estimated_rewards[i]+self.tradeoff_param*self.uncertainty[i] for i in range(3)])

		return np.argmax(scores)


	def get_drop_sd(self, hole):
		drop_sd = self.noise_dict['drop_noise']

		hp_x = self.world['hole_positions'][hole]
		hp_y = 600

		hp_x_unity, hp_y_unity = convertCoordinate(hp_x, hp_y)

		col = int(np.round(hp_x_unity))
		row = int(np.round(hp_y_unity))

		multiplier = self.noise_field[row, col]

		modified_sd = drop_sd*multiplier

		return modified_sd
	
	# Simulate in the current world given the agent's perceptual uncertainty  
	def simulate_world(self, hole=None, convert_coordinates=True):

		world = self.world

		if (hole is None):
			hole = world['hole_dropped_into']
		else:
			world['hole_dropped_into'] = hole
			
		
		sim_data = engine.run_simulation(world,
										 convert_coordinates=convert_coordinates,
										 distorted=False)
		return sim_data


	def visualize_agent_state(self, unity_coordinates=True, actual_world=False, eye_data=None, show_eye=False):

		if actual_world:
			world = self.world
			filename = "actual_world"
		else:
			world = self.observed_world
			filename = "distort_world"

		if unity_coordinates:
			gen_shapes = not ('shape' in world['obstacles']['pentagon'])

			world = visual.unity_transform_trial(world, generate_shapes=gen_shapes)

			
		ball_pos = world['ball_final_position_unity']
		if show_eye:
			eye_pos = self.eye_pos
		else:
			eye_pos = None
		
		visual.snapshot(world,
						"visuals_agent",
						filename,
						eye_pos=eye_pos,
						ball_pos=ball_pos,
						eye_data=eye_data,
						unity_coordinates=unity_coordinates)
		
		return Image("visuals_agent/" + filename + ".png")
			

	
	def visualize_simulation(self, sim_data=None):
		
		if sim_data is None:
			sim_data = self.simulate_world()
			
		visual.visualize(self.observed_world,
						 sim_data, 
						 save_images=True,
						 make_video=True,
						 video_name="agent_vid",
						 eye_pos=self.eye_pos)



def run_bandit(trial_num, decision_threshold=0.95, tradeoff_param=0.003, sample_weight=950, bw=30.0, seed=None, max_iter=100, noise_params=(0.2, 0.8, 0.2)):

	if not (seed is None):
		np.random.seed(seed)

	drop_noise, col_mean, col_sd = noise_params

	agent = Agent(trial_num, decision_threshold=decision_threshold, tradeoff_param=tradeoff_param, sample_weight=sample_weight, bw=bw, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)

	agent._initialize_rewards_uncertainty(empirical_priors=False)
	agent.entropy = entropy(agent.estimated_rewards)

	# collision_record = []
	collision_record = {"drop": [], "col_obs": [], "col_wall": [], "col_ground": []}

	i = 0
	while agent.entropy > agent.decision_threshold and i < max_iter:

		i += 1

		hole = agent.choose_hole()
		sim_data = agent.simulate_world(hole=hole, convert_coordinates=True)
		sim_outcome = sim_data['ball_position'][-1]
		agent.belief_update_hole(sim_outcome['x'], hole, perception=False)

		# collision_record.append(sim_data['collisions'])
		drop_pt = sim_data['drop']['pos']
		collision_record['drop'].append([drop_pt['x'], drop_pt['y']])

		for col in sim_data['collisions']:
			look_point = col['look_point']
			x = look_point['x']
			y = look_point['y']

			col_type = col['objects'][1]

			if col_type in ['triangle', 'rectangle', 'pentagon']:
				collision_record['col_obs'].append([x,y])
			elif col_type == "ground":
				collision_record['col_ground'].append([x,y])
			elif col_type == "walls":
				collision_record['col_wall'].append([x,y])


	return agent, collision_record


def run_fixed_sample(trial_num, num_samples=40, bw=50.0, seed=None, noise_params=(0.2,0.8,0.2)):

	if not (seed is None):
		np.random.seed(seed)

	drop_noise, col_mean, col_sd = noise_params

	agent = Agent(trial_num, sample_weight=1, bw=bw, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)
	agent.estimated_rewards = np.array([0.0,0.0,0.0], dtype=np.float64)

	# collision_record = []
	collision_record = {"drop": [], "col_obs": [], "col_wall": [], "col_ground": []}

	for hole in [0,1,2]:

		for _ in range(num_samples):

			sim_data = agent.simulate_world(hole=hole, convert_coordinates=True)
			sim_outcome = sim_data["ball_position"][-1]
			agent.raw_history[hole].append((sim_outcome['x'],1))

			drop_pt = sim_data['drop']['pos']
			collision_record['drop'].append([drop_pt['x'], drop_pt['y']])

			for col in sim_data['collisions']:
				look_point = col['look_point']
				x = look_point['x']
				y = look_point['y']

				col_type = col['objects'][1]

				if col_type in ['triangle', 'rectangle', 'pentagon']:
					collision_record['col_obs'].append([x,y])
				elif col_type == "ground":
					collision_record['col_ground'].append([x,y])
				elif col_type == "walls":
					collision_record['col_wall'].append([x,y])

	ball_pos = agent.world['ball_final_position_unity']['x']
	for hole in [0,1,2]:
		kde = agent.make_kde(hole)
		p_grid = kde.evaluate(np.arange(39,561))
		agent.kde_obs.append(p_grid)
		p = p_grid[int(np.round(ball_pos) - 39)]

		agent.estimated_rewards[hole] = p

	agent.estimated_rewards /= np.sum(agent.estimated_rewards)

	return agent, collision_record


def run_bandit_all_trials(num_runs=30, decision_threshold=0.95, tradeoff_param=0.003, sample_weight=950, bw=30.0, max_iter=100, noise_params=(0.2, 0.8, 0.2), start=0, end=150):

	time_start = time.time()
	world_num_list = os.listdir("../../../data/stimuli/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	judgment_rt = {"trial":[], "run":[], "judgment": [], "num_cols": []}
	collisions = []

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		trial_collisions = []

		for i in range(num_runs):

			print("Run:", i)

			agent, collision_record = run_bandit(tr_num, decision_threshold=decision_threshold, tradeoff_param=tradeoff_param, sample_weight=sample_weight, bw=bw, seed=None, max_iter=100, noise_params=noise_params)

			judgment = np.argmax(agent.estimated_rewards)
			# num_cols = np.sum([len(sim_cols) for sim_cols in collision_record])
			num_cols = len(collision_record['col_obs']) + len(collision_record['col_wall']) + len(collision_record['col_ground'])

			judgment_rt["trial"].append(tr_num)
			judgment_rt["run"].append(i)
			judgment_rt["judgment"].append(judgment)
			judgment_rt["num_cols"].append(num_cols)

			trial_collisions.append(collision_record)

		collisions.append((tr_num, trial_collisions))

		print()


	df_judgment_rt = pd.DataFrame(judgment_rt)

	judgement_rt_filename = "model_performance/judgment_rt/bandit_runs_{}_threshold_{}_tradeoff_{}_sample_weight_{}_bw_{}_noise_params_{}_{}_{}_trial_{}_{}.csv".format(num_runs, decision_threshold, tradeoff_param, sample_weight, bw, noise_params[0], noise_params[1], noise_params[2], start, end)
	collisions_filename = "model_performance/collisions/bandit_runs_{}_threshold_{}_tradeoff_{}_sample_weight_{}_bw_{}_noise_params_{}_{}_{}_trial_{}_{}.pickle".format(num_runs, decision_threshold, tradeoff_param, sample_weight, bw, noise_params[0], noise_params[1], noise_params[2], start, end)

	df_judgment_rt.to_csv(judgement_rt_filename)

	with open(collisions_filename, "wb") as f:
		pickle.dump(collisions, f)

	print("Params:", (decision_threshold, tradeoff_param, sample_weight, bw))
	print("Runtime:", time.time() - time_start)

	return df_judgment_rt, collisions


def run_fixed_sample_all_trials(num_samples=40, bw=50.0, noise_params=(0.2,0.8,0.2), start=0, end=150):

	time_start = time.time()

	world_num_list = os.listdir("../../../data/stimuli/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	judgment_rt = {"trial":[], "hole1": [], "hole2": [], "hole3": [], "num_cols": []}
	collisions = []

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		agent, collision_record = run_fixed_sample(tr_num, num_samples=num_samples, bw=bw, seed=None, noise_params=noise_params)
		posterior = agent.estimated_rewards

		num_cols = len(collision_record['col_obs']) + len(collision_record['col_wall']) + len(collision_record['col_ground'])

		judgment_rt['trial'].append(tr_num)
		judgment_rt['hole1'].append(posterior[0])
		judgment_rt['hole2'].append(posterior[1])
		judgment_rt['hole3'].append(posterior[2])
		judgment_rt['num_cols'].append(num_cols)

		collisions.append((tr_num, collision_record))

	print()

	df_judgment_rt = pd.DataFrame(judgment_rt)

	judgement_rt_filename = "model_performance/judgment_rt/fixed_sample_num_samples_{}_bw_{}_noise_params_{}_{}_{}_trial_{}_{}.csv".format(num_samples, bw, noise_params[0], noise_params[1], noise_params[2], start, end)
	collisions_filename = "model_performance/collisions/fixed_sample_num_samples_{}_bw_{}_noise_params_{}_{}_{}_trial_{}_{}.pickle".format(num_samples, bw, noise_params[0], noise_params[1], noise_params[2], start, end)

	df_judgment_rt.to_csv(judgement_rt_filename)

	with open(collisions_filename, "wb") as f:
		pickle.dump(collisions, f)

	print("Params:", (num_samples, bw))
	print("Runtime:", time.time() - time_start)

	return df_judgment_rt, collisions