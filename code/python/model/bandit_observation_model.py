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


##### EXTRACT GLOBAL PARAMETERS TO RECENTER UNITY COORDINATES AFTER TRANFORMATION #####
##### THIS COULD BE MOVED INTO THE CONVERT COORDINATE FUNCTION? ######

trial_1 = load_trial(1)

original_center_x = trial_1['screen_size']['width']/2
original_center_y = trial_1['screen_size']['height']/2
	
new_center_x, new_center_y = convertCoordinate(original_center_x, original_center_y)
	
# Transformation distance along x and y required to recenter unity coordinates after transformation
DIFF_X = original_center_x - new_center_x
DIFF_Y = original_center_y - new_center_y

# Procedures to transform a trial from pymunk to unity coordinates
def generate_trial_shapes(trial, generate_shapes=True):
	
	shape_trial = copy.deepcopy(trial)
	
	for ob, ob_dict in shape_trial['obstacles'].items():
		
		center_x = ob_dict['position']['x']
		center_y = ob_dict['position']['y']
		
		if generate_shapes:
			ob_shape = np.array(utils.generate_ngon(ob_dict['n_sides'], ob_dict['size']))

		else:
			ob_shape = np.array(ob_dict['shape'])

		rot = ob_dict['rotation']
		rotmat = np.array([[np.cos(rot), -np.sin(rot)],
							[np.sin(rot), np.cos(rot)]])
		
		rotated_shape = (rotmat@ob_shape.T).T
		
		rotated_shape = rotated_shape + np.array([center_x, center_y])[None,:]
		
		ob_dict['shape'] = rotated_shape.tolist()
		
	return shape_trial


def transform_trial(aligned_shape_trial):
	
	transformed_trial = copy.deepcopy(aligned_shape_trial)
	
	bfp_x = transformed_trial['ball_final_position']['x']
	bfp_y = transformed_trial['ball_final_position']['y']
	
	new_bfp_x, new_bfp_y = convertCoordinate(bfp_x, bfp_y)
	
	transformed_trial['ball_final_position'] = {'x': new_bfp_x, 'y': new_bfp_y}

	ball_top_y = bfp_y + transformed_trial['ball_radius']
	_, new_ball_top_y =  convertCoordinate(bfp_x, ball_top_y)

	unity_radius = new_ball_top_y - new_bfp_y
	assert unity_radius > 0
	transformed_trial['ball_radius'] = unity_radius
	
	for ob, ob_dict in transformed_trial['obstacles'].items():
		
		center_x = ob_dict['position']['x']
		center_y = ob_dict['position']['y']
		
		new_center_x, new_center_y = convertCoordinate(center_x, center_y)
		
		ob_dict['position']['x'] = new_center_x
		ob_dict['position']['y'] = new_center_y
		
		ob_dict['shape'] = [list(convertCoordinate(x,y)) for x,y in ob_dict['shape']]
		
	
	new_hole_positions = []
	transformed_trial['old_hole_positions'] = transformed_trial['hole_positions']
	
	hole_y = 700
	for hole_x in transformed_trial['hole_positions']:
		new_hole_x, _ = convertCoordinate(hole_x, hole_y)
		
		new_hole_positions.append(new_hole_x)
		
	transformed_trial['hole_positions'] = new_hole_positions
	
	return transformed_trial


def recenter_transformed_trial(transformed_trial):
	
	original_center_x = transformed_trial['screen_size']['width']/2
	original_center_y = transformed_trial['screen_size']['height']/2
	
	new_center_x, new_center_y = convertCoordinate(original_center_x, original_center_y)
	
	diff_x = original_center_x - new_center_x
	diff_y = original_center_y - new_center_y
	
	transformed_trial['ball_final_position']['x'] += diff_x
	transformed_trial['ball_final_position']['y'] += diff_y
	
	for ob, ob_dict in transformed_trial['obstacles'].items():
		
		ob_dict['position']['x'] += diff_x
		ob_dict['position']['y'] += diff_y
		
		ob_dict['shape'] = np.array([[x+diff_x, y+diff_y] for x,y in ob_dict['shape']])
		
	transformed_trial['hole_positions'] = [x+diff_x for x in transformed_trial['hole_positions']]
	
	return transformed_trial


def transform_center_trial(trial, generate_shapes=True):
	trial_shapes = generate_trial_shapes(trial, generate_shapes=generate_shapes)
	transformed_trial = transform_trial(trial_shapes)
	recentered_trial = recenter_transformed_trial(transformed_trial)
	
	return recentered_trial


def rotmat(rot):
	return np.array([[np.cos(rot), -np.sin(rot)],
					 [np.sin(rot), np.cos(rot)]])



def extract_top_surfaces(ob_dict, unity_coordinates=True):

	shape = np.array(ob_dict['shape'])
	shape_rotated = (rotmat(ob_dict['rotation'])@shape.T).T
	center = np.array([ob_dict['position']['x'], ob_dict['position']['y']])
	shape_centered = shape_rotated + center

	if unity_coordinates:

		vert_list = shape_centered.tolist()
		unity_list = []
		for x,y in vert_list:
			x_unity = x + DIFF_X
			y_unity = y + DIFF_Y
			unity_list.append(list(convertCoordinate(x_unity,y_unity)))

		shape_centered = np.array(unity_list)


	num_sides = len(shape_centered)

	min_x_index = shape_centered[:,0].argmin()
	min_x_pt = shape_centered[min_x_index,:]

	max_x_index = shape_centered[:,0].argmax()
	max_x_coord = shape_centered[max_x_index,0]


	min_neighbor_1 = shape_centered[(min_x_index-1)%num_sides,:]
	min_neighbor_2 = shape_centered[(min_x_index+1)%num_sides,:]


	if min_neighbor_2[1] > min_neighbor_1[1]:
		iter_order = np.arange(min_x_index, min_x_index+num_sides)%num_sides
	else:
		iter_order = np.arange(min_x_index, min_x_index-num_sides,-1)%num_sides


	top_surfaces = []

	# shape_centered = shape_centered.tolist()

	for i in iter_order:

		first_point = shape_centered[i]
		second_point = shape_centered[(i+1)%num_sides]

		top_surfaces.append((first_point, second_point))

		if second_point[0] == max_x_coord:
			break

	return top_surfaces


# Sample a point from one of the top surface segments
def sample_top_surface_pt(top_surfaces):

	# choose which segment proportionate to it's portion of the
	# total top distance
	distances = [np.linalg.norm(pt2 - pt1) for pt1, pt2 in top_surfaces]

	dist_proportion = distances/np.sum(distances)

	segment_index = np.random.choice(range(len(top_surfaces)), p=dist_proportion)
	segment = top_surfaces[segment_index]

	x_coord = np.random.uniform(segment[0][0], segment[1][0])

	slope = (segment[1][1] - segment[0][1])/(segment[1][0] - segment[0][0])
	y_coord = slope*(x_coord - segment[0][0]) + segment[0][1]

	return [x_coord, y_coord]


class Agent:
	
	def __init__(self, trial_num, experiment='inference', decision_threshold=0.01, tradeoff_param=1, sample_weight=200, bw=20, empirical_priors=True, kde_method="FFT", obs_noise=15, ball_noise=15, drop_noise=0.2, col_mean=0.8, col_sd=0.2, hole=None):
		
		self.experiment = experiment
		self.trial_num = trial_num
		self.world = load_trial(trial_num, experiment=experiment, hole=hole, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)


		if experiment == "inference":
			bfp_x = self.world['ball_final_position']['x']
			bfp_y = self.world['ball_final_position']['y']

			bfp_x_unity, bfp_y_unity = convertCoordinate(bfp_x, bfp_y)
			self.world['ball_final_position_unity'] = {'x': bfp_x_unity+DIFF_X, 'y': bfp_y_unity+DIFF_Y}

		self.observed_world = copy.deepcopy(self.world)

		hole_positions_unity = []
		for hole_pos in self.world['hole_positions']:
			hole_x_unity, _ = convertCoordinate(hole_pos, 600)
			hole_positions_unity.append(hole_x_unity + DIFF_X)

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

		# Record of simulation data that may influence looks
		self.trajectories = {'x': [], 'y': []}
		self.collisions = {'x': [], 'y': []}
		self.top_surfaces = []
		

		# Initialize eye position
		x_pos = self.world['screen_size']['height']/2
		y_pos = self.world['screen_size']['height']/2
		self.eye_pos = np.array([x_pos,y_pos])

		self.noise_field = np.ones((501,601))

		# Initialize perceptual noise on the verticies of the obstacles
		for ob_dict in self.observed_world['obstacles'].values():
			n = ob_dict['n_sides']
			ob_dict['vertex_noise'] = np.ones((n))*obs_noise

		self.noise_dict = {
			"ball_noise": ball_noise,
			"vertex_noise": obs_noise,
			"drop_noise": drop_noise,
			"collision_noise": col_sd 
		}

		# Empirically derived look probs
		# self.look_probs = {
		# 	"drop": 0.02,
		# 	"collision_obs": 0.84,
		# 	"collision_wall": 0.02,
		# 	"ground": 0.12
		# }

		# Arbitrarily chosen look probs
		self.look_probs = {
			"drop": 0.5,
			"collision_obs": 0.8,
			"collision_wall": 0.02,
			"ground": 0.5
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
					# + 50 hack to align priors with unity translated coordinates
					self.raw_history[i].append((row['x'] + DIFF_X, row['count'] / 100.0))
			else:
				# assume 1 sample has been drawn at each int x in [0, 600), each with weight=1
				for j in range(50,650):
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
		x_grid = np.arange(49,651)
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

		bfp_x = self.observed_world['ball_final_position_unity']['x']

		if self.kde_method == "FFT":
			bfp_x = int(np.round(bfp_x) - DIFF_X)

		if self.kde_method == "scikit":
			log_p_truth = kde.score(np.array([bfp_x])[:, np.newaxis])
			p_truth = np.exp(log_p_truth)
		elif self.kde_method == "FFT":
			p_truth = p[bfp_x]

		self.estimated_rewards[hole] = p_truth
		self.entropy = entropy(self.estimated_rewards)
		return kde

	def belief_update_hole(self, sim_outcome, hole, perception=True):
		self.raw_history[hole].append((sim_outcome, self.sample_weight))  # each real sample has weight=10
		kde = self.make_kde(hole)
		self.kde_obs[hole] = kde

		# if self.experiment == 'combined':
		# 	for t in self.collisions['timestamp']:
		# 		self.raw_history_sound[hole].append((t, 50))

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


	# Observation Model
	def fixate_location(self, eye_pos):
		self.eye_pos = np.array(eye_pos)
		return self.eye_pos
		

	# Foveal Peripheral Operating Characteristic, defines noise level
	# as a function of distance        
	# def fpoc(self, distance, spread=1000):
	#     return -(0.9)*np.exp(-(distance**2)/spread)+1

	def fpoc(self, eye_pos, spread=1000):

		y,x = np.mgrid[100:601:1, 50:651:1]
		pos = np.dstack((x,y))

		probs = multivariate_normal.pdf(pos, eye_pos, cov=spread)

		scale = (1/np.max(probs))*0.9*-1

		scale_flip_probs = probs*scale
		kernel = scale_flip_probs + 1

		return kernel


	def look_steps(self, events, look_list, var=np.array([[36,0],[0,36]])):

		if type(events) == dict:

			look_prob = self.look_probs['drop']
			if np.random.binomial(1, look_prob):

				drop_pos = events['pos']
				drop_pos_arr = np.array([drop_pos['x'], drop_pos['y']])
				sampled_pos = np.random.multivariate_normal(drop_pos_arr, var)
				look_list.append(("drop", sampled_pos))
			

		elif type(events) == list:

			for event in events:
				col_type = event['objects'][1]

				if col_type in {'triangle', 'pentagon', 'rectangle'}:
					look_prob = self.look_probs['collision_obs']
				elif col_type == "walls":
					look_prob = self.look_probs['collision_wall']
				elif col_type == "ground":
					look_prob = self.look_probs['ground']
				else:
					raise Exception('No look probability for collision type "{}"'.format(col_type))

				if np.random.binomial(1, look_prob):

					contact_point = event['contact_point']
					col_pos_arr = np.array([contact_point['x'] + DIFF_X, contact_point['y'] + DIFF_Y])
					sampled_pos = np.random.multivariate_normal(col_pos_arr, var)
					look_list.append((col_type, sampled_pos))

		else:
			raise Exception("Events is not drop or collisions.")

		return look_list



	def choose_eye_pos(self, method="random_top", hole=None, sim_data=None):

		if method == 'random_top':
			shape_name = np.random.choice(list(self.observed_world['obstacles'].keys()))
			top_surfaces = extract_top_surfaces(self.observed_world['obstacles'][shape_name])
			eye_pos = sample_top_surface_pt(top_surfaces)

			return eye_pos

		if method == 'simulation_look':
			assert not (hole is None)
			assert not (sim_data is None)
			looks = []
			# sim_data = self.simulate_world(hole, convert_coordinates=False)
			# Might be weird if we have a simulation where the ball dind't make it to the ground
			end_step = len(sim_data['ball_position'])
			looks = self.look_steps(sim_data['drop'], looks)
			looks = self.look_steps(sim_data['collisions'], looks)

			return looks

		else:
			raise Exception("Eye position method {} not implemented.".format(method))

	
	def observe(self, eye_pos=None):
		if eye_pos is None:
			eye_pos = self.eye_pos

		noise_reduction_kernel = self.fpoc(self.eye_pos)

		self.noise_field *= noise_reduction_kernel

		return self.noise_field
	
	
	# Sample shapes (vertex locations) based on the current multipliers in the noise field

	def generate_vertex_noise(self, ob_dict):

		n = ob_dict['n_sides']
		side_length = ob_dict['size']

		rot = ob_dict['rotation']
		x = ob_dict['position']['x']
		y = ob_dict['position']['y']
		pos = np.array([x,y])

		shape = np.array(utils.generate_ngon(n, side_length))
		shape_noise = self.noise_dict['vertex_noise']

		sds = []
		for i in range(n):
			vert = np.array(shape[i,:])
			rot_vert = rotmat(rot)@vert
			pymunk_x, pymunk_y = rot_vert + pos

			unity_x_uncentered, unity_y_uncentered = convertCoordinate(pymunk_x, pymunk_y)

			unity_x = int(np.round(unity_x_uncentered))
			unity_y = int(np.round(unity_y_uncentered))


			row = unity_y
			col = unity_x


			# The noise reduction multiplier is computed based on the unity coordinate
			# of the vertex because the noise field is in the unity space.
			# But the noise value is added to the original pymunk vertex so that we can run it in the simulator
			multiplier = self.noise_field[row, col]
			v_sd = multiplier *shape_noise

			sds.append(v_sd)

			var = v_sd**2
			noise = np.random.multivariate_normal([0,0], np.diag([var,var]))
			shape[i,:] = vert + noise

		return shape, sds


	def generate_ball_noise(self):
		ball_sd = self.noise_dict['ball_noise']

		bfp = self.world['ball_final_position']
		bfp_x = bfp['x']
		bfp_y = bfp['y']

		bfp_x_unity, bfp_y_unity = convertCoordinate(bfp_x, bfp_y)
		bfp_x_unity = int(np.round(bfp_x_unity))
		bfp_y_unity = int(np.round(bfp_y_unity))

		row = bfp_y_unity
		col = bfp_x_unity

		multiplier = self.noise_field[row, col]

		modified_sd = ball_sd*multiplier

		ball_noise = np.random.normal(0, modified_sd)

		return ball_noise, modified_sd


	def get_drop_sd(self, hole):
		drop_sd = self.noise_dict['drop_noise']

		hp_x = self.world['hole_positions'][hole]
		hp_y = 600

		hp_x_unity, hp_y_unity = convertCoordinate(hp_x, hp_y)

		hp_x_unity = int(np.round(hp_x_unity + DIFF_X))
		hp_y_unity = int(np.round(hp_y_unity + DIFF_Y))

		row = hp_y_unity - 100
		col = hp_x_unity - 50

		multiplier = self.noise_field[row, col]

		modified_sd = drop_sd*multiplier

		return modified_sd


	def sample_world(self):

		if self.experiment == "inference": 
			actual_bfp = self.world['ball_final_position_unity']['x']
			ball_noise, _ = self.generate_ball_noise()
			# print(ball_noise)
			observed_pos = actual_bfp + ball_noise
			if observed_pos > 620:
				observed_pos = 620
			if observed_pos < 80:
				observed_pos = 80
			self.observed_world['ball_final_position_unity']['x'] = observed_pos
		
		for _, ob_dict in self.observed_world['obstacles'].items():
			
			noised_shape, sds = self.generate_vertex_noise(ob_dict)
				
			ob_dict['shape'] = noised_shape.tolist()
	
	# Simulate in the current world given the agent's perceptual uncertainty  
	def simulate_world(self, hole=None, convert_coordinates=True, distorted=True, perception=True):

		# print("Agent sim function")

		if perception:
			world = self.observed_world
		else:
			world = self.world

		# print(world)

		if (hole is None):
			hole = world['hole_dropped_into']
		else:
			world['hole_dropped_into'] = hole

		if perception:
			drop_sd = self.get_drop_sd(hole)
			self.observed_world["drop_noise"] = drop_sd
		else:
			distorted = False
			
		
		sim_data = engine.run_simulation(world,
										 self.noise_field,
										 convert_coordinates=convert_coordinates,
										 distorted=distorted)
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

			world = transform_center_trial(world, generate_shapes=gen_shapes)

			
		ball_pos = world['ball_final_position']
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

	def visualize_noise_field(self, unity_background=True, save=True, show_eye=True):

		if unity_background:

			img = plt.imread("../../../figures/images/jpg/final/world_{}.png".format(self.trial_num))
			y,x = np.mgrid[500:-1:-1, 0:601:1]

		else:
			self.visualize_agent_state()
			img = plt.imread("visuals_agent/distort_world.png")
			y,x = np.mgrid[600:99:-1, 50:651:1]


		nf = self.noise_field

		fig, ax = plt.subplots()
		ax.imshow(img)

		ax.contourf(x,y,nf,alpha=0.3)

		ax = draw_eye_plt(ax, self.eye_pos)
		ax.axis("off")

		if save:
			plt.savefig("visuals_agent/heatmaps/world_" + str(self.trial_num).zfill(3) + ".png", dpi=200)

		return ax
			

	
	def visualize_simulation(self, sim_data=None):
		
		if sim_data is None:
			sim_data = self.simulate_world()
			
		visual.visualize(self.observed_world,
						 sim_data, 
						 save_images=True,
						 make_video=True,
						 video_name="agent_vid",
						 eye_pos=self.eye_pos)


	def extract_shapes(self, unity_coordinates=True):
		shapes = []
		vertex_noise = []

		for shape, ob_dict in self.observed_world['obstacles'].items():

			verticies = np.array(ob_dict['shape'])
			pos_x = ob_dict['position']['x']
			pos_y = ob_dict['position']['y']

			if unity_coordinates:

				rot_vert = (rotmat(ob_dict['rotation'])@verticies.T).T
				translate_vert = rot_vert + np.array([pos_x + DIFF_X, pos_y + DIFF_Y])

				verticies = [list(convertCoordinate(x,y)) for x,y in translate_vert.tolist()]


			shapes.append((shape, verticies))
			vertex_noise.append((shape, ob_dict['vertex_noise']))


		return shapes, vertex_noise

	def get_ball_pos(self, unity_coordinates=True):

		ball_pos = self.observed_world['ball_final_position_unity']

		x = ball_pos['x']
		y = ball_pos['y']

		# if unity_coordinates:
		# 	x, y = convertCoordinate(x,y)
		# 	x += DIFF_X
		# 	y += DIFF_Y

		return {'x': x, 'y':y}


def record_timestep(record, agent, i, action, sim_tuple=None):

	record['timestep'].append(i)
	record['action'].append(action)
	record['eye_pos'].append(agent.eye_pos)
	record['expected_reward'].append(np.copy(agent.estimated_rewards))
	record['conditional_entropy'].append(np.copy(agent.uncertainty))
	record['entropy'].append(agent.entropy)
	record['density_samples'].append(copy.deepcopy(agent.raw_history))
	shapes, vertex_noise = agent.extract_shapes()
	record['shapes'].append(shapes)
	record['ball_positions'].append(agent.get_ball_pos())
	record['vertex_noise'].append(vertex_noise)
	record['sim_data'].append(sim_tuple)

	return record

def run_bandit(trial_num, decision_threshold=0.65, tradeoff_param=1, sample_weight=200, bw=20, seed=None, max_iter=100, empirical_priors=True, kde_method="FFT", perception=True, noise_params=(15, 15, 0.2, 0.8, 0.2)):

	if not (seed is None):
		np.random.seed(seed)

	obs_noise, ball_noise, drop_noise, col_mean, col_sd = noise_params

	agent = Agent(trial_num, decision_threshold=decision_threshold, tradeoff_param=tradeoff_param, sample_weight=sample_weight, bw=bw, empirical_priors=empirical_priors, kde_method=kde_method, obs_noise=obs_noise, ball_noise=ball_noise, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)
	agent.sample_world()
	agent._initialize_rewards_uncertainty(empirical_priors=empirical_priors)
	agent.entropy = entropy(agent.estimated_rewards)
	shapes, vertex_noise = agent.extract_shapes()
	ball_pos = agent.get_ball_pos()

	record = {"timestep": [0], 
	"action": ["initialize"],
	"eye_pos": [agent.eye_pos],
	"expected_reward": [np.copy(agent.estimated_rewards)],
	"conditional_entropy": [np.copy(agent.uncertainty)],
	"entropy": [agent.entropy],
	"density_samples": [copy.deepcopy(agent.raw_history)],
	"shapes": [shapes],
	"ball_positions": [ball_pos],
	"vertex_noise": [vertex_noise],
	"sim_data": ["NA"]}

	i = 0
	while agent.entropy > agent.decision_threshold and i < max_iter:

		i += 1

		if perception and i != 1:
			agent.sample_world()

		hole = agent.choose_hole()
		sim_data = agent.simulate_world(hole=hole, convert_coordinates=True, perception=perception)
		sim_outcome = sim_data['ball_position'][-1]
		agent.belief_update_hole(sim_outcome['x'], hole, perception=perception)

		record_timestep(record, agent, i, "simulate", (hole, sim_outcome, sim_data['ball_position']))

		if perception:
			sim_looks = agent.choose_eye_pos(method="simulation_look", hole=hole, sim_data=sim_data)

			for look in sim_looks:

				i += 1
				x = look[1][0]
				y = look[1][1]

				agent.fixate_location([x,y])
				agent.observe()

				record_timestep(record, agent, i, "sim_look", (hole, sim_outcome, sim_data['ball_position']))

	return pd.DataFrame(record), agent


# Taken from Gregory Gunderson's demo on the log-sum-exp trick
# https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
def logsumexp(x):
	c = np.max(x)
	return c + np.log(np.sum(np.exp(x - c)))

# A stable way to normalize log probabilites
# Expects a vector of log probabilities
def normalize_logsumexp(x): return np.exp(x - logsumexp(x))


def run_fixed_sample(trial_num, num_samples, bw=20, seed=None, perception=True, noise_params=(15, 15, 0.2, 0.8, 0.2)):

	obs_noise, ball_noise, drop_noise, col_mean, col_sd = noise_params

	if not (seed is None):
		np.random.seed(seed)

	agent = Agent(trial_num, sample_weight=1, bw=bw, obs_noise=obs_noise, ball_noise=ball_noise, drop_noise=drop_noise, col_mean=col_mean, col_sd=col_sd)
	agent.sample_world()
	# agent._initialize_rewards_uncertainty(empirical_priors=empirical_priors)
	agent.estimated_rewards = np.array([0.0,0.0,0.0], dtype=np.float64)
	agent.entropy = "NA"
	shapes, vertex_noise = agent.extract_shapes()
	ball_pos = agent.get_ball_pos()

	starting_hole = np.argmin(np.abs(np.array(agent.world['hole_positions_unity']) - ball_pos['x']))
	sim_order = np.roll(np.arange(3), -starting_hole)

	record = {"timestep": [0], 
	"action": ["initialize"],
	"eye_pos": [agent.eye_pos],
	"expected_reward": [np.copy(agent.estimated_rewards)],
	"conditional_entropy": [np.copy(agent.uncertainty)],
	"entropy": [agent.entropy],
	"density_samples": [copy.deepcopy(agent.raw_history)],
	"shapes": [shapes],
	"ball_positions": [ball_pos],
	"vertex_noise": [vertex_noise],
	"sim_data": ["NA"]}

	timestep = 0

	sample_lists = [[], [], []]
	for _ in range(num_samples):

		for hole in sim_order:

			timestep +=1

			if perception and timestep != 1:
				agent.sample_world()

			sim_data = agent.simulate_world(hole=hole, convert_coordinates=True, perception=perception)
			sim_outcome = sim_data["ball_position"][-1]
			obs_outcome = agent.observed_world['ball_final_position_unity']['x']

			agent.raw_history[hole].append((sim_outcome['x'],1))


			# score = np.exp(-((sim_outcome['x'] - obs_outcome)/loss_var)**2)
			# score = scipy.stats.norm.logpdf(sim_outcome['x'], loc=obs_outcome, scale=loss_var)
			# print(score)
			# agent.estimated_rewards[hole] = agent.estimated_rewards[hole] + score
			# print(agent.estimated_rewards)

			record_timestep(record, agent, timestep, "simulate", (hole, sim_outcome, sim_data['ball_position']))

			if perception:

				sim_looks = agent.choose_eye_pos(method="simulation_look", hole=hole, sim_data=sim_data)

				for look in sim_looks:

					timestep += 1
					x = look[1][0]
					y = look[1][1]

					agent.fixate_location([x,y])
					agent.observe()

					record_timestep(record, agent, timestep, "sim_look", (hole, sim_outcome, sim_data['ball_position']))

	ball_pos = agent.observed_world["ball_final_position_unity"]['x'] - DIFF_X
	for hole in [0, 1, 2]:
		kde = agent.make_kde(hole)
		if agent.kde_method == "FFT":
			p_grid = kde.evaluate(np.arange(49, 651))
			agent.kde_obs.append(p_grid)
			p = p_grid[int(np.round(ball_pos))]
		else:
			raise Exception("KDE method {} not implemented for fixed sample model".format(agent.kde_method))

		agent.estimated_rewards[hole] = p


	return pd.DataFrame(record), agent






def graph_conditional_dist(ax, hole, density_samples, multiplier=8000, offset=565, unity_background=False, kde_method="FFT"):
	
	color = ['red', 'blue', 'green']
	
	x, weights = zip(*density_samples[hole])
	x = np.array(x)
	weights = np.array(weights)
	
	if unity_background:
		x_grid = np.linspace(10,585,600)
		x = x - 50
		offset = 480
	else:
		x_grid = np.linspace(90, 610, 520)


	if kde_method == "FFT":
		kde = FFTKDE(kernel="gaussian", bw=20).fit(x, weights=weights)
		p = kde.evaluate(x_grid)*-multiplier
	elif kde_method == "scikit":
		kde = KernelDensity(kernel="gaussian", bandwidth=20).fit(x[:,np.newaxis], sample_weight=weights)
		p = np.exp(kde.score_samples(x_grid[:,np.newaxis]))*-multiplier
	
	p += offset
	
	

	if unity_background:
		p = np.insert(p, 0, offset)
		p = np.append(p, offset)
		x_grid = np.insert(x_grid,0,15)
		x_grid = np.append(x_grid,585)
	else:
		p = np.insert(p, 0, offset)
		p = np.append(p, offset)
		x_grid = np.insert(x_grid, 0, 90)
		x_grid = np.append(x_grid, 610)
	
	col = color[hole]
	ax.fill(x_grid, p, color=col, alpha=0.5)
	
	return ax

def visualize_frame(trial,
					action,
					frame_num, 
					shapes,
					ball_pos,
					eye_pos,
					density_samples,
					kde_method="FFT",
					save=False):
	
	frame_name = "frame" + str(frame_num).zfill(3)

	for shape_name, shape in shapes:
		trial['obstacles'][shape_name]['shape'] = shape

	visual.snapshot(trial, 
					"visuals_agent/frames/", 
					frame_name,
					ball_pos=ball_pos,
					unity_coordinates=True)

	img = plt.imread("visuals_agent/frames/{}.png".format(frame_name))
	fig, ax = plt.subplots()
	ax.imshow(img)
	ax.axis("off")

	if action == "simulate":
		label = "simulate"
	elif action == "sim_look":
		label = "look"
	elif action == "initialize":
		label = "initialize"

	ax.text(350,
			650,
			label,
			fontsize=16,
			verticalalignment="center",
			horizontalalignment="center")

	graph_conditional_dist(ax, 0, density_samples, kde_method=kde_method)
	graph_conditional_dist(ax, 1, density_samples, kde_method=kde_method)
	graph_conditional_dist(ax, 2, density_samples, kde_method=kde_method)

	draw_eye_plt(ax, eye_pos)

	ax.text(220, 60,
			"1",
			color="red",
			fontsize=10,
			verticalalignment="center",
			horizontalalignment="center")
	ax.text(350, 60,
			"2", 
			color="blue",
			fontsize=10,
			verticalalignment="center",
			horizontalalignment="center")
	ax.text(480, 60,
			"3",
			color="green",
			fontsize=10,
			verticalalignment="center",
			horizontalalignment="center")

	if save:
		plt.savefig("visuals_agent/frames/{}.png".format(frame_name), 
					dpi=200)
		
	return ax

def draw_eye_plt(ax, eye_pos):

	eye_x = eye_pos[0]
	eye_y = 700 - eye_pos[1]

	outline = plt.Circle((eye_x, eye_y), 11, color="black", zorder=3)
	sclera = plt.Circle((eye_x, eye_y), 10, color="white", zorder=3)
	iris = plt.Circle((eye_x, eye_y), 6, color="deepskyblue", zorder=3)
	pupil = plt.Circle((eye_x, eye_y), 2, color="black", zorder=3)

	ax.add_patch(outline)
	ax.add_patch(sclera)
	ax.add_patch(iris)
	ax.add_patch(pupil)

	return ax

def visualize_simulation(ax, trial, sim_data, frame_num, eye_pos=None):
	
	color = ["red", "blue", "green"]
	frame_name = "frame" + str(frame_num).zfill(3)
	hole, outcome, trajectory = sim_data
	
	x = []
	y = []
	for pt in trajectory:
		x.append(pt['x'])
		y.append(700 - pt['y'])
	
	start_pt = trajectory[0]
	start_x = start_pt['x']
	start_y = 700 - start_pt['y']
	
	col = color[hole]
	ax.plot(x, y, "--", color=col)
	circle1 = plt.Circle((start_x, start_y), 20, color=col)
	circle2 = plt.Circle((outcome['x'],700-outcome['y']), 20, color=col)
	
	ax.add_patch(circle1)
	ax.add_patch(circle2)

	if not (eye_pos is None):
		ax = draw_eye_plt(ax, eye_pos)
	
	plt.savefig("visuals_agent/frames/{}.png".format(frame_name), 
				dpi=200)
	
	return ax
	

def visualize_trial(df_trial, trial_num, viz_type="pdf", frame_rate=3, kde_method="FFT"):
	
	num_rows = df_trial.shape[0]
	
	trial = transform_center_trial(load_trial(trial_num))
	ball_pos = trial['ball_final_position']
	
	frame_num = 0
	for i in range(num_rows):
		
		eye_pos = df_trial['eye_pos'][i]
		ball_pos = df_trial['ball_positions'][i]
		action = df_trial['action'][i]
		
		if action == "initialize":
			shapes = df_trial['shapes'][i]
			density_samples = df_trial['density_samples'][i]
			ax = visualize_frame(trial,
								 action,
								 frame_num,
								 shapes,
								 ball_pos,
								 eye_pos,
								 density_samples,
								 kde_method=kde_method,
								 save=True)
			
		elif action == "simulate":
			
			shapes = df_trial['shapes'][i]
			sim_data = df_trial['sim_data'][i]
		
			density_samples_before = df_trial['density_samples'][i-1]

			eye_pos = df_trial['eye_pos'][i]
		
			ax = visualize_frame(trial, 
								 action,
								 frame_num, 
								 shapes, 
								 ball_pos,
								 eye_pos, 
								 density_samples_before,
								 kde_method=kde_method)

			ax = visualize_simulation(ax, trial, sim_data, frame_num, eye_pos)
			
			frame_num += 1
			
			density_samples_after = df_trial['density_samples'][i]
			
			ax = visualize_frame(trial,
								 action,
								 frame_num,
								 shapes,
								 ball_pos,
								 eye_pos, 
								 density_samples_after,
								 kde_method=kde_method)
			
			ax = visualize_simulation(ax, trial, sim_data, frame_num, eye_pos)

		elif action == "sim_look":

			shapes = df_trial['shapes'][i]
			sim_data = df_trial['sim_data'][i]

			density_samples = df_trial['density_samples'][i]

			eye_pos = df_trial['eye_pos'][i]

			ax = visualize_frame(trial,
								 action,
								 frame_num,
								 shapes,
								 ball_pos,
								 eye_pos,
								 density_samples,
								 kde_method=kde_method)

			ax = visualize_simulation(ax, trial, sim_data, frame_num, eye_pos)
			
		elif action == "top_look":
			
			shapes_before = df_trial['shapes'][i-1]
			shapes_after = df_trial['shapes'][i]
			density_samples = df_trial['density_samples'][i]
			
			ax = visualize_frame(trial,
								 action,
								 frame_num,
								 shapes_before,
								 ball_pos,
								 eye_pos,
								 density_samples,
								 kde_method=kde_method,
								 save=True)
			
			frame_num += 1
			
			ax = visualize_frame(trial,
								 action,
								 frame_num,
								 shapes_after,
								 ball_pos,
								 eye_pos,
								 density_samples,
								 kde_method=kde_method,
								 save=True)
		
		
		frame_num += 1
		
		plt.close("all")
		
	path = 'visuals_agent/frames'

	if viz_type == "pdf":
		frames = sorted(os.listdir(path))
		imgs = []
			
		for fr in frames:
			if fr.startswith("."):
				continue
			img_path = path + "/" + fr
			img = PIL.Image.open(img_path)
			conv_im = img.convert("RGB")
			imgs.append(conv_im)
			os.remove(img_path)
				
		imgs[0].save("visuals_agent/trials/trial" + str(trial_num).zfill(3) + ".pdf",
					 save_all=True,
					 quality=100,
					 append_images = imgs[1:])

	elif viz_type == "video":
		subprocess.run("ffmpeg -framerate {} -i visuals_agent/frames/frame%03d.png -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p visuals_agent/trial_videos/trial{}.mp4".format(frame_rate, str(trial_num).zfill(3)).split(" "))
		for file in os.listdir("visuals_agent/frames"):
			os.unlink("visuals_agent/frames/{}".format(file))
		
		
		
	return ax






def run_bandit_all_trials(num_runs=30, decision_threshold=0.6, tradeoff_param=0.003, sample_weight=200, bw=20, max_iter=100, empirical_priors=True, kde_method="FFT", noise_params=(15.0, 15.0, 0.2, 0.8, 0.2), start=0, end=150):

	time_start = time.time()
	world_num_list = os.listdir("data/selection/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	judgment_rt = {"trial":[], "run":[], "judgment": [], "num_sims": [], "num_looks": []}
	looks = []

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		trial_list = []

		for i in range(num_runs):
			print("Run:", i)

			# run_record = {}

			df_record, agent = run_bandit(tr_num, decision_threshold, tradeoff_param, sample_weight=sample_weight, bw=bw, seed=None, max_iter=100, empirical_priors=empirical_priors, kde_method=kde_method, noise_params=noise_params)

			num_sims = (df_record["action"] == "simulate").sum()
			num_looks = (df_record["action"] == "sim_look").sum()

			judgment = np.argmax(agent.estimated_rewards)
			look_list = np.stack(df_record[(df_record["action"] == "sim_look") | 
				(df_record["action"] == "initialize")]["eye_pos"].to_numpy())

			look_list = look_list - np.array([[DIFF_X, DIFF_Y]])


			judgment_rt["trial"].append(tr_num)
			judgment_rt["run"].append(i)
			judgment_rt["judgment"].append(judgment)
			judgment_rt["num_sims"].append(num_sims)
			judgment_rt["num_looks"].append(num_looks)

			trial_list.append(look_list)

		looks.append((tr_num, trial_list))

		print()

	df_judgment_rt = pd.DataFrame(judgment_rt)

	if empirical_priors:
		prior_type = "empirical_prior"
	else:
		prior_type = "heuristic_prior"

	judgement_rt_filename = "model_performance/judgment_rt/bandit_runs_{}_threshold_{}_tradeoff_{}_sample_weight_{}_bw_{}_look_probs_{}_{}_{}_{}_noise_params_{}_{}_{}_{}_{}_{}_trial_{}_{}.csv".format(num_runs, decision_threshold, tradeoff_param, sample_weight, bw, agent.look_probs['drop'], agent.look_probs['collision_obs'], agent.look_probs['collision_wall'], agent.look_probs['ground'], noise_params[0], noise_params[1], noise_params[2], noise_params[3], noise_params[4], prior_type, start, end)
	looks_filename = "model_performance/looks/bandit_runs_{}_threshold_{}_tradeoff_{}_sample_weight_{}_bw_{}_look_probs_{}_{}_{}_{}_noise_params_{}_{}_{}_{}_{}_{}_trial_{}_{}.pickle".format(num_runs, decision_threshold, tradeoff_param, sample_weight, bw, agent.look_probs['drop'], agent.look_probs['collision_obs'], agent.look_probs['collision_wall'], agent.look_probs['ground'], noise_params[0], noise_params[1], noise_params[2], noise_params[3], noise_params[4], prior_type, start, end)

	df_judgment_rt.to_csv(judgement_rt_filename)

	with open(looks_filename, "wb") as f:
		pickle.dump(looks, f)

	print("Params:", (decision_threshold, tradeoff_param, sample_weight, bw))
	print("Runtime:", time.time() - time_start)

	return df_judgment_rt, looks


def run_bandit_no_perception_all_trials(num_runs=30, decision_threshold=0.6, sample_weight=200, tradeoff_param=0.003, max_iter=100, empirical_priors=True, kde_method="FFT", start=0, end=150):
	time_start = time.time()
	world_num_list = os.listdir("data/selection/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	out_dict = {"trial":[], "run":[], "judgment": [], "num_sims": []}

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		trial_list = []

		for i in range(num_runs):
			print("Run:", i)

			df_record, agent = run_bandit(tr_num, decision_threshold, tradeoff_param, seed=i, max_iter=100, empirical_priors=empirical_priors, kde_method=kde_method, sample_weight=sample_weight, perception=False)

			num_sims = (df_record["action"] == "simulate").sum()
			judgment = np.argmax(agent.estimated_rewards)

			out_dict["trial"].append(tr_num)
			out_dict["run"].append(i)
			out_dict["judgment"].append(judgment)
			out_dict["num_sims"].append(num_sims)

		print()

	df_judgment = pd.DataFrame(out_dict)

	if empirical_priors:
		prior_type = "empirical_prior"
	else:
		prior_type = "heuristic_prior"

	filename = "model_performance/no_perception/bandit_runs_{}_threshold_{}_sample_weight_{}_{}_trial_{}_{}.csv".format(num_runs, decision_threshold, sample_weight, prior_type, start, end)

	df_judgment.to_csv(filename)


	print("Runtime:", time.time() - time_start)

	return df_judgment


def run_fixed_sample_all_trials(num_samples=100, bw=20, noise_params=(15.0,15.0,0.2,0.8,0.2), start=0, end=150):

	time_start = time.time()
	world_num_list = os.listdir("data/selection/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	judgment_rt = {"trial":[], "hole1": [], "hole2": [], "hole3": [], "num_sims": [], "num_looks": []}
	looks = []

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		df_record, agent = run_fixed_sample(tr_num, num_samples, bw=20, perception=True, noise_params=noise_params)

		# exp_rewards = np.exp(agent.estimated_rewards)
		# agent.estimated_rewards = normalize_logsumexp(agent.estimated_rewards)
		agent.estimated_rewards = agent.estimated_rewards/np.sum(agent.estimated_rewards)

		num_sims = (df_record["action"] == "simulate").sum()
		num_looks = (df_record["action"] == "sim_look").sum()

		look_list = np.stack(df_record[(df_record["action"] == "sim_look") | 
			(df_record["action"] == "initialize")]["eye_pos"].to_numpy())

		look_list = look_list - np.array([[DIFF_X, DIFF_Y]])

		judgment_rt["trial"].append(tr_num)
		judgment_rt["hole1"].append(agent.estimated_rewards[0])
		judgment_rt["hole2"].append(agent.estimated_rewards[1])
		judgment_rt["hole3"].append(agent.estimated_rewards[2])
		judgment_rt["num_sims"].append(num_sims)
		judgment_rt["num_looks"].append(num_looks)

		looks.append((tr_num, look_list))

	df_judgment_rt = pd.DataFrame(judgment_rt)


	judgement_rt_filename = "model_performance/judgment_rt/fixed_sample_num_samples_{}_bw_{}_look_probs_{}_{}_{}_{}_noise_params_{}_{}_{}_{}_{}_trial_{}_{}.csv".format(num_samples, bw, agent.look_probs['drop'], agent.look_probs['collision_obs'], agent.look_probs['collision_wall'], agent.look_probs['ground'], noise_params[0], noise_params[1], noise_params[2], noise_params[3], noise_params[4], start, end)
	looks_filename = "model_performance/looks/fixed_sample_num_samples_{}_bw_{}_look_probs_{}_{}_{}_{}_noise_params_{}_{}_{}_{}_{}_trial_{}_{}.pickle".format(num_samples, bw, agent.look_probs['drop'], agent.look_probs['collision_obs'], agent.look_probs['collision_wall'], agent.look_probs['ground'], noise_params[0], noise_params[1], noise_params[2], noise_params[3], noise_params[4], start, end)

	df_judgment_rt.to_csv(judgement_rt_filename)

	with open(looks_filename, "wb") as f:
		pickle.dump(looks, f)

	print("Params:", (num_samples, bw))
	print("Runtime:", time.time() - time_start)

	return df_judgment_rt, looks


def run_fixed_sample_no_perception_all_trials(num_samples=5, loss_var=100, seed=1, start=0, end=150):

	time_start = time.time()
	world_num_list = os.listdir("data/selection/ground_truth/")
	world_num_list = sorted([int(x[6:-5]) for x in world_num_list])

	world_num_list = world_num_list[start:end]

	judgment_dict = {"trial":[], "hole1": [], "hole2": [], "hole3": [], "num_sims": []}

	np.random.seed(seed)

	for tr_num in world_num_list:
		print("Trial:", tr_num)

		df_record, agent = run_fixed_sample(tr_num, num_samples, loss_var, perception=False)

		agent.estimated_rewards = normalize_logsumexp(agent.estimated_rewards)

		num_sims = (df_record['action'] == "simulate").sum()

		judgment_dict["trial"].append(tr_num)
		judgment_dict["hole1"].append(agent.estimated_rewards[0])
		judgment_dict["hole2"].append(agent.estimated_rewards[1])
		judgment_dict["hole3"].append(agent.estimated_rewards[2])
		judgment_dict['num_sims'].append(num_sims)

	df_judgment = pd.DataFrame(judgment_dict)

	filename = "model_performance/no_perception/fixed_sample_num_samples_{}_loss_var_{}_trial_{}_{}.csv".format(num_samples, loss_var, start, end)

	df_judgment.to_csv(filename)

	print("Runtime:", time.time() - time_start)

	return df_judgment







