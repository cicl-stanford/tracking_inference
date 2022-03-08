# import libraries 
from __future__ import division
import numpy as np
import json
import convert_coordinate
import csv
import scipy.stats as st

# import files 
import config

def main():
	# combine_video_and_audio(video = 'data/videos/test.mp4', audio = 'data/wav/test.wav')
	tmp = loss(prop = 60, target = 63, sd = 2)
	print("tmp", tmp)

def generate_ngon(n, rad):
	""" Function to generate ngons (e.g. Pentagon) """ 
	pts = []
	ang = 2 * np.pi / n
	for i in range(n):
		pts.append((np.sin(ang * i) * rad, np.cos(ang * i) * rad))
	return pts

def get_vertices(shape):
	vertices = []
	for v in shape.get_vertices():
		x, y = v.rotated(shape.body.angle) + shape.body.position
		vertices.append((x, y))
	return vertices

def radius_reg_poly(side_length, n):
	return side_length/(2*(np.pi/n))

def get_points_on_segment(x1, y1, x2, y2, bound_1, bound_2):
	# divide the segment into 6 parts
	num = 20
	pts = []
	for i in range(num):
		x, y = x1/num*i + x2/num*(num-i), y1/num*i + y2/num*(num-i)
		if bound_1 <= x <= bound_2:
			x, y = convert_coordinate.convertCoordinate(x, y)
			pts.append({'x': x, 'y': y})
	return pts

def get_top_surfaces(pts, center):
	x, y = center
	top_surfaces = []

	for i in range(len(pts)-1):
		x1, y1, x2, y2 = pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1]
		mid_point = (y1+y2)/2
		if mid_point > y:
			if pts[i][0] <= pts[i+1][0]:
				top_surfaces.append([pts[i], pts[i+1]])
			else:
				top_surfaces.append([pts[i+1], pts[i]])

	if len(pts) > 2:
		x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]
		mid_point = (y1 + y2) / 2
		if mid_point > y:
			if pts[0][0] <= pts[-1][0]:
				top_surfaces.append([pts[0], pts[-1]])
			else:
				top_surfaces.append([pts[-1], pts[0]])

	return top_surfaces

def overlap(s1, s2):
	# check if the x ranges of two surfaces overlap with each other
	x10, x11 = s1[0][0], s1[1][0]
	x20, x21 = s2[0][0], s2[1][0]
	if x10 < x20 < x11:
		return True
	elif x20 < x10 < x21:
		return True
	return False

def remove_overlap(surfaces):
	# if two the x ranges of two surfaces overlap, keep only the one above
	top_surfaces = []

	### TODO: FIX THE SURFACE LOOKS
	# # surfaces are of the form [[[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]], ...]
	# for a, o1 in enumerate(surfaces):
	# 	for b, o2 in enumerate(surfaces):
	# 		## o1, o2 are two lists of surfaces from different obj [[(x1, y1), (x2, y2)]...] and [[(xn, yn), (xn+1, yn+1)]]
	# 		for i, s1 in enumerate(o1):
	# 			bound_1, bound_2 = s1[0][0], s1[1][0]
	# 			for j, s2 in enumerate(o2):
	# 				## s1, s2 are two surfaces from different obj
	# 				# no-op if s1 is higher than s2
	# 				if s1[0][1] > s2[0][1]:
	# 					continue
	# 				#  ---   s2
	# 				# ---    s1
	# 				if s1[0][0] < s2[0][0] < s1[1][0] < s2[1][0]:
	# 					bound_2 = min(bound_2, s2[0][0])
	# 				#  ---   s2
	# 				#   ---  s1
	# 				elif s2[0][0] < s1[0][0] < s2[1][0] < s1[1][0]:
	# 					bound_1 = max(bound_1, s2[1][0])
	# 				#  ----   s2
	# 				#   --    s1
	# 				elif s2[0][0] < s1[0][0] < s1[1][0] < s2[1][0]:
	# 					bound_1, bound_2 = 1, -1
	# 					break
	# 			if bound_1 < bound_2:
	# 				top_surfaces.append([s1, (bound_1, bound_2)])

	for i, s1 in enumerate(surfaces):
		bound_1, bound_2 = s1[0][0], s1[1][0]

		# check if its x range overlaps with any other surfaces
		for j, s2 in enumerate(surfaces):
			if i != j:
				# no-op if candidate is higher than s2
				if s1[0][1] > s2[0][1]:
					continue
				#  ---   s2
				# ---    s1
				if s1[0][0] < s2[0][0] < s1[1][0] < s2[1][0]:
					bound_2 = min(bound_2, s2[0][0])
				#  ---   s2
				#   ---  s1
				elif s2[0][0] < s1[0][0] < s2[1][0] < s1[1][0]:
					bound_1 = max(bound_1, s2[1][0])
				#  ----   s2
				#   --    s1
				elif s2[0][0] < s1[0][0] < s1[1][0] < s2[1][0]:
					bound_1, bound_2 = 1, -1
					break
		if bound_1 < bound_2:
			top_surfaces.append([s1, (bound_1, bound_2)])

	result = []
	for s in top_surfaces:
		x1, y1, x2, y2, bound_1, bound_2 = s[0][0][0], s[0][0][1], s[0][1][0], s[0][1][1], s[1][0], s[1][1]
		result += get_points_on_segment(x1, y1, x2, y2, bound_1, bound_2)

	# result is of the form [{'x': x1, 'y': y1}, {'x': x2, 'y': y2},...]
	return result

def gaussian_noise(mean = 0, sd = 1):
	""" Apply gaussian noise """
	return np.random.normal(mean, sd)

def loss(prop, target, sd = 100):
	""" Gaussian loss """
	return -( (prop - target) / sd) ** 2

def flipy(c, y):
    """Small hack to convert chipmunk physics to pygame coordinates"""
    return -y+c['screen_size']['height']

def load_config(name):
	c = config.get_config()

	with open(name, 'rb') as f:
		ob = json.load(f)

		# PARAMETERS 
	c['drop_noise'] = ob['parameters']['drop_noise']
	c['collision_noise_mean'] = ob['parameters']['collision_noise_mean']
	c['collision_noise_sd'] = ob['parameters']['collision_noise_sd']
	# c['loss_sd_vision'] = ob['parameters']['loss_sd_vision']
	# c['loss_sd_sound'] = ob['parameters']['loss_sd_sound']
	# c['loss_penalty_sound'] = ob['parameters']['loss_penalty_sound']

	# GLOBAL SETTINGS 
	c['dt'] = ob['global']['timestep']
	c['substeps_per_frame'] = ob['global']['substeps']
	c['med'] = ob['global']['midpoint']
	c['gravity'] = ob['global']['gravity']
	c['screen_size'] = ob['global']['screen_size']
	c['hole_dropped_into'] = ob['global']['hole_dropped_into'] - 1
	# c['hole_dropped_into'] = ob['global']['hole_dropped_into']

	# PLINKO BOX SETTINGS 
	c['width'] = ob['box']['width']
	c['height'] = ob['box']['height']
	c['hole_width'] = ob['box']['holes']['width']
	c['hole_positions'] = ob['box']['holes']['positions']
	c['wall_elasticity'] = ob['box']['walls']['elasticity']
	c['wall_friction'] = ob['box']['walls']['friction']
	c['ground_elasticity'] = ob['box']['ground']['elasticity']
	c['ground_friction'] = ob['box']['ground']['friction']
	c['ground_y'] = ob['box']['ground']['position']['y']

	# BALL SETTINGS 
	c['ball_radius'] = ob['ball']['radius']
	c['ball_mass'] = ob['ball']['mass']
	c['ball_elasticity'] = ob['ball']['elasticity']
	c['ball_friction'] = ob['ball']['friction']

	# OBSTACLE SETTINGS
	c['obstacles'] = ob['obstacles']

	# BALL FINAL POSITION
	c['ball_final_position'] = ob['simulation'][c['hole_dropped_into']]['ball']['position'][-1]
	c['paths'] = [x['ball']['position'] for x in ob['simulation']]
	
	return c

def combine_video_and_audio(video, audio, filename = 'test_with_sound.mp4'):
	"""
	Function to combine video and audio.
	video = path to mp4 file 
	audio = path to wav file 
	filename = path to created file 
	"""
	import subprocess as sp
	sp.call('ffmpeg -i {video} -i {audio} -c:v copy -c:a aac -strict experimental {filename}'.format(video = video, audio = audio, filename = filename), shell=True)

def load_cache_eye_data_heatmap(world):
	"""
	Load the cached heatmap of
	"""
	z = np.loadtxt(open('../../data/cached_eye_data_heatmap_matrix/world_' + str(int(world)) + '.csv', 'rb'), delimiter=',', skiprows=0)
	return z


def save_eye_data_matrix(z, world):
	"""
	Save an eye data matrix to csv
	"""
	np.savetxt('../../data/cached_eye_data_heatmap_matrix/world_' + str(int(world)) + '.csv', z, delimiter=',')


def kde_scipy(vals1, vals2, r1, r2, N1, N2, w):
	# vals1, vals2 are the values of two variables (columns)
	# (a,b) interval for vals1; usually larger than (np.min(vals1), np.max(vals1))
	# (c,d) -"-          vals2
	(a, b) = r1
	(c, d) = r2
	x = np.linspace(a, b, N1)
	y = np.linspace(c, d, N2)
	X, Y = np.meshgrid(x, y)
	positions = np.vstack([Y.ravel(), X.ravel()])

	values = np.vstack([vals1, vals2])
	kernel = st.gaussian_kde(values, weights=w)

	Z = np.reshape(kernel(positions).T, X.shape)
	return [x, y, Z]


def get_heatmap(df, ybins=600, xbins=500):
	df = df[(df['x'] >= 0) & (df['x'] < 600) & (df['y'] >= 0) & (df['y'] < 500)].copy()
	weights = None
	if 'dur' in df:
		weights = df['dur']
	return kde_scipy(df['y'], df['x'], (0, 600), (0, 500), ybins, xbins, weights)


def get_average_zvalue(df, exps):
	# avg kde across all worlds for vision experiment
	# df_vision = df[(df['experiment']=='vision']
	# _, _, z_vision = get_heatmap(df_vision, ybins=60, xbins=50)
	z_vision = None

	# WARNINGS!! COMMENT OUT TO SAVE TIME
	z_vision_sound = None
	# avg kde across all worlds for vision_sound experiment
	df_combined = df[(df['experiment']=='vision_sound')]
	_, _, z_vision_sound = get_heatmap(df_combined, ybins=60, xbins=50)

	return z_vision, z_vision_sound


def get_adjusted_heatmap(df, z_vision, z_vision_sound, exp, ybins=600, xbins=500):
	x, y, z = get_heatmap(df, ybins, xbins)

	# subtract mean kde map from current world's kde, create mask: 0 - noise, 1 - non noise
	if exp == 'vision':
		z_adjusted = z - z_vision
	elif exp == 'vision_sound':
		z_adjusted = z - z_vision_sound
	mask = np.where(z_adjusted < 0, 0, 1)

	# filter fixation data using mask
	df = df[(df['x'] >= 0) & (df['x'] < 600) & (df['y'] >= 0) & (df['y'] < 500)].copy()
	df['mask'] = df.apply(lambda row: mask[int(row['y'] / 10)][int(row['x'] / 10)] > 0, axis=1)
	df = df[df['mask'] > 0]
	weights = None
	if 'dur' in df:
		weights = df['dur']
	xp, yp, zp = kde_scipy(df['y'], df['x'], (0, 600), (0, 500), 600, 500, weights)

	return xp, yp, zp, df

if __name__ == '__main__':
	main()