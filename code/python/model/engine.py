# import libraries 
from __future__ import division
import sys
from glob import glob
import numpy as np
from scipy.stats import truncnorm
import pymunk
from pymunk import Vec2d
import time

# import files
import utils
import visual
import config
import convert_coordinate

shape_code = {'walls' : 0,
			  'ground' : 1,
			  'ball' : 2,
			  'rectangle' : 3,
			  'triangle' : 4,
			  'pentagon' : 5}

inverse_shape_code = {shape_code[key] : key for key in shape_code}

TIMEOUT = 1

VELOCITY_SD = 0.0

def main():
	# c = config.get_config() # generate config 
	c = utils.load_config("data/cases/ground_truth/world_309.json")  # load a config file 
	# c = utils.load_config("data/json/world_6541.json")  # load a config file 
	c['hole_dropped_into'] = 1
	c['drop_noise'] = 0
	c['falling_noise'] = 0
	c['collision_noise_mean'] = 1
	c['collision_noise_sd'] = 0
	# for i in range(0,5):
	# print(c)
	data = run_simulation(c) 
	visual.visualize(c, data)
	# tmp = utils.loss(prop = 0, target = 50, sd = 2)
	# print("tmp", tmp)
	# pass
	# path_screenshot_write = 'data/images/pygame'
	# for world in range(1,201):
	# world = 6
	# for i in range(0,5):
	# 	c = utils.load_config("data/json/world_" + str(world) + ".json")
	# 	c['hole_dropped_into'] = 1
	# 	# c['drop_noise'] = 0
	# 	# c['collision_noise_mean'] = 1
	# 	# c['collision_noise_sd'] = 0
	# 	data = run_simulation(c) 
	# 	visual.visualize(c, data)
	# 	# visual.snapshot(c, image_path = path_screenshot_write + "/", image_name = 'world' + str(world)) #save snapshot

def run_simulation(c, noise_field=np.ones((500,600)), convert_coordinates=False, distorted=False):

	# print("Engine simulate function")
	# print()

	if convert_coordinates:

		original_center_x = c['screen_size']['width']/2
		original_center_y = c['screen_size']['height']/2
    
		new_center_x, new_center_y = convert_coordinate.convertCoordinate(original_center_x, original_center_y)
    
		# Transformation distance along x and y required to recenter unity coordinates after transformation
		DIFF_X = original_center_x - new_center_x
		DIFF_Y = original_center_y - new_center_y

	# PHYSICS PARAMETERS
	space = pymunk.Space()
	space.gravity = (0.0, c['gravity'])
	space.noise_field = noise_field
	space.convert_coordinates = convert_coordinates

	# noise applied to how the ball is dropped 
	ball_drop_noise(c, sd = c['drop_noise'])
	# print("sd = c['drop_noise']", c['drop_noise'])

	# CREATE OBJECTS 
	make_walls(c, space)
	top_surfaces = make_obstacles(c, space, distorted=distorted)
	ball = make_ball(c, space)
	make_ground(c, space)

	# CREATE COLLISION HANDLERS 
	h = space.add_wildcard_collision_handler(shape_code['ball'])
	# h.post_solve = record_collision #records each time step at which collision occurs
	h.begin = record_collision #records only the beginning of each collision
	h.post_solve = enable_velocity_noise
	h.data['collisions'] = []
	h.data['ball'] = ball

	# set ball to velocity 0 when it hits the ground
	g = space.add_collision_handler(shape_code['ground'], shape_code['ball'])
	g.post_solve = ground_collision
	g.data['running'] = True
	g.data['ball'] = ball

	# jitter velocity at end of every collision
	for ob in ['rectangle', 'triangle', 'pentagon']:
		ch = space.add_collision_handler(shape_code['ball'], shape_code[ob])
		ch.separate = jitter_velocity_collision
		ch.data['ball'] = ball
		# ch.data['collision_noise_record'] = []
		ch.data['collision_noise_sd'] = c['collision_noise_sd']
		ch.data['collision_noise_mean'] = c['collision_noise_mean']

	# add velocity noise at each step
	global VELOCITY_SD
	VELOCITY_SD = c['falling_noise']
	ball.velocity_func = jitter_velocity_falling

	###############
	## MAIN LOOP ##
	###############
	timestep = 0

	all_data = {}

	ball_pos = []
	ball_vel = []

	start = time.time()
	while g.data['running']:	# run into ground collision callback
		### Update physics

		for _ in range(c['substeps_per_frame']):
			# add gaussian noise to ball's velocity at each time step
			space.step(c['dt'] / c['substeps_per_frame'])

		timestep += 1

		# Convert to 3d scene coordinates when flag is set true
		x, y = ball.position.x, ball.position.y
		if convert_coordinates:
			x, y = convert_coordinate.convertCoordinate(x, y)
			x += DIFF_X
			y += DIFF_Y

		ball_pos.append({'x' : x,
						 'y' : y})
		ball_vel.append({'x' : ball.velocity.x,
						 'y' : ball.velocity.y})
		h.data['current_timestep'] = timestep

		if time.time() > start + TIMEOUT:
			# print('here')
			drop = {"pos": ball_pos[0], 'step': 0, 'sd': c['drop_noise'], 'angle': c['ball_initial_angle']}
			collisions = clean_collisions(collisions=h.data['collisions'])
			all_data['drop'] = drop
			all_data['collisions'] = collisions
			all_data['ball_position'] = ball_pos
			all_data['ball_velocity'] = ball_vel
			all_data['top_surfaces'] = top_surfaces
			# print("here")
			return all_data
	
	# clean up collisions
	# start_pos_x = c['hole_positions'][c['hole_dropped_into']]
	# start_pos_y = 600
	# start_ball_pos = {'x': start_pos_x, 'y': start_pos_y}
	drop = {"pos": ball_pos[0], 'step': 0, 'sd': c['drop_noise'], 'angle': c['ball_initial_angle']}
	# print("here")
	collisions = clean_collisions(collisions=h.data['collisions'])
	all_data['drop'] =  drop
	all_data['collisions'] = collisions
	# all_data['collision_noise_record'] = ch.data['collision_noise_record']
	all_data['ball_position'] = ball_pos
	all_data['ball_velocity'] = ball_vel
	all_data['top_surfaces'] = top_surfaces

	return all_data

def make_ball(c, space):
	inertia = pymunk.moment_for_circle(c['ball_mass'], 0, c['ball_radius'], (0,0))
	body = pymunk.Body(c['ball_mass'], inertia)
	x = c['hole_positions'][c['hole_dropped_into']]
	y = c['med'] + c['height'] / 2 + c['ball_radius']
	body.position = x, y
	shape = pymunk.Circle(body, c['ball_radius'], (0,0))
	shape.elasticity = c['ball_elasticity']
	shape.friction = c['ball_friction']

	shape.collision_type = shape_code['ball']

	space.add(body, shape)

	# used for setting initial velocity (should not be part of the ball definition)
	ang = c['ball_initial_angle']
	amp = 100
	off = 3 * np.pi / 2
	# so that clockwise is negative
	body.velocity = Vec2d(amp * -np.cos(ang + off), amp * np.sin(ang + off))

	return body

def make_ground(c, space):

	sz = (c['width'], 10)  # 4is for border

	body = pymunk.Body(body_type=pymunk.Body.STATIC)
	body.position = (c['med'], c['ground_y'])

	shape = pymunk.Poly.create_box(body, sz)
	shape.elasticity = 1
	shape.friction = 1

	shape.collision_type = shape_code['ground']
	space.add(body, shape)

def make_walls(c, space):

	walls = pymunk.Body(body_type=pymunk.Body.STATIC)
	
	topwall_y = c['med'] + c['height']/2
	
	static_lines = [
		# top horizontal: 1
		pymunk.Segment(walls,
					a = (c['med'] - c['width']/2, topwall_y),
					b = (c['hole_positions'][0] - c['hole_width']/2, topwall_y),
					radius = 2.0),
		# top horizontal: 2
		pymunk.Segment(walls,
					a = (c['hole_positions'][0] + c['hole_width']/2, topwall_y),
					b = (c['hole_positions'][1] - c['hole_width']/2, topwall_y),
					radius = 2.0),
		# top horizontal: 3
		pymunk.Segment(walls,
					a = (c['hole_positions'][1] + c['hole_width']/2, topwall_y),
					b = (c['hole_positions'][2] - c['hole_width']/2, topwall_y),
					radius = 2.0),
		# top horizontal: 4
		pymunk.Segment(walls,
					a = (c['hole_positions'][2] + c['hole_width']/2, topwall_y),
					b = (c['med'] + c['width']/2, topwall_y),
					radius = 2.0),

		# left vertical
		pymunk.Segment(walls,
					a = (c['med'] - c['width']/2, c['med'] - c['height']/2),
					b = (c['med'] - c['width']/2, c['med'] + c['height']/2),
					radius = 2.0),

		# right vertical
		pymunk.Segment(walls,
					a = (c['med'] + c['width']/2, c['med'] - c['height']/2),
					b = (c['med'] + c['width']/2, c['med'] + c['height']/2),
					radius = 2.0)]
	space.add(walls)
	for line in static_lines:
		line.elasticity = c['wall_elasticity']
		line.friction = c['wall_friction']
		line.collision_type = shape_code['walls']
		space.add(line)

def make_obstacles(c, space, distorted=False):
	# make obstacles in the space, and return a list of sample points on top surfaces of the obstacles
	# note that if two obstacles stack over each other, only the top surfaces of the higher one would be returned
	# the output would be used to generate topological looking in the attention heatmap
	top_surfaces = []
	for ob, ob_dict in c['obstacles'].items():

		rigid_body = pymunk.Body(body_type	= pymunk.Body.STATIC)
		if not distorted:
			polygon	= utils.generate_ngon(c['obstacles'][ob]['n_sides'], c['obstacles'][ob]['size'])
		else:
			polygon = ob_dict['shape']

		shape	= pymunk.Poly(rigid_body, polygon)
		shape.elasticity	= c['obstacles'][ob]['elasticity']
		shape.friction	= c['obstacles'][ob]['friction']

		pos = c['obstacles'][ob]['position']

		# add gaussian noises to x and y
		position_noise_x = utils.gaussian_noise(0, c['position_noise_sd'][ob]) 
		position_noise_y = utils.gaussian_noise(0, c['position_noise_sd'][ob]) 
		rigid_body.position = pos['x']+position_noise_x, pos['y']+position_noise_y
		# add gaussian noise to rotation
		position_noise_r = utils.gaussian_noise(0, c['rotation_noise_sd'][ob])
		rigid_body.angle = c['obstacles'][ob]['rotation']+position_noise_r

		shape.collision_type = shape_code[ob]	# key ob is the name

		space.add(shape, rigid_body)

		vertices = utils.get_vertices(shape)
		## TODO: FIX THE SURFACE LOOK
		top_surfaces += utils.get_top_surfaces(vertices, rigid_body.position)  # [[a1, a2], [b1, b2]]
	top_surfaces_non_overlap = utils.remove_overlap(top_surfaces)
	return top_surfaces_non_overlap


def generate_distorted_shape(ob_dict, eye_pos, perturb=10, divider=4, version=1):
	n = ob_dict['n_sides']
	side_length = ob_dict['size']
	rot = ob_dict['rotation']

	if version == 0:
		# Generates a shape with verticies randomly distributed around the circle
		# that circumscribes the polygon

		radius = utils.radius_reg_poly(side_length, n)

		points_with_angles = []

		for i in range(n):
			random_angle = np.random.rand()*2*np.pi
			random_radius = np.random.normal(radius, radius/divider)
			x = random_radius*np.cos(random_angle)
			y = random_radius*np.sin(random_angle)

			points_with_angles.append([(x,y),random_angle])

		sorted_points = sorted(points_with_angles, key=lambda x: x[1])

		return [pt[0] for pt in sorted_points]

	elif version == 1:
		# Perturb the verticies of the actual object
		actual_shape = utils.generate_ngon(n, side_length)
		# To do:
		# Evaluate the distance to each vertex in the shape.
		# Add noise to that vertex
		# Amount of noise is dependent on the distance to the point. Not sure of the best functional dependence
		# I'm imagining an upside down gaussian shape. It's lowest at the fovea (mean), and asymptotes in either
		# direction. This is analagous to what I've seen in the visual search literature, all though here I'm
		# reducing noise rather than adding signal. The variance of this gaussian will be a hyper-parameter. For
		# now can work with a gaussian that is symettric in 2d, though that's also not quite right.

		# std = 
		distorted_shape = [[coordinate + np.random.normal(scale=perturb) for coordinate in pt] for pt in actual_shape]


		return distorted_shape



### CALLBACKS

# records collisions between the ball and obstacles/walls
def record_collision(arbiter, space, data):
	ob1 = inverse_shape_code[int(arbiter.shapes[0].collision_type)]
	ob2 = inverse_shape_code[int(arbiter.shapes[1].collision_type)]
	# data['collisions'].append((data['current_timestep'], ob1, ob2))
	contact_point = arbiter.contact_point_set.points[0].point_b

	if space.convert_coordinates:
		x, y = convert_coordinate.convertCoordinate(contact_point.x, contact_point.y)
	else:
		x = contact_point.x
		y = contact_point.y

	data['collisions'].append({'objects': (ob1, ob2), 'step': data['current_timestep'],
							   'contact_point': {'x': x, 'y': y }})
	# disable collsion noise
	data['ball'].velocity_func = pymunk.Body.update_velocity
	return True

# records when the ball hits the ground 
def ground_collision(arbiter, space, data):
	data['ball'].velocity = Vec2d(0, 0)
	data['running'] = False
	return True

def jitter_velocity_collision(arbiter, space, data):

	# Grab the contact point from the last collision recorded in the collision handler h
	contact_point = space._handlers[2].data['collisions'][-1]['contact_point']
	# print(contact_point)
	# already converted to unity in record_collision
	if not space.convert_coordinates:
		unity_x, unity_y = convert_coordinate.convertCoordinate(contact_point['x'], contact_point['y'])
	else:
		unity_x = contact_point['x']
		unity_y = contact_point['y']
	row = int(np.round(unity_y))
	col = int(np.round(unity_x))

	noise_multiplier = space.noise_field[row, col]
	modified_sd = noise_multiplier*data['collision_noise_sd']

	# velocity_multiplier = utils.gaussian_noise(data['collision_noise_mean'], modified_sd) #potentially asymmetric noise
	mean = data['collision_noise_mean']
	sd = modified_sd

	if sd == 0:
		velocity_multiplier = mean
	else:
		lower_bound = (0 - mean)/sd
		velocity_multiplier = truncnorm.rvs(lower_bound, np.inf, mean, sd)

	# data['collision_noise_record'].append(velocity_multiplier)
	
	# change magnitude of velocity
	cur_vel = data['ball'].velocity
	new_vel = Vec2d(cur_vel.x * velocity_multiplier, cur_vel.y * velocity_multiplier)
	data['ball'].velocity = new_vel

def clean_collisions(collisions): 
	"""
	Interactions with shapes sometimes result in multiple collisions, but we only want to keep the first collision. 
	"""	
	idx = 1 
	while idx < len(collisions): 
		if collisions[idx-1]['objects'] == collisions[idx]['objects']:
			del collisions[idx]
		else: 
			idx += 1
			
	return(collisions)

def ball_drop_noise(c, sd):
	""" Noise applied to the angle in which the ball is dropped. """
	c['ball_initial_angle'] = utils.gaussian_noise(0, sd)

def ball_collision_noise(c, mean, sd):
	"""
	Noise applied to the ball's velocity after collisions.
	A value of 1 means no noise.
	"""
	c['collision_noise_mean'] = mean
	c['collision_noise_sd'] = sd

def disable_velocity_noise(arbiter, space, data):
	data['ball'].velocity_func = pymunk.Body.update_velocity
	return True

def enable_velocity_noise(arbiter, space, data):
	data['ball'].velocity_func = jitter_velocity_falling

def jitter_velocity_falling(body, gravity, damping, dt):
	pymunk.Body.update_velocity(body, gravity, damping, dt)
	ang = utils.gaussian_noise(0, VELOCITY_SD)
	cur_vel = body.velocity

	new_x = cur_vel.x * np.cos(ang) - cur_vel.y * np.sin(ang)
	new_y = min(0.0, cur_vel.x * np.sin(ang) + cur_vel.y * np.cos(ang))

	new_vel = Vec2d(new_x, new_y)
	body.velocity = new_vel


if __name__ == '__main__':
	main()
