from psychopy import visual, event
import pygaze
from pygaze import libscreen, eyetracker, libtime
import time
import numpy as np
import datetime
import os
import json
import subprocess

ttype = 'dummy'
# ttype = 'eyelink'
vid_length = 5000
testing = False
if testing:
	break_period = 3
else:
	break_period = 30
show_movies = True
col_demo = True


# Procedure to collect demographic data
def collect_info():
	# Five demographic categories
	categories = ['first name', 'gender', 'age', 'race', 'ethnicity (Hispanic or non-Hispanic)']
	participant = {}
	for cat in categories:
		inp = raw_input('Please type your ' + cat + ':\n')
		print
		# special handling for ethnicity
		if cat == 'ethnicity (Hispanic or non-Hispanic)':
			participant['ethnicity'] = inp
		elif cat == 'first name':
			participant['fname'] = inp
		else:
			participant[cat] = inp
	# Add current date. Construct proper format
	date = datetime.datetime.now()
	participant['date'] = (str(date.month) + '/' + str(date.day) + '/' + str(date.year))
	return participant

def wait_loop(klist):
	event.clearEvents()
	keep_going = True
	while keep_going:
		keys = event.getKeys(keyList=klist)
		if len(keys) != 0:
			keep_going = False

	return keys

# A procedure to write text to the display
def show_text(disp, win, text, cont_inst=True, klist=['space']):
	if cont_inst:
		text = text + '\n\n(press space to continue)'

	inst = visual.TextStim(win, text=text, units='norm', height=0.1, wrapWidth=1.5)
	inst_sc = libscreen.Screen(disptype='psychopy')
	inst_sc.screen.append(inst)
	# if cont_inst:
	# 	cont = visiual.TextStim(win)
	disp.fill(inst_sc)
	inst.draw()
	disp.show()

	keys = wait_loop(klist)

	return keys[0]

# # A procedure to show arbitrary text
# def show_text(disp, words, cont_inst=True, klist=['space']):
# 	# By default the procedure provides instructions for how to continue
# 	if cont_inst:
# 		words = words + '\n\n(press the space bar to continue)'
# 	# Instantiate the display, draw the text, add to the display and show
# 	inst = libscreen.Screen(disptype='psychopy')
# 	inst.draw_text(words, fontsize=fsize)
# 	disp.fill(screen=inst)
# 	disp.show()

# 	# Show until the subject hits space
# 	keep_going = True
# 	while keep_going:
# 		keys = event.getKeys(keyList=klist)
# 		if keys:
# 			keep_going = False

# 	# Return the key value (generally unused)
# 	return keys[0]

# A procedure to run the movie at a particular path
def show_movie(disp, win, path, proc):
	# set up the screen
	# mov = visual.MovieStim3(win, path, units='norm', size=(1, 1))
	mov = visual.MovieStim3(win, path, units='pix', size=(600,500))
	mov_sc = libscreen.Screen(disptype='psychopy')
	# if proc:
	# 	proceed = visual.TextStim(win, text='Press space to view the next clip', units='norm', pos=(0,0), height=0.05)
	#    	mov_sc.screen.append(proceed)
	mov_sc.screen.append(mov)


	disp.fill(screen=mov_sc)
	disp.show()

	while mov.status != visual.FINISHED:
		mov.draw()
		win.flip()

	if proc:
		world_num = path[16:-11]
		last_path = '../images/last_frames/last_frame_' + world_num + '.png'
		# last_frame = visual.ImageStim(win, last_path, units='norm', size=(1, 1))
		last_frame = visual.ImageStim(win, last_path, units='pix', size=(600,500))
		proc_text = visual.TextStim(win, text='Press space to proceed', units='pix', pos=(0,-275), height=30)
		proc_sc = libscreen.Screen(disptype='psychopy')
		proc_sc.screen.append(last_frame)
		proc_sc.screen.append(proc_text)
		disp.fill(proc_sc)
		disp.show()

		wait_loop(['space'])

	return




# A procedure to run the instructions for the experiment
def show_instruct(disp, win, sound):

	inst1 = 'In this experiment, we will ask you to make judgments about physical interactions. These interactions take place in a plinko box, a box with three holes on the top and obstacles inside. We drop a marble into one of the holes and it falls to the bottom of the plinko box bouncing off any obstacles in the way. Here are some examples to familiarize you with how the plinko box works:'

	show_text(disp, win, inst1)
	video_loc = '../videos/'
	vids = os.listdir(video_loc)
	vids = [v for v in vids if v[-4:] == '.mp4']

	if show_movies:
		for vid in vids:
			show_movie(disp, win, video_loc + vid, False)
			time.sleep(3)
			show_movie(disp, win, video_loc + vid, True)

	inst2 = "Now that you have a sense for how the world works, we can run some trials. In each trial we will show you a still image of the plinko box after we have dropped in the ball. It's your job to determine which of the three holes the ball came from. You will answer on the keypad using the number keys 1, 2, and 3. Hole 1 is the farthest to your left, 2 is the middle, and 3 is on your right. Let's get started with calibration and then proceed to the experiment."

	show_text(disp, win, inst2)
	# show_movie(disp, win, video_loc + vids[len(vids) - 1])
	# time.sleep(3)


def disp_im(disp, win, tracker, image):

	# setup the image stimulus
	# inst = visual.TextStim(win, text="In which hole was the ball dropped?", units='norm', pos=(0, 0.8), height=0.05)
	num1 = visual.TextStim(win, text="1", units='pix', pos=(-148,270), height=40)
	num2 = visual.TextStim(win, text="2", units='pix', pos=(0, 270), height=40)
	num3 = visual.TextStim(win, text="3", units='pix', pos=(148,270), height=40)
	# im = visual.ImageStim(win, image, units='norm', size=(1.5, (5/3.0)))
	im = visual.ImageStim(win, image, units='pix', size=(600, 500))

	tr_sc = libscreen.Screen(disptype='psychopy')
	# tr_sc.screen.append(inst)
	tr_sc.screen.append(num1)
	tr_sc.screen.append(num2)
	tr_sc.screen.append(num3)
	tr_sc.screen.append(im)

	# x and y coordinates, time list, and pupil size list
	x_list = []
	y_list = []
	t_list = []
	p_list = []

	# get time


	# begin tracking
	tracker.start_recording()

	# get first time
	t1 = 0

	# sample and record
	event.clearEvents()
	keys = []

	libtime.expstart()
	# Place the screen in the display
	disp.fill(screen=tr_sc)
	disp.show()

	while len(keys) == 0:
		# if at least one millisecond has passed sample (1000 samples per second)
		t2 = libtime.get_time()
		if t2 - t1 > 1.0:
			x,y = tracker.sample()
			p = tracker.pupil_size()

			x_list.append(x)
			y_list.append(y)
			t_list.append(t2)
			p_list.append(p)

			# push up the clock
			t1 = t2

		keys = event.getKeys(['1', '2', '3'])

	# stop tracking
	tracker.stop_recording()

	# return the eyedata
	return keys[0], {'x': x_list, 'y': y_list, 't': t_list, 'p': p_list}


def get_judgment(disp, win):
	inst = 'Please indicate which hole you think the ball entered from. 1 is the left, 2 is the middle, and 3 is the right.'

	return show_text(disp, win, inst, cont_inst=False, klist=['1', '2', '3'])


def run_trial(disp, win, tracker, image):

	# Check participant's fixation
	fixsc = libscreen.Screen(disptype='psychopy')
	fixsc.draw_fixation()
	disp.fill(fixsc)
	disp.show()

	# If fixation is off, drift correct
	checked = False
	while not checked:
		checked = tracker.drift_correction()

	# Run data collection
	judgment, eye_info = disp_im(disp, win, tracker, image)

	# judgment = get_judgment(disp, win)

	return judgment, eye_info

# A procedure to extract the world number from the image filename
def world_num(wstr): 
	try:
		return int(wstr[22:-4])
	except:
		raise Exception('Improper string format for file name ' + wstr)

# A procedure to get the next participant's number
def part_num():
	files = os.listdir('Output')
	files = [f for f in files if 'json' in f]
	files = [int(name[3:-5]) for name in files]
	files.sort()
	if len(files) != 0:
		return files[len(files) - 1] + 1
	else:
		return 1


# A procedure to run the full experiment
def run_experiment():

	if col_demo:
		# part = int(raw_input('Please provide participant number.\n'))
		part = part_num()
		# print
		demo = collect_info()
	else:
		part = part_num()
		demo = {}


	# Collect the stimuli names
	prac_path = '../images/practice/'
	practice = [prac_path + pic for pic in os.listdir(prac_path)]
	trial_path = '../images/trial/'
	trials = [trial_path + pic for pic in os.listdir(trial_path)]

	# shuffle the trials
	np.random.shuffle(trials)
	if testing:
		trials = trials[:9]

	# initialize the display, window, and tracker
	display = libscreen.Display(disptype='psychopy')
	window = pygaze.expdisplay
	tracker = eyetracker.EyeTracker(display, trackertype=ttype)

	# run the instructions
	show_instruct(display, window, False)

	# calibrate
	# window.winHandle.minimize()
	tracker.calibrate()
	# tracker.runSetupProcedure()
	# window.winHandle.activate()
	# window.winHandle.maximize()


	# run the practice trials
	for stim in practice:
		run_trial(display, window, tracker, stim)


	trial_data = []

	break_text = "Time for a quick break. Feel free to blink your eyes and move about. We'll come back and recalibrate when you are ready!"

	# run the actual trials, breaking after every fiftieth picture
	for i in range(len(trials)):
		tr = trials[i]
		judgment, eye_info = run_trial(display, window, tracker, tr)
		trial_data.append({'trial': world_num(tr), 'judgment': judgment, 'eye_data': eye_info})

		if i + 1 != len(trials) and (i + 1) % break_period == 0:
			show_text(display, window, break_text)
			tracker.calibrate()

	show_text(display, window, "That's all. Thanks for your participation!")
	tracker.close()
	display.close()

	exp_record = {'participant': part, 'demographics': demo, 'trials': trial_data}

	j = json.JSONEncoder()
	out = j.encode(exp_record)

	with open('Output/out' + str(part) + '.json', 'w+') as f:
		f.write(out)

	if ttype == 'eyelink':
		os.system('mv default.edf Output/out' + str(part) + '.edf')

	return out





# display = libscreen.Display(disptype='psychopy')
# window = pygaze.expdisplay
# tracker = eyetracker.EyeTracker(display, trackertype=ttype)
# run_trial(display, window, tracker, '../images/trial/world_1.png')
# show_instruct(display, window, True)
# show_text(display, window, inst1)

run_experiment()

# judge, info = run_trial(display, window, tracker, '../images/trial/world_1.png')
# print len(info['x'])

