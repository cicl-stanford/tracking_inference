# import libraries 
import json
import os
import shutil
from random import randint
import pygame
import numpy as np
from numpy import array, cos, dot, pi, sin, argmin, absolute
from pygame.color import THECOLORS
from pygame.constants import QUIT, KEYDOWN, K_ESCAPE
import copy
from matplotlib import pyplot as plt
from KDEpy import FFTKDE
import PIL


# import files 
import config
import engine
import utils
from convert_coordinate import convertCoordinate

def main():
    c = config.get_config()
    # c = utils.load_config("data/json/world_1777.json")
    c = utils.load_config("data/json/world_6.json")
    sim_data = engine.run_simulation(c)
    visualize(c, sim_data)

def visualize(c, sim_data, save_images = False, make_video = False, video_name = 'test', eye_pos=None):
    """
    Visualize the world. 
    save_images: set True to save images 
    make_video: set True to make a video 
    """     
    
    # setup pygame
    screen = pygame.display.set_mode((c['screen_size']['width'], c['screen_size']['height']))

    # set up the rotated obstacles
    rotated = rotate_shapes(c)

    # make temporary directory for images 
    if save_images: 
        img_dir = 'images_temp'
        try:
            shutil.rmtree(img_dir)
        except:
            pass
        os.mkdir(img_dir)

    if len(sim_data['ball_position']) > 400:
        sim_data['ball_position'] = sim_data['ball_position'][:400]

    for t, frame in enumerate(sim_data['ball_position']):
        screen.fill(THECOLORS['white'])

        colors = [THECOLORS['blue'], THECOLORS['red'], THECOLORS['green']]

        # draw objects
        draw_obstacles(rotated, screen, colors)
        draw_ground(c, screen)
        if not (eye_pos is None):
            draw_eye(c, screen, eye_pos)
        draw_ball(c, screen, frame)
        draw_walls(c, screen)
        
        pygame.event.get()  # this is just to get pygame to update
        pygame.display.flip()
        # pygame.time.wait(1)

        if save_images:
            pygame.image.save(screen, os.path.join(img_dir, '%05d' % t + '.png'))
    
    if make_video:
        import subprocess as sp
        sp.call('ffmpeg -y -framerate 60 -i {ims}  -c:v libx264 -profile:v high -crf 10 -pix_fmt yuv420p {videoname}.mp4'.format(ims = img_dir+"/\'%05d.png\'", videoname = "data/videos/" + video_name), shell=True)
        shutil.rmtree(img_dir) #remove temporary directory

    running = True

    while running: 
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == KEYDOWN and e.key == K_ESCAPE:
                running = False

def snapshot(c, image_path, image_name, eye_pos=None, ball_pos=None, eye_data=None, unity_coordinates=False):
    """
    Create a snapshot of the world with the obstacles 
    """
    # setup pygame 
    # pygame.init()
    
    screen = pygame.Surface((c['screen_size']['width'],c['screen_size']['height']))
    screen.fill(THECOLORS['white'])

    colors = [THECOLORS['blue'], THECOLORS['red'], THECOLORS['green']]

    if unity_coordinates:
        draw_box_unity(c, screen)
        draw_obstacles_unity(c, screen, colors)
        if not (eye_pos is None):
            draw_eye(c, screen, eye_pos)
        if not (ball_pos is None):
            draw_ball(c, screen, ball_pos)
        if not (eye_data is None):
            draw_eye_data(c, screen, eye_data)

    # set up the rotated obstacles
    else:
        rotated = rotate_shapes(c)
        draw_obstacles(rotated, screen, colors)
        draw_ground(c, screen)
        draw_walls(c, screen)
        if not (ball_pos is None):
            draw_ball(c, screen, ball_pos)
        if not (eye_pos is None):
            draw_eye(c, screen, eye_pos)
        if not (eye_data is None):
            draw_eye_data(c, screen, eye_data)

    # save image
    pygame.image.save(screen, os.path.join(image_path, image_name + '.png'))
    
    # quit pygame 
    # pygame.quit()

##############
# HELPER FUNCTIONS 
##############

def rotate_shapes(c):
    # set up rotated shapes
    rotated = {name: [] for name in c['obstacles']}
    for shape in c['obstacles']:
        

        if 'shape' not in c['obstacles'][shape]:
            poly = utils.generate_ngon(c['obstacles'][shape]['n_sides'],
                                       c['obstacles'][shape]['size'])

        else:
            poly = c['obstacles'][shape]['shape']

        ob_center = array([c['obstacles'][shape]['position']['x'],
                           c['obstacles'][shape]['position']['y']])



        rot = c['obstacles'][shape]['rotation']
        for p in poly:
            rotmat = array([[cos(rot), -sin(rot)],
                            [sin(rot), cos(rot)]])


            rotp = dot(rotmat, p)

            rotp += ob_center
            rotp[1] = utils.flipy(c, rotp[1])

            rotated[shape].append(rotp.tolist())

    return rotated

def draw_ball(c, screen, frame):
    pygame.draw.circle(screen,
                           THECOLORS['black'],
                           (int(frame['x']), int(utils.flipy(c, frame['y']))),
                           c['ball_radius'])


def draw_box_unity(c, screen, adjust=False):



    # Set these manually
    # Pymunk center is 350,350
    # Width is 300 and height is 250
    # Y values are subtracted from 700 when switching to pygame
    pymunk_y_top = 100
    pymunk_y_ground = 600
    pymunk_x_left = 50
    pymunk_x_right = 650


    pymunk_hole1_left = c['old_hole_positions'][0] - c['hole_width']/2
    pymunk_hole1_right = c['old_hole_positions'][0] + c['hole_width']/2
    pymunk_hole2_left = c['old_hole_positions'][1] - c['hole_width']/2
    pymunk_hole2_right = c['old_hole_positions'][1] + c['hole_width']/2
    pymunk_hole3_left = c['old_hole_positions'][2] - c['hole_width']/2
    pymunk_hole3_right = c['old_hole_positions'][2] + c['hole_width']/2


    # draw ground
    unity_x_left, unity_y_ground = convertCoordinate(pymunk_x_left, pymunk_y_ground)
    unity_x_right, _ = convertCoordinate(pymunk_x_right, pymunk_y_ground)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_x_left, unity_y_ground),
        (unity_x_right, unity_y_ground))

    # draw left wall
    _, unity_y_top = convertCoordinate(pymunk_x_left, pymunk_y_top)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_x_left, unity_y_ground),
        (unity_x_left, unity_y_top))

    # draw right wall
    unity_x_right, _ = convertCoordinate(pymunk_x_right, pymunk_y_top)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_x_right, unity_y_ground),
        (unity_x_right, unity_y_top))


    # Top horizontal 1
    unity_hole1_left, _ = convertCoordinate(pymunk_hole1_left, pymunk_y_top)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_x_left, unity_y_top),
        (unity_hole1_left, unity_y_top))

    # Top horizontal 2
    unity_hole1_right, _ = convertCoordinate(pymunk_hole1_right, pymunk_y_top)
    unity_hole2_left, _ = convertCoordinate(pymunk_hole2_left, pymunk_y_top)


    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_hole1_right, unity_y_top),
        (unity_hole2_left, unity_y_top))

    # Top horizontal 3
    unity_hole2_right, _ = convertCoordinate(pymunk_hole2_right, pymunk_y_top)
    unity_hole3_left, _ = convertCoordinate(pymunk_hole3_left, pymunk_y_top)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_hole2_right, unity_y_top),
        (unity_hole3_left, unity_y_top))

    # Top horizontal 4
    unity_hole3_right, _ = convertCoordinate(pymunk_hole3_right, pymunk_y_top)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (unity_hole3_right, unity_y_top),
        (unity_x_right, unity_y_top))



def draw_obstacles_unity(c, screen, colors):

    for ob_dict in c['obstacles'].values():
        flipped_shape = [(x,utils.flipy(c,y)) for x,y in ob_dict['shape']]
        pygame.draw.polygon(screen, "black", flipped_shape)




def draw_walls(c, screen):
    # top horizontal: 1
        top_wall_y = utils.flipy(c,c['med'] + c['height']/2)
        
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['med'] - c['width']/2, top_wall_y),
                    (c['hole_positions'][0] - c['hole_width']/2, top_wall_y))
        
        # top horizontal: 2
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][0] + c['hole_width']/2, top_wall_y),
                    (c['hole_positions'][1] - c['hole_width']/2, top_wall_y))
        
        # top horizontal: 3
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][1] + c['hole_width']/2, top_wall_y),
                    (c['hole_positions'][2] - c['hole_width']/2, top_wall_y))
        
        # top horizontal: 4
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][2] + c['hole_width']/2, top_wall_y),
                    (c['med'] + c['width']/2, top_wall_y))

        # left vertical
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['med'] - c['width']/2, c['med'] - c['height']/2),
                    (c['med'] - c['width']/2, c['med'] + c['height']/2))

        # right vertical
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['med'] + c['width']/2, c['med'] - c['height']/2),
                    (c['med'] + c['width']/2, c['med'] + c['height']/2))



def draw_ground(c, screen):
    pygame.draw.line(screen,
                         THECOLORS['black'],
                         (c['med'] - c['width'] / 2, c['med'] + c['height']/2),
                         (c['med'] + c['width'] / 2, c['med'] + c['height']/2))

def draw_obstacles(rotated, screen, colors):
    for idx, shape in enumerate(rotated):
        pygame.draw.polygon(screen, "black", rotated[shape])


def draw_eye(c, screen, eye_pos):
    pygame.draw.circle(screen,
        THECOLORS['white'],
        (eye_pos[0], utils.flipy(c, eye_pos[1])),
        10,
        width=0)
    pygame.draw.circle(screen,
        THECOLORS['black'],
        (eye_pos[0], utils.flipy(c, eye_pos[1])),
        10,
        width=1)
    pygame.draw.circle(screen,
        THECOLORS['deepskyblue'],
        (eye_pos[0], utils.flipy(c, eye_pos[1])),
        6,
        width=0)
    pygame.draw.circle(screen,
        THECOLORS['black'],
        (eye_pos[0], utils.flipy(c, eye_pos[1])),
        3,
        width=0)

def draw_eye_data(c, screen, eye_data):
    n = eye_data.shape[0]

    for i in range(n):
        x, y = eye_data[i,:]

        pygame.draw.circle(screen,
            THECOLORS['darkorchid2'],
            # (x, utils.flipy(c, y)),
            (x, y),
            3,
            width=0)


########################
# Trial Transformation #
########################

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
    transformed_trial['screen_size'] = {"width": 600, "height": 500}
    
    return transformed_trial


def unity_transform_trial(trial, generate_shapes=True):
    trial_shapes = generate_trial_shapes(trial, generate_shapes=generate_shapes)
    transformed_trial = transform_trial(trial_shapes)
    
    return transformed_trial


############################
# Visualize Agent Behavior #
############################

def graph_conditional_dist(ax, hole, density_samples, multiplier=8000, offset=465, unity_background=False, kde_method="FFT"):
    
    color = ['red', 'blue', 'green']
    
    x, weights = zip(*density_samples[hole])
    # unity_x = [convertCoordinate(pt, 100)[0] for pt in x]
    x = np.array(x)
    weights = np.array(weights)

    # print(x)
    
    if unity_background:
        x_grid = np.linspace(10,585,600)
        # x = x - 50
        offset = 480
    else:
        # x_grid = np.linspace(90, 610, 520)
        x_grid = np.linspace(39,561,600)


    if kde_method == "FFT":
        kde = FFTKDE(kernel="gaussian", bw=20).fit(x, weights=weights)
        p = kde.evaluate(x_grid)*-multiplier
    elif kde_method == "scikit":
        kde = KernelDensity(kernel="gaussian", bandwidth=20).fit(x[:,np.newaxis], sample_weight=weights)
        p = np.exp(kde.score_samples(x_grid[:,np.newaxis]))*-multiplier
    
    p += offset
    
    # print(x_grid)

    if unity_background:
        p = np.insert(p, 0, offset)
        p = np.append(p, offset)
        x_grid = np.insert(x_grid,0,15)
        x_grid = np.append(x_grid,585)
    else:
        p = np.insert(p, 0, offset)
        p = np.append(p, offset)
        x_grid = np.insert(x_grid, 0, 39)
        x_grid = np.append(x_grid, 561)
    
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


    snapshot(trial, 
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

    ax.text(300,
            550,
            label,
            fontsize=16,
            verticalalignment="center",
            horizontalalignment="center")

    graph_conditional_dist(ax, 0, density_samples, kde_method=kde_method)
    graph_conditional_dist(ax, 1, density_samples, kde_method=kde_method)
    graph_conditional_dist(ax, 2, density_samples, kde_method=kde_method)

    draw_eye_plt(ax, eye_pos)

    ax.text(151, -30,
            "1",
            color="red",
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center")
    ax.text(300, -30,
            "2", 
            color="blue",
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center")
    ax.text(448, -30,
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
    eye_y = 500 - eye_pos[1]

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
        y.append(500 - pt['y'])
    
    start_pt = trajectory[0]
    start_x = start_pt['x']
    start_y = 500 - start_pt['y']
    
    col = color[hole]
    ax.plot(x, y, "--", color=col)
    circle1 = plt.Circle((start_x, start_y), 20, color=col)
    circle2 = plt.Circle((outcome['x'],500-outcome['y']), 20, color=col)
    
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    if not (eye_pos is None):
        ax = draw_eye_plt(ax, eye_pos)
    
    plt.savefig("visuals_agent/frames/{}.png".format(frame_name), 
                dpi=200)
    
    return ax
    

def visualize_trial(df_trial, trial_num, trial, viz_type="pdf", frame_rate=3, kde_method="FFT"):
    
    num_rows = df_trial.shape[0]
    
    trial = unity_transform_trial(trial)
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

if __name__ == '__main__':
    main()
