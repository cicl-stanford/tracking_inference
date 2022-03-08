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


    original_center_x = c['screen_size']['width']/2
    original_center_y = c['screen_size']['height']/2

    new_center_x, new_center_y = convertCoordinate(original_center_x, original_center_y)

    diff_x = original_center_x - new_center_x
    diff_y = original_center_y - new_center_y

    # Could modify this with the ground value
    topwall_y_pymunk = c['med'] + c['height']/2
    ground_y_pymunk = c['med'] - c['height']/2
    x_left_wall_pymunk = c['med'] - c['width']/2
    x_right_wall_pymunk = c['med'] + c['width']/2


    hole1_left_pymunk = c['hole_positions'][0] - c['hole_width']/2
    hole1_right_pymunk = c['hole_positions'][0] + c['hole_width']/2
    hole2_left_pymunk = c['hole_positions'][1] - c['hole_width']/2
    hole2_right_pymunk = c['hole_positions'][1] + c['hole_width']/2
    hole3_left_pymunk = c['hole_positions'][2] - c['hole_width']/2
    hole3_right_pymunk = c['hole_positions'][2] + c['hole_width']/2


    # draw ground
    x_left_unity, y_bottom_unity = convertCoordinate(x_left_wall_pymunk, ground_y_pymunk)
    x_right_unity, _ = convertCoordinate(x_right_wall_pymunk, ground_y_pymunk)

    x_left_unity += diff_x
    x_right_unity += diff_x
    y_bottom_unity += diff_y


    y_bottom_unity = utils.flipy(c, y_bottom_unity)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (x_left_unity, y_bottom_unity),
        (x_right_unity, y_bottom_unity))

    # draw left wall
    _, y_top_unity = convertCoordinate(x_left_wall_pymunk, topwall_y_pymunk)

    y_top_unity += diff_y


    y_top_unity = utils.flipy(c, y_top_unity)

    pygame.draw.line(screen,
        THECOLORS['black'],
        (x_left_unity, y_bottom_unity),
        (x_left_unity, y_top_unity))

    # draw right wall
    x_right_unity, _ = convertCoordinate(x_right_wall_pymunk, topwall_y_pymunk)

    x_right_unity += diff_x

    pygame.draw.line(screen,
        THECOLORS['black'],
        (x_right_unity, y_bottom_unity),
        (x_right_unity, y_top_unity))


    # Top horizontal 1
    hole1_left_unity, _ = convertCoordinate(hole1_left_pymunk, topwall_y_pymunk)

    hole1_left_unity += diff_x

    pygame.draw.line(screen,
        THECOLORS['black'],
        (x_left_unity, y_top_unity),
        (hole1_left_unity, y_top_unity))

    # Top horizontal 2
    hole1_right_unity, _ = convertCoordinate(hole1_right_pymunk, topwall_y_pymunk)
    hole2_left_unity, _ = convertCoordinate(hole2_left_pymunk, topwall_y_pymunk)

    hole1_right_unity += diff_x
    hole2_left_unity += diff_x

    pygame.draw.line(screen,
        THECOLORS['black'],
        (hole1_right_unity, y_top_unity),
        (hole2_left_unity, y_top_unity))

    # Top horizontal 3
    hole2_right_unity, _ = convertCoordinate(hole2_right_pymunk, topwall_y_pymunk)
    hole3_left_unity, _ = convertCoordinate(hole3_left_pymunk, topwall_y_pymunk)

    hole2_right_unity += diff_x
    hole3_left_unity += diff_x

    pygame.draw.line(screen,
        THECOLORS['black'],
        (hole2_right_unity, y_top_unity),
        (hole3_left_unity, y_top_unity))

    # Top horizontal 4
    hole3_right_unity, _ = convertCoordinate(hole3_right_pymunk, topwall_y_pymunk)

    hole3_right_unity += diff_x

    pygame.draw.line(screen,
        THECOLORS['black'],
        (hole3_right_unity, y_top_unity),
        (x_right_unity, y_top_unity))



def draw_obstacles_unity(c, screen, colors):

    for ob_dict in c['obstacles'].values():
        flipped_shape = [(x,utils.flipy(c,y)) for x,y in ob_dict['shape']]
        pygame.draw.polygon(screen, "black", flipped_shape)




def draw_walls(c, screen):
    # top horizontal: 1
        topwall_y = utils.flipy(c,c['med'] + c['height']/2)
        
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['med'] - c['width']/2, topwall_y),
                    (c['hole_positions'][0] - c['hole_width']/2, topwall_y))
        
        # top horizontal: 2
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][0] + c['hole_width']/2, topwall_y),
                    (c['hole_positions'][1] - c['hole_width']/2, topwall_y))
        
        # top horizontal: 3
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][1] + c['hole_width']/2, topwall_y),
                    (c['hole_positions'][2] - c['hole_width']/2, topwall_y))
        
        # top horizontal: 4
        pygame.draw.line(screen,
                    THECOLORS['black'],
                    (c['hole_positions'][2] + c['hole_width']/2, topwall_y),
                    (c['med'] + c['width']/2, topwall_y))

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


if __name__ == '__main__':
    main()
