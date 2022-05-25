import numpy as np
import pandas as pd
import evaluate_heatmaps as e
from sklearn.linear_model import LinearRegression
import pickle
import time
import os


def load_model_perf(model_version):
    with open(model_version, "rb") as f:
        model_perf = pickle.load(f)
        
    return model_perf


def make_events_heatmaps(trial_events, model_type):
    

    if model_type == "bandit":
        
        drop_pts = []
        col_obs_pts = []
        col_wall_pts = []
        col_ground_pts = []
        
        for run in trial_events:
            drop_pts.extend(run['drop'])
            col_obs_pts.extend(run['col_obs'])
            col_wall_pts.extend(run['col_wall'])
            col_ground_pts.extend(run['col_ground'])
            
    elif model_type == "fixed_sample":
        
        drop_pts = trial_events['drop']
        col_obs_pts = trial_events['col_obs']
        col_wall_pts = trial_events['col_wall']
        col_ground_pts = trial_events['col_ground']
        
        
    else:
        raise Exception("Model {} not implemented.".format(model_type))
            
    drop_hist = e.make_kde(np.array(drop_pts), 1) if len(drop_pts) != 0 else np.zeros((501, 601))
    obs_hist = e.make_kde(np.array(col_obs_pts), 1) if len(col_obs_pts) != 0 else np.zeros((501, 601))
    wall_hist = e.make_kde(np.array(col_wall_pts), 1) if len(col_wall_pts) != 0 else np.zeros((501, 601))
    ground_hist = e.make_kde(np.array(col_ground_pts), 1) if len(col_ground_pts) != 0 else np.zeros((501, 601))
    
    return drop_hist, obs_hist, wall_hist, ground_hist


def load_human_heatmap(tr_num, split="test"):
    with open("heatmaps/human_trial_{}_{}.pickle".format(split, tr_num), "rb") as f:
        human_hm = pickle.load(f)
        
    return human_hm

def load_heatmaps(tr_num):
    
    with open("heatmaps/obs_trial_{}.pickle".format(tr_num), "rb") as f:
        obs_hm = pickle.load(f)
        
    with open("heatmaps/ball_trial_{}.pickle".format(tr_num), "rb") as f:
        ball_hm = pickle.load(f)
        
    with open("heatmaps/holes.pickle", "rb") as f:
        hole_hm = pickle.load(f)
        
    with open("heatmaps/center.pickle", "rb") as f:
        center_hm = pickle.load(f)
        
    human_hm = load_human_heatmap(tr_num)
        
    return obs_hm, ball_hm, hole_hm, center_hm, human_hm


def setup_trial_regression(tr_num,
                           tr_events,
                           model_type):
    
    obs_hm, ball_hm, hole_hm, center_hm, human_hm = load_heatmaps(tr_num)


    if model_type == "bandit" or model_type == "fixed_sample":

        drop_hm, dyn_obs_hm, wall_hm, ground_hm = make_events_heatmaps(tr_events, model_type)

        features = [obs_hm,
                    ball_hm,
                    hole_hm,
                    center_hm,
                    drop_hm,
                    dyn_obs_hm,
                    wall_hm,
                    ground_hm]


    elif model_type == "visual_features":

        features = [obs_hm, ball_hm, hole_hm, center_hm]

    else:

        raise Exception("Model type {} not implemented.".format(model_type))
    
    features = np.array([np.ravel(arr) for arr in features]).T
    part_vec = np.ravel(human_hm)
    
    return features, part_vec


world_files = os.listdir("../../../figures/images/png/final/")
world_nums = sorted([int(file[6:-4]) for file in world_files])


def setup_model_regression(model_type, model_events=None):

    if model_type == "visual_features":
        model_events = zip(world_nums, [{}]*len(world_nums))
    
    feature_list = []
    label_list = []
    
    for tr_num, tr_events in model_events:
        
        features, part_vec = setup_trial_regression(tr_num,
                                                   tr_events,
                                                   model_type)
        
        feature_list.append(features)
        label_list.append(part_vec)
        
    return np.concatenate(feature_list), np.concatenate(label_list)


def compute_regression(model_events, model_type):
	model_features, labels = setup_model_regression(model_type, model_events)
	reg = LinearRegression().fit(model_features, labels)
	model_pred = reg.predict(model_features)

	return reg, model_pred, model_features, labels


def evaluate_model(model_events, model_type):

	_, model_pred, _, labels = compute_regression(model_events, model_type)

	return np.sum((model_pred - labels)**2)



