# plinko_tracking_public
This repo contains experiment, modeling, and analysis code for the Plinko eye-tracking project.

## code

### python

#### model
This section contains model code. 

1. `bandit_observation_model.py` contains the primary model code, including an agent class which represents the world, simulates, looks around the scene and infers the most likely drop location of the ball. Code for both the bandit model and uniform sampler are included in this module.

2. `engine.py` contains the physics engine code which supports the agent's physical inference procedure. 

3. `run_model.py` is a script that can be used to generate behavior for a model at a given set of parameters.

4. `convert_coordinate.py` is a self-contained module for transforming pymunk coordinates to unity coordinates.

5. `evaluate_heatmaps.py` contains procedures to measure the earth-movers distance between two heatmaps representing distributions of fixations.

6. `compute_emd.py` contains code to run the heatmap evaluation for a given model file.

7. `visual.py` contains visualization tools for the model behavior.

8. `config.py` contains code for generating new trial stimuli.

9. `utils.py` contains additional utilities and procedures for the model.

10. `model_performance` contains records of model performance, including judgments and response times, fixations locations, and computed earth-movers distance scores. Subfolders pre-pended with "grid" contain model behavior pre-computed on Stanford's high-performance computing cluster and are included ease of replicating paper results.