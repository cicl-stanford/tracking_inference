# plinko_tracking_public
This repo contains experiment, modeling, and analysis code for the Plinko eye-tracking project.

## code

### python

#### model
This section contains model code. 

1. `model.py` contains the primary model code, including an agent class which represents the world, simulates, and infers the most likely drop location of the ball. Code for both the sequential sampler (bandit) and uniform sampler (fixed_sampler) are included in this module.

2. `engine.py` contains the physics engine code which supports the agent's physical inference procedure. 

3. `run_model.py` is a script that can be used to generate behavior for a model at a given set of parameters.

4. `convert_coordinate.py` is a module for transforming pymunk coordinates to unity coordinates.

5. `regression_analysis.py` contains procedures for fitting regressions from features of model behavior to distributions of human eye-movement.

6. `compute_regression_error.py` is a script to compute the squared error between distributions of human eye-movment and model eye-movement distributions predicted with features of model behavior.

5. `evaluate_heatmaps.py` contains procedures to measure the earth-movers distance between two heatmaps representing distributions of fixations.

6. `compute_emd.py` contains code to run the heatmap evaluation for a given model file.

7. `visual.py` contains visualization tools for the model behavior.

8. `config.py` contains code for generating new trial stimuli.

9. `utils.py` contains additional utilities and procedures for the model.

10. `model_performance` contains records of model performance, including judgments and response times, fixations locations, and computed earth-movers distance scores. Subfolders pre-pended with "grid" contain model behavior pre-computed on Stanford's high-performance computing cluster and are included for ease of replicating paper results.