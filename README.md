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

7. `evaluate_heatmaps.py` contains procedures to measure the earth-movers distance between two heatmaps representing distributions of fixations.

8. `compute_emd.py` contains code to run the heatmap evaluation for a given model file.

9. `visual.py` contains visualization tools for the model behavior.

10. `config.py` contains code for generating new trial stimuli.

11. `utils.py` contains additional utilities and procedures for the model.

12. `model_performance` contains records of model performance, including judgments and response times, fixations locations, and computed earth-movers distance scores. Subfolders pre-pended with "grid" contain model behavior pre-computed on Stanford's high-performance computing cluster and are included for ease of replicating paper results.

13. `heatmaps` contains precomputed kernel density estimates for regression analysis. Histograms are pre-computed for all physical features of all trials (obstacles, holes, ball location, center), as well as human eye-gaze distributions computed for a train set (half participants) and test set (all participants).


### R

This section contains analysis and visualization code to produce model figures.

1. `analysis.Rmd` performs the grid search on the pre-computed model performance files. It then loads the top performing models and plots model results for the three different data signals.

2. `figures` folder containing base figures for results presentation in the paper.


### experiment

#### exp_code

1. `plinko_eyetracking.py` code to run the experiment. By default tracker is set to "dummy" mode, but can be connected to an actual eye-tracker by changing the ttype parameter at the top of the file.

2. `Output` folder where experiment output is saved as json file (empty).


#### images

Images for presenting stimuli, pratice trials, and training stills.

#### videos

Pratice videos


## data

### human_data

Contains compressed human data for the full experiment. The data is a pickled pandas dataframe compressed in xz format.

### stimuli

#### ground_truth

Contains json files representing the 150 trial stimuli. This files can be used by the physics engine to represent and simulate in the world according to the given conditions.

#### practice

Contains json files representing the 2 practice stimuli.


## figures

### cogsci_2022

Paper figures.

### images

Still images of the trial stimuli.


# Replicating paper results

## Using pre-computed model-performance

## From scratch

### Generate Model Behavior

Model behavior for either the sequential sampler or the uniform sampler can be generated using the `run_model.py` script in `code/python/model/`.

The sequential sampler has four parameters: a decision threshold, reward-uncertainty tradeoff, kernel density bandwidth, and sample weight.

To generate model behavior for the sequential sampler at a given parameter setting navigate to the folder `code/python/model` and run the following:

```
python run_model.py bandit <seed> <decision_threshold> <tradeoff> <bandwidth> <sample_weight>
```

The script will generate a csv recording judgments and number of collisions for 30 runs on each trial in the folder `model_performance/judgment_rt`. The model will also generate a pickle file recording all the physical events from all the simulations in the folder `model_performance/collisions`.

The uniform sampler has two parameters: the number of samples and the bandwidth of the kernel density.

You can generate model behavior for the uniform sampler in an analogous way:

```
python run_model.py fixed_sample <seed> <num_samples> <bandwidth>
```


