# Stacked Negative Correlation Learning for Deep Ensembling

This repository contains the code used for the experiments in the Master thesis "From Generalists to Specialists: Empowering Weak Learners in Deep Ensembles"

## Repository structure
Note: All experimental results are saved in the folder `/plots/` as .png and .svg with descriptive naming.

This repository combines code for multiple datasets and two different repositories. It is structured as follows:

- `SNCLtraining.py`: The main script to train ensembles with SNCL, GNCL or independently. We recommend to use Neptune.ai for detailed tracking of results. Trained models are saved in the folder `/models/`.
- `performTrainingTrials.sh`: Bash script to perform all training trials included in the thesis.
- `accuracyAndDiversityPlots.py`: Plots accuracy and diversity for the tested configurations. Note: This script requires a .csv export from neptune.ai to retrieve the detailed results.
- `removalExperiment.py`: Performs a pruning experiment to assess the robustness of specific ensembles.
- `qualitativeAnalysesForSpecialization.py`: Visualizes the weight matrices of meta-learners and class-specific accuracies of submodels and ensembles.
- `conformalPredictionWithMAPIE.py`: Performs conformal prediction with the library MAPIE to assess the in-domain uncertainty of previously trained ensemble models
- `generateAndPlotEntropies.py`: Generates the softmax entropies for the predictions of the previously trained ensemble models on the datasets CIFAR-10, CIFAR-10C and SVHN. The entropies are saved as `.npz` files in `/data/` for subsequent ood detection based on thresholding. Plot the entropy densities (estimated via gaussian KDE) per dataset and per model.
- `oodDetectionWithSoftmaxEntropies.py`: Performs ood detection based on the softmax entropies emitted by the trained ensembles, provides AUROC plots and saves results as `.csv` in `/data/`.
- `models.py`: Contains implementations of the ensemble model and meta-learner.
- `utils.py`: Contains utilities for dataset retrieval and pre-processing.
- `temperatureScaling.py`: Temperature scaling for stacked ensembles, adapted from https://github.com/gpleiss/temperature_scaling.

  

## Requirements

The following packages are required to run all of the provided code:
- numpy
- scipy
- PyTorch
- matplotlib
- MAPIE (https://github.com/scikit-learn-contrib/MAPIE/tree/master)
- sklearn
- neptune

## Running the experiments

We recommend to use Neptune.ai for detailed tracking of results. Please configure the api key and project name in `SNCLtraining.py`. We provide the bash script `performTrainingTrials.sh` to perform all training trials included in the thesis. Individual configurations can be performed through the command-line, e.g.: 

    python3 'SNCLtraining.py' --ncl_weight 0.5 --ncl_type GNCL --base_models 10 --neptuneai none --silent True

To reproduce the experimental results we suggest the following order of execution:

  1) `performTrainingTrials.sh`
  2) `accuracyAndDiversityPlots.py`
   
  Note: The subsequent scripts must be provided with run-ids of specific models trained in 1).

  3) `removalExperiment.py`
  4) `qualitativeAnalysesForSpecialization.py`
  5) `conformalPredictionWithMAPIE.py`
  6) `generateAndPlotEntropies.py`
  7) `oodDetectionWithSoftmaxEntropies.py`

All experimental results will be saved in the folder `/plots/` as .png and .svg with descriptive naming.
