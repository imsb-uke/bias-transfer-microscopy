# Unsupervised bias transfer for medical images
Code for the paper "Deep learning-based bias transfer for overcoming laboratory differences of microscopic images".
Included is the implementation of color transfer, cycleGAN, U-Net cycleGAN, Fixed-Point GAN and all additional losses 
in Tensorflow 2. Training, validation and test images are defined via csv files (see data folder) and training runs
are tracked via sacred (+ mongoDB and omniboard). All requirements are defined in the docker_context folder, including 
a Dockerfile that can be used to set up Docker images and containers for execution.

## data
Contains an example for the required data structure.

## debiasmedimg
An installable python module containing all approaches (pip install --user ./debiasmedimg)

## docker_context
Contains a Dockerfile for setting up a docker environment including all package requirements.

## output
Output directory. Tensorflow checkpoints, generated images, and evaluation metrics get saved here.

## scripts
Scripts using sacred to track experiments. An example script for each approach is located here.
The subfolder 'configs' contains .yaml files which define the hyperparameters for the experiments.

