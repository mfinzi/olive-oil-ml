[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/mfinzi/pristine-ml)](https://codeclimate.com/github/mfinzi/pristine-ml)
[![CodeClimate](http://img.shields.io/codeclimate/mfinzi/pristine-ml.svg?style=flat)](https://codeclimate.com/github/mfinzi/pristine-ml
"CodeClimate")

# Snake-Oil-ML

Nuts and bolts deep learning library to make training neural networks easier.
Features:
* Logging functionality not at set time intervals but as a percentage of the total training time
* Convenient specification for Random and Grid hyperparameter search: queued with a single GPU, split over multiple local GPUs, or over a Slurm Cluster
* Clean implementation of popular methods/problems in CV such as Vanilla Image Classification, Regression, the PiModel for Semi-Supervised learning, and SN-GAN; all through the common Trainer abstraction

# Installation
To install, run "pip install ." in the snake-oil-ml folder. Dependencies will be checked and installed from the setup.py file.

# Dependencies
* [Python 3.6+]
* [PyTorch](http://pytorch.org/) version 1.0.0
* [torchvision](https://github.com/pytorch/vision/)
* (optional) [tensorboardX](https://github.com/lanpa/tensorboardX)
* (optional) [tqdm]
