[![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability-percentage/mfinzi/pristine-ml)](https://codeclimate.com/github/mfinzi/pristine-ml)
[![CodeClimate](http://img.shields.io/codeclimate/mfinzi/pristine-ml.svg?style=flat)](https://codeclimate.com/github/mfinzi/pristine-ml
"CodeClimate")

# Olive-Oil-ML

Nuts and bolts deep learning library to make training neural networks easier.
Features:
* Logging functionality not at set time intervals but as a percentage of the total training time
* Convenient specification for Random and Grid hyperparameter search: queued with a single GPU, split over multiple local GPUs, or over a Slurm Cluster
* Clean implementation of popular methods/problems in CV such as Vanilla Image Classification, Regression, the PiModel for Semi-Supervised learning, and SN-GAN; all through the common Trainer abstraction

# Installation
To install, run `pip install git+https://github.com/mfinzi/olive-oil-ml`. Dependencies will be checked and installed from the setup.py file.

# Dependencies
* Python 3.7+
* [PyTorch](http://pytorch.org/) 1.3.0+
* [torchvision](https://github.com/pytorch/vision/)
* [tqdm](https://tqdm.github.io/) 4.40+
* [natsort]()
* (optional) [tensorboardX](https://github.com/lanpa/tensorboardX)

# Jump into training a single model

To get a feel for the library, try training a (Classifier, Regressor, PiModel, GAN) model from our recipes.
For classification try running 
* `python oil/recipes/simpleTrial.py --dataset CIFAR100 --num_epochs 10`

Or, to train a conditional GAN model: 
* `python oil/recipes/simpleCGan.py --dataset SVHN --lr 2e-4`

Or train a PiModel semisupervised on CIFAR10 using only 1k labels: 
* `python oil/recipes/simplePi.py --dataset CIFAR10 --train 1000`


You can use `-h` to see the full range of arguments available. Command line arguments and defaults are automatically inferred
from the code used to construct the trial, so you can make a new trial (that uses some exotic data augmentation strategy for example) and the command line parser will generated for you, see the example recipes for how this works.

# Perform a hyperparameter search
Example: Search over hyperparameters for CNN classifier on Cifar100
* `python oil/recipes/exampleHyperSearch.py --dataset CIFAR100 --bs [50,32,64] --k [64,96] --num_epochs 100`
See example code for programmatic way of specifying the hyperparameter search.
Automatically parallelizes the search over multiple GPUs if available.
# Logging Support

# Interfacing with external libraries
