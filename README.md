# cms_hh_tf_inference
Plugin to run ensembles of TensorFlow models on incoming data in CMSSW

# Installation

## For inference

1. cmsrel CMSSW_10_2_16
1. cd CMSSW_10_2_16/src
1. cmsenv
1. git clone git@github.com:GilesStrong/cms_hh_tf_inference.git
1. scram b -j 12

## For testing

1. conda env create -f environment.yml
1. conda activate tf_inference

# Testing

The file `testing_setup/python/Model_training_and_export.ipynb` will train and export 5 PyTorch models to TensorFlow buffers and generate some training data with the expected predictions, plus files necessary to preprocess the inputs.
Building the tool in CMSSW will generate an executable in `CMSSW_10_2_16/test/slc7_amd64_gcc700/` called `testloop` which will run the example data through the TensorFlow models and verify that the predictions match the PyTorch ones to a tolerance of 1e-5.

# Usage

The main interface is the `InfWrapper` class. This is instantiated using the absolute path to a directory, which I will distribute during normal usage but an example one will be created by `Model_training_and_export.ipynb`. An `InfWrapper` contains two `Pipeline` classes, one for each ensemble. `InfWrapper.predict` takes a vector of floats (the unpreprocessed input vector in the expected order)and an unsigned long int (the event ID). The event ID is used to determine which pipeline is used to predict the event.

The `Pipeline` class consists of: A `Preproc` class, which loads a set of means and standard deviations from a text file to correctly preprocess the input features and convert the to a Tensor; And an `Ensemble` class.

The `Ensemble` class contains a several TensorFlow models and a set of weights to modify how important each model is in the final prediction. `Ensemble.predict` takes the input Tensor and passes it through each model in turn, and computes the weighted mean of predictions.

`inference/test/test.cc` provides an example of instantiating an `InfWrapper` and passing example data through it.

## Important!

The input vector to `InfWrapper.predict` should not contain any default 'placeholder' values like -999.00. These should be replaced with `std:nanf`. `Preproc.process` will later replace them with zeros after preprocessing non-NaN values.

# Notes

`.pb` files must be created with the same version of TensorFlow that will be later used to read them: 1.6 for CMSSW_10_2+X, 1.13.1 for 10_6_X. Package requirements are configured to install TF 1.6.
