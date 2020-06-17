# PyTorch Re-implementation of "A Deep Learning Framework for Character Motion Synthesis and Editing"

## Prerequisites
1. Python 3.7
2. PyTorch
3. Numpy
4. Matplotlib
5. tqdm

To install dependencies:
1. For pip run `pip install -r requirements.txt`
2. (recommended) For conda run `conda env create -f environment.yml`

## Code Structure and Usage
All the code files are in "synth" folder. The "data" folder must be placed in 
the root directory of the project (on the same level as "synth"). You can install 
the "data" from [here.](http://theorangeduck.com/media/uploads/other_stuff/motionsynth_data.zip).

There are three types of code files:
1. Train - Used to train the network
2. Demo - Used to generate results
3. Show - Used for debugging or to generate results

### Train
The train files include:
- train_footstepper.py - Trains the Footstepper network
- train.py - Trains the Autoencoder (called Core)
- train_regression_kicking.py - Trains the Regressor network to generate kicking animation
- train_regression_punching.py - Trains the Regressor network to generate punching animation
- train_regression.py - Trains the Regressor network

The neural networks are defined in the "network.py" file.

### Demo
Some of the notable demo files are:
- demo_style_transfer.py - Generates style transfer results
- demo_crowd.py - Generates crowds results
- demo_regression.py - Generates regression results
- demo_kicking.py - Generates kicking results
- demo_punching.py - Generates punching results

The cost functions (mentioned in Eq 13, and Eq 14 in section 7.1, and 7.2 respectively) 
are defined in the "constraints.py" file. The Gram matrix calculation is defined
in "utils.py" file.

### Show
This includes the "show_weights.py" file that generates visualization of the
Convolution layers. 

The weights of the model are saved in "models" folder. The "motion" folder 
contains helper functions for generating the visualizations in Demo files.

I've included the weights for all the models. But these weights are not optimal 
because I did not train the networks until completion. I included them so that
the demo files can be run and tested.

I recommend training the networks until completion before doing any rigorous 
data collection or experiment.
