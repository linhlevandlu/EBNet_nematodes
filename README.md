# EBNet_nematodes
This repository contains the program to train EB-Net on nematode images.
The structure of EB-Net is kept the same as description in orginal work. 
But the input and the output of the model have been modified to adapt with the new data.
We have applied the cross-validation to train the model from scratch in order to obtain the
predicted coordinates for all images.
We have also used fine-tuning technique to improve the results by using the trained parameters 
which have been obtained with beetle's images.

# Requirements:
	- Python (>= 2.7)
	- Theano (1.0.1)
	- Lasagne (0.2.dev1)
	- Cuda version 10.2

# Files and their functions:
The function of files in this repository are described in following:
	- readCSV.py: load and normalize data. Normally, the data (link to the images and coordinates of landmarks) is stored in csv files.
	- utils.py: implements the util functions such as drawing the losses, writing the file, or drawing the results, etc.
	- model.py: defines the structure of the network and a method to train the network.
	- fineTune.py: contains the functions to load the trained model and to fine-tune it.
	- runTraining.py: this is the main file to train the model from scratch with cross-validation
	- runFineTuning.py: to fine-tune the trained model in different folds of data (cross-validation)
	- runTest.py: runs to predict the landmarks on the images of the test set.

# To train the model:
	1. In the runTraining.py file:
		a. Change the path to the training/testing data (csv file)
		b. Change the path to the output folder (where we store the checkpoint, images, etc)
	2. Modify the model (model.py) if necessary.	
	3. Open the terminal and run the command: python runTraining.py

# To fine-tune the model
Beside training from scratch, we provide also the simple program in able to fine-tune a trained model on nemtode images.
All the program are in fineTune\*.py. In order to use this, we need to:
	1. Change the path the trained model file (checkpoint)
	2. Change the path to training data
	3. Change the path to ouput folder (where we would like to store the outputs)
	4. Run command: python runFineTuning.py on terminal 

# Bibtex
If you want to cite this work, feel free to use this:

@inproceedings{le2018landmarks,
  title={Landmarks detection by applying Deep networks},
  author={Le, Van-Linh and Beurton-Aimar, Marie and Zemmari, Akka and Parisey, Nicolas},
  booktitle={2018 1st International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  pages={1--6},
  year={2018},
  organization={IEEE}
}

