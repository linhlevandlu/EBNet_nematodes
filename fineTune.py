try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import lasagne
import sys
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet, TrainSplit
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss_2, write_file, test,EarlyStopping
from lasagne.layers import DenseLayer
import theano

# CONSTANT
#FMODEL = '/data3/linhlv/2018/saveModels/cnnmodel3_all_10000_epochs_.pickle'

def build_model1(nlayers,epochs, frozen=False):
	net3 = NeuralNet(
	layers=nlayers,

		# learning parameters
		update= lasagne.updates.nesterov_momentum,
		update_learning_rate=theano.shared(np.float32(0.01)),
		update_momentum=theano.shared(np.float32(0.9)),
		regression=True,
		on_epoch_finished = [
			AdjustVariable('update_learning_rate', start = 0.01, stop = 0.00001),
			AdjustVariable('update_momentum', start = 0.9, stop = 0.9999),
			EarlyStopping(1000)
		],
		max_epochs=epochs, # maximum iteration
		train_split = TrainSplit(eval_size=0.4),
		verbose=1,
	)
	if frozen:
		for layer in net3.layers[:frozenlayers]:
			layer.trainable = False
	return net3

def build_model():
	
	net = {}
	net['input'] = lasagne.layers.InputLayer((None,1,200,250))
	net['conv1'] = lasagne.layers.Conv2DLayer(net['input'], 32, (3,3),stride=3,pad=(41,16))
	net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1'],pool_size=(2,2))
	net['drop2'] = lasagne.layers.DropoutLayer(net['pool1'],p=0.1)
	net['conv2'] = lasagne.layers.Conv2DLayer(net['drop2'], 64, (2,2))
	net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2'],pool_size=(2,2))
	net['drop3'] = lasagne.layers.DropoutLayer(net['pool2'],p=0.2)
	net['conv3'] = lasagne.layers.Conv2DLayer(net['drop3'], 128, (2,2))
	net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3'],pool_size=(2,2))
	net['drop4'] = lasagne.layers.DropoutLayer(net['pool3'],p=0.3)
	net['hidden4'] = lasagne.layers.DenseLayer(net['drop4'],num_units=1000)
	net['drop5'] = lasagne.layers.DropoutLayer(net['hidden4'],p=0.5)
	net['hidden5'] = lasagne.layers.DenseLayer(net['drop5'],num_units=1000)
	net['output'] = lasagne.layers.DenseLayer(net['hidden5'],num_units=16,nonlinearity=None)
	return net

def set_weights(model_file,frozen=False,listLayers = []):
	with open(model_file) as f:
		model = pickle.load(f)
	print('Set the weights...')
	#newnet = model
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	output_layer = lasagne.layers.DenseLayer(net['hidden5'],num_units = 8, nonlinearity=None) # change num_units = 16 to 8 to adapt with 10 output
	
	if frozen:
		print('\nFreeze the parameters...\n')
		all_layers = lasagne.layers.get_all_layers(net)
		if len(all_layers) < len(listLayers):
			print('\nNumber of layers are less than number layers that you would like to freeze.\n')
		else:
			for flayer in listLayers:
				if flayer in all_layers:
					freeze_layer(net[flayer])

	all_layers = lasagne.layers.get_all_layers(net)
	for layer in all_layers:
		print('%s - %s' % (layer,net[layer].params))

	return output_layer	

def freeze_layer(layer):
	for param in layer.params.values():
		param.remove('trainable')

'''
def build_model2(modelfile):
	with open(modelfile) as f:
		model = pickle.load(f)
	print('Set the weights...')
	print(model)
	all_param = lasagne.layers.get_all_param_values(model.layers)
	net = build_model()
	lasagne.layers.set_all_param_values(net['output'],all_param,trainable=True)
	newlayers = lasagne.layers.DenseLayer(net['hidden5'],num_units = 16, nonlinearity=None)
	#model.layers = newlayers
	print(model)
	return model
'''
def fine_tune(fmodel,ftrain,epochs,ftest,savemodel,saveloss,savetest):
	X1,y1 = load2d(ftrain,test=False)
	listFrozens = [] #['conv1','conv2','conv3']
	newlayers = set_weights(fmodel,frozen=False,listLayers=listFrozens)
	net2 = build_model1(newlayers,epochs)
	net2.fit(X1,y1)

	
	sys.setrecursionlimit(1500000)
	with open(savemodel,'wb') as f:
		pickle.dump(net2,f,-1)
	draw_loss_2(net2,saveloss)
	test(net2,ftest,savetest)

