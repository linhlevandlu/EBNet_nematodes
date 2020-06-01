try:
	import cPickle as pickle
except ImportError:
	import pickle
import os
import sys
import numpy as np
from lasagne import layers
from nolearn.lasagne import NeuralNet
from matplotlib import pyplot
from readCSV import loaddata, load2d
from utils import AdjustVariable, plot_sample, draw_loss, write_file
from pandas.io.parsers import read_csv
import theano


def loadCSV(fname = None):
    df = read_csv(os.path.expanduser(fname))
    df = df.dropna()
    imagePaths = df['Image']
    return imagePaths

def extract_fileNames(imagePaths):
    paths = imagePaths.values
    alist=[]
    print(len(alist))
    for i in range(len(paths)):
        pathi = paths[i]
        lastIndex = pathi.rfind('/')
        name = pathi[lastIndex+1:]
        alist.append(name)

    print(len(alist))
    return alist

# trained model
FMODEL = '/data3/linhlv/Tete_dec/Output/fs_train_losses/loss_5000eps_rate82_0001_Early_' #v18.pickle'
FTEST = '/data3/linhlv/Tete_dec/Chav/test_'#v18.csv'
FSAVEFOLDER = '/data3/linhlv/Tete_dec/Chav/output/'
filename = FSAVEFOLDER + 'fs_loss_5000eps_rate82_0001_Early_'#v18.txt"

FSAVEIMAGES = FSAVEFOLDER + 'fs_loss_5000eps_rate82_0001_Early_'
#DATA=['v10','v11','v12','v13','v14','v15','v16','v17','v18']
DATA=['v10']
for i in DATA:
	fmodelf = FMODEL + i + '.pickle'
	ftestf = FTEST + i + '.csv'
	flandmarks = filename + i + '.txt'
	net = None
	sys.setrecursionlimit(100000)
	with open(fmodelf, 'rb') as f:
		net = pickle.load(f)

	X, _ = load2d(ftestf,test=True)
	y_pred = net.predict(X)

	# try to display the estimated landmarks on images
	paths = loadCSV(ftestf)
	fileNames = extract_fileNames(paths)

	for i in range(len(y_pred)):
		predi = y_pred[i]
		write_file(flandmarks,predi)
		saveImg = FSAVEIMAGES + fileNames[i];
        saveImg = saveImg.replace("JPG","PNG",3)
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plot_sample(X[i],predi,ax)
        fig.savefig(saveImg)
        pyplot.close(fig)

	print('Finish!')

