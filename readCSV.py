

import os
import cv2
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

def readImage(arrayImgs):
	X = arrayImgs.values
	for i in range(len(X)):
		image = cv2.imread(X[i],cv2.IMREAD_GRAYSCALE)
		image = image.reshape(-1)
		image = image/255.
		image = image.astype(np.float32)
		X[i] = image
	X = np.vstack(X)
	return X

#scale the target to [-1,1]
def scaleTarget(target):
	print('Normalize target...')
	evencol = (target[:,::2] - 125)/125
	oddcol = (target[:,1::2] - 100)/100
	rs = np.empty((evencol.shape[0],evencol.shape[1] + oddcol.shape[1]))
	rs[:,::2] = evencol
	rs[:,1::2] = oddcol
	return rs

def loaddata(fname = None,test=False):
	if fname == None:
		fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname))
	df = df.dropna()
	imagePath = df['Image']
	X = readImage(imagePath)
	if not test:
		y = df[df.columns[1:]].values
		y = y.astype(np.float32)
		y = scaleTarget(y)
		X,y = shuffle(X,y,random_state=42)
		y = y.astype(np.float32)
		#print(y)
	else:
		y = None
	return X,y

# reshape (convert) the data from 49152 to 192x256 (h x w)
def load2d(fname=None,test=False):
	print(fname)
	X,y = loaddata(fname,test=test)
	X = X.reshape(-1,1,200,250)
	if not test:
		print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
		print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape, y.min(), y.max()))
	return X,y
