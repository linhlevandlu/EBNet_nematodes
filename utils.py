import sys
import os
import numpy as np
from matplotlib import pyplot
from readCSV import load2d

# drawing the loss of train and valid to predict overfitting
# the data is stored in train_histogry_ of the network
def draw_loss(net):
	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	pyplot.plot(train_loss, linewidth=3, label="train")
	pyplot.plot(valid_loss, linewidth=3, label="valid")
	pyplot.grid()
	pyplot.legend()
	pyplot.xlabel("epoch")
	pyplot.ylabel("loss")
	pyplot.ylim(1e-6,1e0)
	pyplot.yscale("log")
	pyplot.show()

def draw_loss_2(net,savepath):
    train_loss = None
    valid_loss = None
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.figure(np.random.randint(100))
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-5,1e0)
    pyplot.yscale("log")
    pyplot.savefig(savepath,dpi=90)

def test(net,ftest,fsave):
	X, _ = load2d(ftest,test=True)
	y_pred = net.predict(X)
	fig = pyplot.figure(figsize=(4, 4))
	fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)
	fig.savefig(fsave,dpi=90)
	pyplot.close(fig)

# draw a result sample
def plot_sample(x, y, axis):
    img = x.reshape(200, 250)
    axis.imshow(img, cmap='gray')
    #axis.scatter(y[0::2], y[1::2], marker='x', s=10)
    axis.scatter((y[::2] * 125) + 125, (y[1::2] * 100) + 100, marker='x', s=5)


def plot_weights(weights):
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(weights[:, i].reshape(200, 250), cmap='gray')
    pyplot.show()

# write the predicted landmarks into a file
def write_file(filename,y_predict):
	f = open(filename,'a+')
	#f.write('\n')
	for j in range(len(y_predict)):
		f.write(str(y_predict[j]))
		if j%2 == 0:
			f.write('\t')
		else:
			f.write('\n')		
	f.close()

# define a class to update the learning parameter (learning rate and momentum)
class AdjustVariable(object):
	'''
	This class defines the way to update the learning_rate and momentum
	during training. 
	'''
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self,nn,train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start,self.stop,nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = np.float32(self.ls[epoch-1])
		getattr(nn,self.name).set_value(new_value)

# define class stop early
class EarlyStopping(object):
	def __init__(self, continue_in = 500):
		self.patience = continue_in
		self.best_valid = np.inf
		self.best_valid_epoch = -1
		self.best_weights = None

	def __call__(self,nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping")
			print("Best valid loss was {:.6f} at epoch {}.".format(self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()
