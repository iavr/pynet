import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('..')
from pynet import __, timer
import pynet as pn
import models

# minimal 2d toy example of pynet00.py adapted for pynet

def main(display=False, save=False):

	# generate data
	K = 3
	data = generate(dim=2, classes=K, samples=100);

	if display or save:

		# initialize plots
		plt.hold(True)
		plt.close('all')

		# display the raw data
		plt.figure()
		filename = "spiral_raw.pdf" if save else __
		disp_data(data, ([-1,1], [-1,1]), display, filename)

	# network models
	hidden = 100  # size of hidden layer
	std = .01     # weight initialization scale
	linear = models.linear(std)
	two_layer = models.two_layer(std, hidden)

	# batch parameters
	batch = 100                   # mini-batch size
	interval = lambda e: e / 10   # every how many epochs to report
	seq = pn.split(batch)(data)   # split data

	# learning parameters
	rate = 1e-0   # learning rate
	decay = 1e-3  # weight decay parameter
	data_loss = pn.logistic()
	reg_loss = pn.l2_reg(decay)
	solver = pn.sgd(rate)

	# learn classifiers
	for (model, name, epochs) in (
			(linear, "linear", 50),
			(two_layer, "two-layer", 300),
		):

		print("\n%s classifier:" % name)
		network = pn.supervise(model, data_loss, reg_loss)
		classify, train = pn.learn(network, solver, dim=K)
		loop(train(*seq), epochs, interval(epochs))

		# dump network
		print('\nnetwork dump:')
		pn.net.dump()

		# evaluate on training set
		evaluate(data, classify)

		# display classifier
		filename = "spiral_%s.pdf" % name if save else __
		if display or save: disp_class(data, classify, display, filename)

	# finalize plots
	if display or save: plt.hold(False)

#----------
# utilities

# evaluate classifier
def evaluate((X, Y), classify):
	print("training accuracy: %.2f" % (np.mean(classify(X) == Y)))

# main learning loop
def loop(seq, epochs, interval):
	temp = "training epoch %d: training loss %f, time %.3fs"
	for e in range(epochs):
		with timer() as time: loss = pn.mean(seq)
		if e % interval == 0: print(temp % (e, loss, time()))

# generate data
def generate(dim=2, classes=3, samples=100):
	D = dim        # dimensionality
	K = classes    # number of classes
	Nc = samples   # number of points per class
	N = Nc * K     # total number of points
	intra = 4                 # intra-class angle increment
	inter = 2 * math.pi / K   # inter-class angle increment
	X = np.zeros([N, D])
	Y = np.zeros(N, dtype='uint8')
	np.random.seed(0)
	for k in xrange(K):
		I = range(Nc * k, Nc * (k+1))
		r = np.linspace(0, 1, Nc)                       # radius
		t = np.linspace(k*inter, k*inter + intra, Nc)   # theta mean
		t = t + np.random.randn(Nc) * 0.2               # theta noise
		X[I] = np.c_[r * np.cos(t), r * np.sin(t)]
		Y[I] = k
	return (X, Y)

# display data
def disp_data((X, Y), lim=__, disp=False, name=__):
	plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)
	if lim:
		plt.xlim(lim[0])
		plt.ylim(lim[1])
	if name: plt.savefig(name)
	if disp: plt.show()

# display classifier
def disp_class((X, Y), classify, disp=False, name=__, step=.02):
	x, y = X[:, 0], X[:, 1]
	rx = np.arange(x.min() - 1, x.max() + 1, step)
	ry = np.arange(y.min() - 1, y.max() + 1, step)
	Gx, Gy = np.meshgrid(rx, ry)
	data = np.c_[Gx.ravel(), Gy.ravel()];
	Z = classify(data)
	Z = Z.reshape(Gx.shape)
	fig = plt.figure()
	plt.contourf(Gx, Gy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	extr = lambda x: (x.min(), x.max())
	disp_data((X, Y), (extr(rx), extr(ry)), disp, name)

# execute main() when in script
if __name__ == "__main__":
	pn.cmd(main)([], {"d:display": False, "s:save": False})()
