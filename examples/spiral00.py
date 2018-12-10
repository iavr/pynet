import sys
import numpy as np
import matplotlib.pyplot as plt

# minimal 2d toy example adapted from
# http://cs231n.github.io/neural-networks-case-study/

# train a linear classifier
def linear(X, y, K, plot=True, save=False):

	# number of points, dimensions, ground truth
	N, D = X.shape
	Y = one_hot(y, K)

	# initialize parameters randomly
	W = 0.01 * np.random.randn(D, K)
	b = np.zeros((1, K))

	# hyperparameters
	step_size = 1e-0
	reg = 1e-3 # regularization strength

	# gradient descent loop
	for i in xrange(200):

		# evaluate class scores, [N x K]
		scores = np.dot(X, W) + b

		# compute the class probabilities
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

		# compute the loss: average cross-entropy loss and regularization
		cross_entropy = -np.log(np.sum(probs * Y, axis=1, keepdims=True))
		data_loss = np.sum(cross_entropy)/N
		reg_loss = 0.5*reg*np.sum(W*W)
		loss = data_loss + reg_loss
		if i % 20 == 0:
			print "iteration %d: loss %f" % (i, loss)

		# compute the gradient on scores
		dscores = (probs - Y) / N

		# back-propagate the gradient to the parameters (W,b)
		dW = np.dot(X.T, dscores)
		db = np.sum(dscores, axis=0, keepdims=True)

		dW += reg*W # regularization gradient

		# perform a parameter update
		W += -step_size * dW
		b += -step_size * db

	# evaluate training set accuracy
	scores = np.dot(X, W) + b
	predicted_class = np.argmax(scores, axis=1)
	print 'training accuracy: %.2f' % np.mean(predicted_class == y)

	# plot the resulting classifier
	h = 0.02
	x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
	y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
								np.arange(y_min, y_max, h))
	x = np.c_[xx.ravel(), yy.ravel()]
	Z = np.dot(x, W) + b
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)

	if plot:
		fig = plt.figure()
		plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
		plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.show()
		if save:
			fig.savefig('spiral_linear.png')

# train a two-layer network
def two_layer(X, y, K, plot=True, save=False):

	# number of points, dimensions, ground truth
	N, D = X.shape
	Y = one_hot(y, K)

	# initialize parameters randomly
	h = 100 # size of hidden layer
	W = 0.01 * np.random.randn(D, h)
	b = np.zeros((1, h))
	W2 = 0.01 * np.random.randn(h, K)
	b2 = np.zeros((1, K))

	# some hyperparameters
	step_size = 1e-0
	reg = 1e-3 # regularization strength

	# gradient descent loop
	for i in xrange(2000):

		# evaluate class scores, [N x K]
		hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
		scores = np.dot(hidden_layer, W2) + b2

		# compute the class probabilities
		exp_scores = np.exp(scores)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

		# compute the loss: average cross-entropy loss and regularization
		cross_entropy = -np.log(np.sum(probs * Y, axis=1, keepdims=True))
		data_loss = np.sum(cross_entropy)/N
		reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
		loss = data_loss + reg_loss
		if i % 100 == 0:
			print "iteration %d: loss %f" % (i, loss)

		# compute the gradient on scores
		dscores = (probs - Y) / N

		# back-propagate the gradient to the parameters
		# first backprop into parameters W2 and b2
		dW2 = np.dot(hidden_layer.T, dscores)
		db2 = np.sum(dscores, axis=0, keepdims=True)
		# next backprop into hidden layer
		# backprop the ReLU non-linearity
		dhidden = np.dot(dscores, W2.T) * (hidden_layer > 0)
		# finally into W,b
		dW = np.dot(X.T, dhidden)
		db = np.sum(dhidden, axis=0, keepdims=True)

		# add regularization gradient contribution
		dW2 += reg * W2
		dW += reg * W

		# perform a parameter update
		W += -step_size * dW
		b += -step_size * db
		W2 += -step_size * dW2
		b2 += -step_size * db2


	# evaluate training set accuracy
	hidden_layer = np.maximum(0, np.dot(X, W) + b)
	scores = np.dot(hidden_layer, W2) + b2
	predicted_class = np.argmax(scores, axis=1)
	print 'training accuracy: %.2f' % np.mean(predicted_class == y)

	# plot the resulting classifier
	h = 0.02
	x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
	y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
								np.arange(y_min, y_max, h))
	x = np.c_[xx.ravel(), yy.ravel()]
	Z = np.dot(np.maximum(0, np.dot(x, W) + b), W2) + b2
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)

	if plot:
		fig = plt.figure()
		plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
		plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.show()
		if save:
			fig.savefig('spiral_net.png')
		plt.hold(False)

# generate data
def generate(plot=True, save=False):

	if plot:
		plt.hold(True)
		plt.close("all")
		plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
		plt.rcParams['image.interpolation'] = 'nearest'
		plt.rcParams['image.cmap'] = 'gray'

	Nc = 100 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes
	N = Nc * K # total number of points
	X = np.zeros((N,D))
	y = np.zeros(N, dtype='uint8')
	for j in xrange(K):
		ix = range(Nc*j,Nc*(j+1))
		r = np.linspace(0.0,1,Nc) # radius
		t = np.linspace(j*4,(j+1)*4,Nc) + np.random.randn(Nc)*0.2 # theta
		X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
		y[ix] = j

	if plot:
		fig = plt.figure()
		plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
		plt.xlim([-1,1])
		plt.ylim([-1,1])
		plt.show()
		if save:
			fig.savefig('spiral_raw.png')

	return X, y, K

# one-hot target encoding
def one_hot(a, k):
	n = a.size
	b = np.zeros([n, k])
	b[xrange(n), a] = 1
	return b

def main(plot=True, save=False):
	X, y, K = generate(plot, save)
	linear(X, y, K, plot, save)
	two_layer(X, y, K, plot, save)

# execute main() when in script
if __name__ == "__main__":
	main()
