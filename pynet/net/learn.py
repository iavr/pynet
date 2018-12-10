import numpy as np
from ..lib import __, given, iota, seq, vec
from defn import fix, net, param

# shuffle array and return it
shuffle = np.random.permutation

# split a set of sequences into batches,
# optionally shuffled (choose 'order=shuffle' for that)
def split(step, order=iota):
	def fun(X, step):
		def gen():
			n = len(X)
			k, idx = min(step, n), order(n)
			for i in range(0, n-k+1, k):
				yield X[idx[i:i+k]]
		return seq(gen)
	return lambda X: vec(fun)(X, step)

# model accuracy
def acc(test):
	return vec(lambda X, Y: np.mean(test(X) == Y))

# model validation [loss, accuracy]
def valid(test):
	return vec(lambda X, Y: [test(X, Y), np.mean(test(X) == Y)])

# model training
def learn(model, solver, **k):

	# initial parameters, optimizer
	U = param()
	opt = solver.opt(model)

	# test output with fixed parameters unless ground truth given
	def test(X, Y = __):
		return model(X, U, Y, **k)() if given(Y) else model(X, fix(U))

	# optimize and return model output
	def train(X, Y):
		return opt(X, U, Y, **k)

	return vec(test), vec(train)
