import numpy as np
from ..lib import __, loop, req, sum_
from defn import node, tag, tagged, unit

# linear
def linear(weight = np.random.randn, bias = np.zeros):
	def init(X, U, dim=__):
		M, N = X.shape[1], req(dim)
		U[:] = [tag('reg')(weight((M, N))), bias((1, N))]
	def fwd(test, X, (W, b)):
		A = np.dot(X, W) + b
		if test: return A
		def back(dA, dX, (dW, db)):
			dX += np.dot(dA, W.T)
			dW += np.dot(X.T, dA)
			db += np.sum(dA, axis=0)
		return node(A, back)
	return unit(fwd, init, name='linear')

# rectifier
def relu():
	def fwd(test, A):
		Z = np.maximum(0, A)
		if test: return Z
		def back(dZ, dA): dA += dZ * (Z > 0)
		return node(Z, back)
	return unit(fwd, name='relu')

# sum
def add():
	def fwd(test, *X):
		S = sum(X)
		if test: return S
		def back(dS, *dX):
			for dx in dX: dx += dS
		return node(S, back)
	return unit(fwd, name='add')

# binary operator +
node.__add__ = lambda *a: add()(*a)

# logistic regression (average cross-entropy) loss
def logistic():
	def fwd(test, X, Y):
		N, K = X.shape
		Y = one_hot(Y, K)
		P = np.exp(X)
		P[:] = P / np.sum(P, axis=1, keepdims=True)
		L = -np.sum(np.log(np.sum(P * Y, axis=1))) / N
		if test: return L
		def back(dL, dX, _): dX += dL * (P - Y) / N
		return node(L, back)
	return unit(fwd, name='logistic')

# l2 regularization loss
def l2_reg(decay):
	def fwd(test, W):
		L = decay / 2 * sum_(np.linalg.norm)(W)
		if test: return L
		def back(dL, dW):
			def fun(x, dx): dx += dL * decay * x
			loop(fun)(W, dW)
		return node(L, back)
	return unit(fwd, pre=tagged('reg'), name='reg')

# one-hot target encoding
def one_hot(a, k):
	n = a.size
	b = np.zeros([n, k])
	b[xrange(n), a] = 1
	return b

