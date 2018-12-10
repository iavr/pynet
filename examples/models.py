import numpy as np
import pynet as pn
from pynet import __

#-------------
# linear model

def linear(std):

	# units
	fill = pn.normal(std)
	linear = pn.linear(fill)

	# evaluate class scores; return best class if in test mode
	def f(test, X, U, dim=__):
		scores = linear(X, U, dim=dim)
		return np.argmax(scores, axis=1) if test else scores

	return pn.fun(f, name='linear_net')

#------------------------
# two layer network model

def two_layer(std, hidden):

	# units
	fill = pn.normal(std)
	linear = pn.linear(fill)
	relu = pn.relu()

	# evaluate class scores; return best class if in test mode
	def f(test, X, U, dim=__):
		activ = relu(linear(X, U[0], dim=hidden))
		scores = linear(activ, U[1], dim=dim)
		return np.argmax(scores, axis=1) if test else scores

	return pn.fun(f, name='2layer_net')
