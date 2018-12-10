from ..lib import __
from defn import fun

# supervision by data (Y) and regularization (U) loss
def supervise(model, data, reg):

	# evaluate output; return if in test mode
	# else compute data and regularization loss
	def f(test, X, U, Y = __, **k):
		out = model(X, U, **k)
		return out if test else data(out, Y) + reg(U)

	return fun(f, name='supervise')
