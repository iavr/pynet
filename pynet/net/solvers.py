from defn import solver

# stochastic gradient descent
def sgd(rate):
	def update(u, du): u.val[:] -= rate * du
	return solver(update)

# sgd with momentum
def momentum(rate, coef):
	def init(u):
		u.mom = np.zeros(u.val.shape)
	def update(u, du):
		u.mom[:] = coef * u.mom + rate * du
		u.val[:] -= u.mom
	return solver(update, init)
