from ..lib import __, has, trav, vec

# function call history
class net(list):

	def __init__(self, fun, arg):
		list.__init__(self)
		self.__idx = 0
		self.out = __
		self.push(fun, arg)

	def push(self, fun, arg):
		self.fun = fun
		self.arg = arg
		net.stack.append(self)
		return self

	@property
	def seq(self):
		return tuple(self) if self.fun.rec else ()

	def insert(self, c):
		self.__idx += 1
		self.append(c)
		return c

	def get(self, fun, arg):
		try: c = self[self.__idx]
		except IndexError: raise LookupError
		self.__idx += 1
		return c.push(fun, arg)

	def __call__(self, out):
		if self.__idx < len(self):
			self[:] = self[:self.__idx]
		self.__idx = 0
		net.stack.pop()
		self.out = out
		return self.out

	@staticmethod
	def rec(fun, *arg):
		try:
			return net.stack[-1].get(fun, arg)
		except IndexError: return net.root.push(fun, arg)
		except LookupError: return net.stack[-1].insert(net(fun, arg))
		except AttributeError:
			net.stack = []
			net.root = net(fun, arg)
			return net.root

	@staticmethod
	def traverse(*a):
		return trav(lambda c: c.seq)(net.root)(*a)

	@staticmethod
	def dump():
		rep = vec(lambda e: (
			e.shape if has('shape')(e) and len(e.shape) else
			rep(e()) if has('__call__')(e) else type(e)
		))
		def pre(c, tab):
			print '%s%s' % (tab, rep(c.out)), '<-', c.fun.name, rep(c.arg)
			if len(c.seq): print '%s{' % tab
		def post(c, tab):
			if len(c.seq): print '%s}' % tab
		def next(c, tab):
			return (tab + '   ',)
		net.traverse(pre, post, next)('')
