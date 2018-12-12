import numpy as np
from .. import lib
from ..lib import __, any_, given, id, is_, has, none, not_
from ..lib import vec, vector
from net import net

#-----------------------------------------------------------------------------
# in the following, a 'list' refers to a single object, a (python) list or
# tuple of objects, or recursively a nested list or tuple of any of the above.
#-----------------------------------------------------------------------------

#-----------------------------------------------------------------------------
# an object is variable if it contains a member called diff() representing a
# derivative. this may be a cell, node or block (see below). an object or list
# can become variable by calling var(), which introduces derivatives.
# only_var() keeps only variables from a list and replaces each fixed object
# by __, for which updates by operator += are ignored. a list is fixed if it
# does not contain any variable. it can become fixed by calling fix(), which
# discards derivatives.

is_var = has('diff')
var = lambda x: lib.make(cell, type(__))(x)
only_var = lib.only(is_var)
loop_var = lib.loop_if(is_var)
fixed = not_(any_(is_var))
fix = lib.call()

#-----------------------------------------------------------------------------
# apart from the standard properties & methods defined in lib module, non-
# standard are defined here. properties like 'shape' (e.g. of numpy.ndarray)
# are defined as __ and method like 'diff' (e.g. of cell below) as none, a
# function returning __.

__.shape = __
__.diff = none

#-----------------------------------------------------------------------------
# an object or list of objects may given one or more tags with function
# tag(), providing the tag name(s). tag() converts the object(s) to cell(s)
# (see below) before setting the appropriate attributes. given a list of
# objects, tagged() recursively selects only the ones having one or more
# tags, otherwise keeping the list structure.

def tagged(*name):
	return lib.filt(has(*name))

def tag(*name):
	def fun(x):
		for n in name: setattr(x, n, __)
		return x
	return vec(lambda x: fun(var(x)))

#-----------------------------------------------------------------------------
# a parameter can be initialized as param() and passed as input into operator
# () of a unit/fun (see below). in this case its address remains the same to
# be reused later, but its elements are automatically substituted by the
# elements of a list returned by method init() of the unit/fun.
#
# param class itself inherits a list and automatically extends itself with
# empty param() objects upon element access by operator [] or attribute access
# by operator ".". an empty param() is initialized to actually contain an __
# object, denoting a placeholder. This is removed before extending.

class param(vector):

	key = {'op': id}

	def __init__(self, elem=[__], attr={}):
		vector.__init__(self, elem)
		self.__attr = attr

	def attr(self):
		return lib.copy(self.__attr)

	def __getitem__(self, i):
		self.__extend(i)
		return vector.__getitem__(self, i)

	def __setitem__(self, i, x):
		self.__extend(i, x)
		return vector.__setitem__(self, i, param.key['op'](x))

	def __getattr__(self, n):
		if n[:1] == "_": return vector.__getattr__(self, n)
		if n not in self.__attr: self.__setattr__(n, param())
		return self[self.__attr[n]]

	def __setattr__(self, n, x):
		if n[:1] == "_": return vector.__setattr__(self, n, x)
		self.__init()
		if n not in self.__attr: self.__attr[n] = len(self)
		self[self.__attr[n]] = x

	def __init(self):
		if self == [__]: del self[:]

	def __extend(self, i, x=()):
		if is_(slice)(i): i = max(xrange(*i.indices(len(x))))
		self.__init()
		k = i - len(self)
		if k >= 0: self.extend([param() for _ in range(k + 1)])

	@staticmethod
	def let(**key):
		return lib.let(param.key, key)

#-----------------------------------------------------------------------------
# pair of a value (val) and the corresponding derivative (diff). diff() is a
# getter / setter for the otherwise private member __diff. operator () returns
# val; operator += increases diff and -= decreases val.

class cell:

	def __init__(self, val):
		self.val = val
		self.__diff = __
		self.reset()

	def reset(self):
		self.__fresh = True

	def set(self):
		if self.__fresh:
			self.__diff = np.ones(self.val.shape)
			self.__fresh = False

	def diff(self):
		return self.__diff

	def __call__(self): return self.val

	def __iadd__(self, D):
		self.__diff = D if self.__fresh else self.__diff + D
		self.__fresh = False

#-----------------------------------------------------------------------------
# computational node holding a cell or list of cells (data) and a custom
# backward function (back) that receives as first argument the cell's diff().
# the remaining arguments are replaced by __ if they contain no cell, hence
# are not trainable. if no arguments contain a cell, then the the node is never
# constructed (see below). operations (), += and diff() delegate to data, but
# extend to lists.

class node:

	def __init__(self, val, back, pre=id):
		self.__data = lib.make(cell, node)(val)
		self.__back = back
		self.__pre = pre

	def set(self):
		op = lambda x: x.set()
		lib.loop(op)(self.__data)

	def diff(self, *a):
		op = lambda x: x.diff(*a)
		return vec(op)(self.__data)

	def __call__(self):
		return lib.call()(self.__data)

	def __iadd__(self, D):
		def op(x, v): x += v
		lib.loop(op)(self.__data, D)

	def __back__(self, *a):
		self.__back(self.diff(), *a)

	def back(self, *a):
		self.set()
		self.__back__(*only_var(self.__pre(a)))

	def pre(self, p): self.__pre = p

#-----------------------------------------------------------------------------
# computational block; like a node but back() is assumed to contain other
# nodes or blocks rather than custom code. back() does not receive diff() as
# an argument; rather, diff() is automatically propagated into the underlying
# output node/block.

class block(node):

	def __back__(self, *a):
		self._node__back(*a)

#-----------------------------------------------------------------------------
# forward unit holding a custom forward function (fun) as well as
# initialization (init).
#
# fun() receives as first argument a flag that is false if any of the
# remaining contains a cell, hence is trainable (training mode), in which case
# fun() is assumed to return a unit; otherwise, the flag is true (test mode)
# and fun() is assumed to return an arbitrary value. in any case, all remaining
# arguments to fun() are striped to their raw values if they are pairs.
#
# init() is assumed to initialize any parameters

class unit:

	rec = False

	def __init__(self, fun, init=__, pre=id, name=__):
		self.__fun = fun
		self.__init = init
		self.__pre = pre
		self.name = name if given(name) else '<anonymous function>'

	def __call__(self, *a, **k):
		a = self.__pre(a)
		if any_(not_(given))(a):
			with param.let(op=var): self.__init(*fix(a), **k)
		ret = net.rec(self, *a)
		return ret(self.pre(self.__fun(fixed(a), *fix(a))))

	def pre(self, x):
		lib.method('pre', self.__pre)(x)
		return x

#-----------------------------------------------------------------------------
# forward function; like a unit but fun() is assumed to contain other units or
# functions rather than custom code. fun() receives arguments as given
# including pairs and returns a block in training mode, whose backward function
# is automatically computed.
#
# function opt() calls method back() only if it exists; useful because
# intermediate computation results may be units/blocks if they have received
# at least one cell as input (hence are trainable and can operate backwards)
# or arbitrary values if they have received fixed input (hence are not
# trainable and have no method back().

class fun(unit):

	rec = True

	def __init__(self, fun, init=__, pre=id, name=__):
		unit.__init__(self, fun, init=init, pre=pre, name=name)

	def __call__(self, *a, **k):
		a = self._unit__pre(a)
		ret = net.rec(self, *a)
		test = fixed(a)
		out = self._unit__fun(test, *a, **k)
		if test: return ret(out)
		def back(*_):
			for c in lib.flip(ret.seq):
				lib.method('back', *c.arg)(c.out)
		return ret(block(out, back, pre=self._unit__pre))

#-----------------------------------------------------------------------------
# forward-only unit. merely propagates forward and caches the output like all
# units. does not propagate backward.

def forward(fun, name):
	def fwd(test, *X):
		X = fun(*X)
		return X if test else node(X, none)
	return unit(fwd, name=name)

#-----------------------------------------------------------------------------
# solver object. reset(), to be called before backpropagation, resets
# parameters, setting derivatives to zero. update(), to be called after the
# backward pass, updates parameters using the computed derivatives and a
# particular update rule. init(), automatically called only once if needed,
# initializes any auxiliary variable assigned to each parameter, depending
# on the solver.
#
# opt() is the main learning function that applies a forward and backward
# pass to a model so that gradients are computed for the model parameters;
# then it updates the parameters according to its update rule and returns the
# network output.

class solver:

	def __init__(self, update, init=__):
		self.__update = update
		self.__init = init

	def init(self, U):
		loop_var(lambda u: self.__init(u))(U)

	def reset(self, U):
		loop_var(lambda u: u.reset())(U)

	def update(self, U):
		def op():
			loop_var(lambda u: self.__update(u, u.diff()))(U)
		try: op()
		except AttributeError:
			self.init(U)
			op()

	def opt(self, model):
		def fun(*X, **k):
			out = model(*X, **k)   # forward pass
			self.reset(X)          # reset gradients
			out.back(*X)           # backward pass
			self.update(X)         # update parameters
			return out()
		return fun
