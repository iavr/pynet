import numpy as np
from copy import copy
from time import time
from itertools import chain, izip, repeat
from contextlib import contextmanager

# identity map, constant maps, negation
id = lambda x: x
const = lambda c: lambda *_, **k: c
true, false = const(True), const(False)
void = const(None)
not_ = lambda f: lambda *a: not f(*a)

#------------------------------------------------------------------------------

# anonymous class instance
def instance():
	class _: pass
	return _()

# "not an object", returning itself whatever the operation
__ = instance()
given = lambda x: x is not __
none = const(__)
__.__add__ = none
__.__call__ = none
__.__getitem__ = none
__.__iadd__ = none
__.__nonzero__ = false
__.__repr__ = const('__')
# (optional operations)
__.__len__ = const(0)
__.__iter__ = none
__.__next__ = none

# workaround for required keyword arguments in python2
def req(v):
	if not given(v): raise TypeError('keyword argument required')
	return v

# optional value
opt = lambda what = __: lambda x: x if given(x) else what
fall = lambda what: lambda x: x if given(x) else what()

# has / get attribute, instance of
has = lambda name: lambda x: hasattr(x, name)
get = lambda name, what = __: lambda x: getattr(x, name, what)
is_ = lambda *t: lambda x, *_: isinstance(x, t)

#------------------------------------------------------------------------------

# context-managed dictionary backup/restore
@contextmanager
def let(var, val):
	bak = copy(var)
	var.update(val)
	yield __
	var.clear()
	var.update(bak)

#------------------------------------------------------------------------------

# pack/unpack parameters
pack = lambda f: lambda a: f(*a)
unpack = lambda f: lambda *a: f(a)
do = pack(none)

# tuple/list utilities
elem = lambda i: lambda *a: a[i]
flip = lambda x: reversed(x)

#------------------------------------------------------------------------------

# sequence, given generator function
class seq(object):
	def __init__(self, gen=__): self.gen = gen
	def __iter__(self): return self.gen()

# vector base class; deprecated slice operations redirected
class vector(list):
	def __getslice__(self, a, b):
		return self.__getitem__(slice(a, b))
	def __setslice__(self, a, b, x):
		return self.__setitem__(slice(a, b), x)

# recursive loop utilities
def attr(X):
	try: return [X.attr()]
	except AttributeError: return []
is_iterable = lambda t: t in (tuple, list) or issubclass(t, (seq, vector))
is_iter = lambda x: is_iterable(type(x))
make_iter = lambda t, *p: lambda x: t(x) if t is seq else t(x(), *p)
atom = lambda x: x if is_iter(x) else repeat(x)
def fresh(x):
	try: return x == type(x)()
	except ValueError: return False

# recursive tuple/list map-reducer
def reduce(r, f=id, axis=0):
	i, m = is_iterable, make_iter
	a = lambda X: (atom(x) for x in X)
	def fun(*X):
		x = X[axis]
		t, p = type(x), attr(x)
		g = lambda: (fun(*x) for x in izip(*a(X)))
		return (x if fresh(x) else r(m(t, *p)(g))) if i(t) else f(*X)
	return fun

# recursive versions of standard reductions, maps and void functions
vec   = lambda f, **k: reduce(id,   f, **k)
loop  = lambda f, **k: reduce(none, f, **k)
any_  = lambda f=id, **k: reduce(any, f, **k)
all_  = lambda f=id, **k: reduce(all, f, **k)
sum_  = lambda f=id, **k: reduce(sum, f, **k)
flat  = lambda f=id, **k: reduce(pack(tup), f, **k)
first = lambda f=id, **k: reduce(pack(car), f, **k)

# loop on f conditioned on c / recursive filtering
loop_if = lambda c, **k: lambda f: loop(lambda *x: f(*x) if c(*x) else __, **k)
only = lambda c, **k: vec(lambda x: x if c(x) else __, **k)

#------------------------------------------------------------------------------

# recursive filtering, including map-reduce
def filt(c, r=id, f=id, axis=0):
	i, m = is_iterable, make_iter
	a = lambda X: (atom(x) for x in X)
	k = lambda x: is_iter(x[axis]) or c(*x)
	e = lambda x: is_iter(x) and not len(x)
	s = lambda t, *p: lambda X: m(t, *p)(lambda: (x for x in X if not e(x)))
	def fun(*X):
		x = X[axis]
		t, p = type(x), attr(x)
		g = lambda: (fun(*x) for x in izip(*a(X)) if k(x))
		return (x if fresh(x) else r(s(t, *p)(m(t, *p)(g)))) if i(t) else f(*X)
	return fun

select = lambda c, **k: filt(c, tup, **k)

#------------------------------------------------------------------------------

# recursive tree map-reducer
def tree(r, s):
	def fun(descend, pre=__, post=__, next=__):
		def visit(e, *arg):
			a = pre(e, *arg) if given(pre) else __
			n = next(e, *arg) if given(next) else ()
			b = r(*(visit(x, *n) for x in descend(e)))
			c = post(e, *arg) if given(post) else __
			return s(a, b, c)
		return visit
	return fun

# tree traversor
traverse = tree(none, none)
trav = lambda d: lambda t: lambda *a: lambda *b: traverse(d, *a)(t, *b)

#------------------------------------------------------------------------------

# recursively get x if a dst, otherwise what / convert to dst
inst = lambda dst, what = __, **k: vec(lambda x: x if is_(dst)(x) else what, **k)
make = lambda dst, src = __, **k: vec(lambda x: x if is_(opt(dst)(src))(x) else dst(x), **k)

# recursive method / function caller
method = lambda name, *a, **k: vec(lambda x: opt(x)(get(name)(x)(*a)), **k)
call = lambda *a, **k: method('__call__', *a, **k)

#------------------------------------------------------------------------------

# mean on iterables, iterators or generators
def mean(x):
	x = iter(x)
	try: (n, s) = (1, copy(np.array(next(x))))
	except IndexError: return float('nan')
	for v in x:
		n += 1
		s += np.array(v)
	return s / n

# random access on infinite range [0, ...)
class iota: pass
iota.__init__ = void
iota.__getitem__ = elem(1)

#------------------------------------------------------------------------------

# timer functions
def tic(a = __):
	tic.time = time()
	return a

def toc():
	return time() - tic.time

# context-managed timer
@contextmanager
def timer():
	last = [time()]
	yield lambda: last[0]
	last[0] = time() - last[0]

#------------------------------------------------------------------------------

# dictionary union
def union(X):
	U = {}
	for x in X: U.update(x)
	return U

# like enumerate, but with iterator instead of index
def iterate(X):
	i = iter(X)
	while True:
		try: yield i, next(i)
		except StopIteration: break
