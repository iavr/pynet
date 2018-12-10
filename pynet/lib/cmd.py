import string, sys
from lib import __, given, is_, iterate, union

# command-line parsing

class CommandLineError(ValueError):
	def __init__(self, msg):
		ValueError.__init__(self, msg)

def cmd(fun=__):

	# function signature, including positional and keyword arguments
	def sign(a=[], k={}):

		# exit or raise exception depending on fun
		def error(msg):
			raise CommandLineError(msg)

		def packed(a):
			return len(a) and a[-1][0] == '*'

		# print help
		def help(input):
			if packed(a): a[-1] = a[-1][1:] + "..."
			args = " ".join(["<%s>" % n for n in a])
			print "Usage: %s [options] %s\nOptions:" % (input[0], args)
			for s, l, v in k + [["h", "help", False]]:
				t = type(v).__name__
				if is_(bool)(v): print "-%s, --%s" % (s, l)
				else: print "-%s <%s>, --%s=<%s> [default: %s]" % (s, t, l, t, v)
			sys.exit(0)

		# split short/long flag name
		def split(name):
			s = name.split(':', 1)
			if len(s) < 2: s = [''] + s
			assert(len(s[0]) < 2 and len(s[1]) > 0)
			return s

		# split all given flag names; check all argument names
		k = [split(n) + [v] for n, v in k.iteritems()]

		# flag lookup
		single = {s: l for s, l, _ in k}
		double = {"--" + l: l for _, l, _ in k}
		lookup = union([single, double])

		# process flag as keyword argument
		def _flag(input, map, key, val):
			if key in ("h", "--help"):
				if given(fun): help(input)
				else: error("built-in flag '%s'" % key)
			try:
				l = lookup[key]
				typ = type(map[l])
				map[l] = True if typ is bool else typ(val())
			except KeyError:
				error("unrecognized command line option '%s'" % key)
			except (IndexError, StopIteration):
				error("missing argument after '%s'" % key)
			except ValueError:
				error("argument of type '%s' expected for '%s'" % (typ.__name__, key))

		# function call, given input string list
		def run(input=sys.argv):

			# initialize
			args, kargs = [], {l: v for s, l, v in k}
			flag = lambda *a: _flag(input, kargs, *a)
			flags = True

			# parse input arguments, separating flags from positional arguments
			for it, arg in iterate(input[1:]):
				if flags and arg[:2] == "--":
					arg = arg.split("=", 1)
					flag(arg[0], lambda: arg[1])
				elif flags and arg[0] == "-":
					if arg == "-": flags = False; continue
					for c in arg[1:]:
						flag(c, lambda: next(it))
				else: args.append(arg)

			# process positional arguments, optionally packing
			giv = len(args)
			if packed(a):
				exp = len(a) - 1
				args += [__] * (exp - giv)
				args = args[:exp] + [args[exp:]]
			else:
				exp = len(a)
				args += [__] * (exp - giv)
				if exp < giv:
					error("up to %d arguments expected (%d given)" % (exp, giv))

			# call function f if given, else return arguments
			if given(fun): return fun(*args, **kargs)
			else: return args, kargs

		return run

	return sign

#------------------------------------------------------------------------------

if __name__ == "__main__":

	def test(a=[], k={}):
		print "cmd(call)(%s, %s):\n" % (a, k)
		run = cmd()(a, k)
		def fun(c=""):
			split = [] if c == "" else c.split(" ")
			c = ["run"] + split
			print ">", " ".join(c)
			try:
				a, k = run(c)
				a = [repr(x) for x in a]
				k = ["%s=%s" % (n, v) for n, v in k.items()]
				print "call(%s)\n" % ", ".join(a + k)
			except CommandLineError as e:
				print "!cmd error: %s\n" % e
		return fun

	t = test(["input"])
	t()
	t("file")
	t("file extra")

	t = test(["input", "*more"])
	t()
	t("file")
	t("file one")
	t("file one two")

	t = test([], {"l:level": 2})
	t("")
	t("-l 3")
	t("-l text")
	t("-l")
	t("--level=3")
	t("--level=text")
	t("--level")

	t = test([], {"f:fast": False})
	t("")
	t("-f")
	t("-f maybe")
	t("--fast")
	t("--fast=maybe")
	t("--fact")

	t = test(["short", "long"], {"f:flag": False})
	t("h help -f")
	t("-f h help")
	t("-f -h --help")
	t("-f - -h --help")

	t = test([], {"o:one": False, "m:more": False, "f:flag": "unknown"})
	t("--one")
	t("--one --more")
	t("--one --more --flag=long")
	t("--one -m -f short")
	t("-omf shorter")
