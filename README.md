# pynet

`pynet` is a minimal Python library for dynamic automatic differentiation. The focus is on simplicity and it is meant to accompany the [differentiation lecture](https://sif-dlv.github.io/slides/diff.pdf) of [Deep Learning for Vision](https://sif-dlv.github.io/) course.

Licence
-------

`pynet` has a 2-clause BSD license. See file [LICENSE](/LICENSE) for the complete license text.

Directory structure
-------------------

`pynet` consists of Python code only. The directory structure is:

	/pynet       the pynet library
	/pynet/net   the core part of the library
	/pynet/lib   generic utilities
	/examples    concrete examples

Requirements
------------

`pynet` has been tested on Python 2.7.6. It does not have any requirements other than common Python libraries, in particular `math`, `numpy`, `matplotlib`, `sys`, `string`, `copy`, `time`, `itertools`, `contextlib`.

Usage
-----

The usage of `pynet` is demonstrated through a toy machine learning example, given in two versions

	python spiral00.py
	python spiral.py

both in directory `/examples`. The first version is a single-file independent implementation that is not using the library, and the second is adapted with the use of `pynet`. Try `python spiral.py -h` for options.
