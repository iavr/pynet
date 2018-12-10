import numpy as np

normal = lambda a = 1: lambda shape: a * np.random.randn(*shape)
const = lambda a = 1: lambda shape: a * np.ones(shape)