# -*- coding: utf-8 -*-
#
# @fileoverview Copyright (c) 2019-2021, Stefano Gualandi,
#                via Ferrata, 1, I-27100, Pavia, Italy
#
#  @author stefano.gualandi@gmail.com (Stefano Gualandi)
#

import numpy as np
from OT1D import OT1D, parasort

np.random.seed(13)

N = 1000000

# Uniform samples
x = np.random.uniform(1, 2, N)
y = np.random.uniform(0, 1, N)

z = OT1D(x, y, p=2, sorting=True, threads=16)

print('Wasserstein distance of order 2, W2(x,y) =', z)

# Output:
# Wasserstein distance of order 2, W2(x,y) = 1.0002549459050794