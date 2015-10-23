#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pt

phi = np.loadtxt(open("output.csv", "rb"), delimiter=",")

pt.imshow(phi, origin='lower')
pt.colorbar()

pt.show()
