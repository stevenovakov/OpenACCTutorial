#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as pt

zz = np.loadtxt(open("phi_oacc.csv", "rb"), delimiter=",")
xx = np.loadtxt(open("xx.csv", "rb"), delimiter=",")
yy = np.loadtxt(open("yy.csv", "rb"), delimiter=",")

lvls = np.linspace(np.min(zz), np.max(zz), 40, endpoint=True)

fig = pt.figure()
ax = fig.add_subplot(111)
cf = ax.contourf(xx, yy, zz, levels=lvls, cmap=pt.cm.RdGy)
circ = pt.Circle((0,0), 1, color='black', fill=False)
fig.gca().add_artist(circ)
pt.colorbar(cf, pad = 0.15)

fig.suptitle(r"$\Phi(x,y)$ w/ Relaxation Method - Jackson 2.13", fontsize=20)
ax.set_xlabel("x", fontsize=17)
ax.set_ylabel("y", fontsize=17)

fig.show()
raw_input()

fig.clf()
pt.close(fig)

aa = np.loadtxt(open("a_oacc.csv", "rb"), delimiter=",")
pt.imshow(aa, origin='lower', cmap=pt.cm.RdGy_r)
pt.title("P4All Jacobi Iteration Result", fontsize=20)
pt.xlabel("x", fontsize=17)
pt.ylabel("y", fontsize=17)
pt.colorbar()
pt.show()


