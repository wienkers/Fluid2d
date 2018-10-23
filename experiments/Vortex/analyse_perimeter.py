#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 14:20:45 2018

@author: roullet
"""

from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset('Vortex_00_his.nc', 'r')


kt = 0

vort = nc.variables['vorticity'][kt, :, :]
psi = nc.variables['psi'][kt, :, :]
nx = nc.dimensions['x'].size
nt = nc.dimensions['t'].size
time = nc.variables['t'][:]
#nc.close()

dx = 1./nx
# vort is at cell centers
# psi is at upper right corners

tmp = 0.5*(psi+np.roll(psi, 1, axis=0))
psi_c = 0.5*(tmp+np.roll(tmp, 1, axis=1))

# %%


def perimeter(field2d, isovalues):
    cont = plt.contour(field2d, isovalues)
    # print(np.shape(c.allsegs), len(c.allsegs[0]))
    nlevs = len(isovalues)
    lengths = np.zeros((nlevs, ))
    for j in range(nlevs):
        nbsegs = len(cont.allsegs[j])
        length = 0.
        for k in range(nbsegs):
            xpoly = cont.allsegs[j][k][:, 0]
            ypoly = cont.allsegs[j][k][:, 1]
            segs = np.sqrt((xpoly-np.roll(xpoly, 1))**2
                           +(ypoly-np.roll(ypoly, 1))**2)
            length += np.sum(segs)
        lengths[j] = length
    return lengths


# %%
bins = np.arange(-0.3, 1.3, 0.01)
value = 0.5*(bins[1:]+bins[:-1])
nbins = len(bins)-1
pdf = np.zeros((nt, nbins))
for kt in range(nt):
    vort = nc.variables['vorticity'][kt, :, :]
    n, b = np.histogram(vort, bins)
    pdf[kt, :] = n*(value**2)

plt.subplot(211)
plt.pcolor(time, value, pdf.T, vmin=0, vmax=4)
plt.subplot(212)
plt.plot(np.sum(pdf, axis=1))
# %%
levs = np.arange(0., 1., 0.05)+1e-6
perim = np.zeros((len(levs), nt))
for kt in range(nt):
    vort = nc.variables['vorticity'][kt, :, :]
    perim[:, kt] = perimeter(vort, levs)*dx
plt.close()
# %%
plt.pcolor(time, levs, np.log10(perim))

# %%
plt.figure()
for j, lev in enumerate(levs):
    plt.plot(time, perim[j, :], '+-', label=str(lev))
plt.legend()
plt.xlabel('time')
plt.ylabel('L')
plt.show()