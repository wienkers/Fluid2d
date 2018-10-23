from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np

nc = Dataset('Vortex_00_his.nc', 'r')


kt = 0

vort = nc.variables['vorticity'][kt,:,:]
psi = nc.variables['psi'][kt,:,:]

# vort is at cell centers
# psi is at upper right corners

tmp = 0.5*(psi+np.roll(psi,1,axis=0))
psi_c = 0.5*(tmp+np.roll(tmp,1,axis=1))


plt.figure()
plt.plot(vort.ravel(), psi_c.ravel(), '.')
plt.xlabel('vorticity')
plt.ylabel('psi')
plt.show()
