from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np

param = Param('default.xml')
param.modelname = 'boussinesq'
param.expname = 'boxexchange_1'

# domain and resolution
param.nx = 64*2
param.ny = param.nx/2
param.npx = 1
param.Lx = 2.
param.Ly = param.Lx/2
param.geometry = 'square'

param.hydroepsilon = 1.

# time
param.tend = 10.
param.cfl = 0.2
param.adaptable_dt = True
param.dt = 1e-3
param.dtmax = 1e-1

# discretization
param.order = 3

# output
param.plot_var = 'buoyancy'
param.var_to_save = ['vorticity', 'buoyancy', 'v', 'psi']
param.list_diag = ['ke', 'pe', 'energy', 'vorticity', 'enstrophy', 'brms']
param.freq_his = 0.05
param.freq_diag = 0.01

# plot
param.plot_interactive = True
#  param.plotting_module = 'plotting_rayleigh'
param.freq_plot = 10
param.colorscheme = 'imposed'
param.cax = [-1, 1]
param.generate_mp4 = True

# physics
param.gravity = 1.
param.diffusion = True
param.noslip = True

grid = Grid(param)
xr, yr = grid.xr, grid.yr

y = yr[:, 0]/grid.Ly
delta = 0.12
grid.msk[(y < (0.5-delta/2)) | (y > (0.5+delta/2)), grid.nx//2+grid.nh] = 0
# add a mask
# hb  = np.exp( -(xr/param.Lx-0.5)**2 *50)*0.7
# grid.msk[(yr<hb)]=0
grid.finalize_msk()

grid.finalize_msk()


# Prandtl number is Kvorticity / Kbuoyancy

prandtl = 7  # prandtl = 7 for Water, cf wikipedia

deltab = 1.  # this is directly the Rayleigh number is L=visco=diffus=1

param.deltab = deltab  # make it known to param

L = param.Ly
visco = 1e-3
diffus = visco / prandtl

diffus = 2e-4
# diffus=1e-4 => boxexchange_0.mp4
# diffus=2e-4 => boxexchange_1.mp4
visco = diffus * prandtl

# Rayleigh number is
Ra = deltab * L**3 / (visco * diffus)
print('Rayleigh number is %4.1f' % Ra)

param.Kdiff = {}
param.Kdiff['vorticity'] = visco
param.Kdiff['buoyancy'] = diffus  # param.Kdiff['vorticity'] / prandtl

# time step is imposed by the diffusivity
param.dt = 0.25*grid.dx**2 / max(visco, diffus)
param.dtmax = param.dt
print('dt = %f' % param.dt)

f2d = Fluid2d(param, grid)
model = f2d.model

buoy = model.var.get('buoyancy')
vor = model.var.get('vorticity')


def sigmoid(x):
    return 1/(1+np.exp(-(x-0.5)*param.nx/4))


def stratif():
    b = (sigmoid(yr/param.Ly + 0.1*np.cos(np.pi/4+2*xr/param.Lx*np.pi))-0.5)
    return b * grid.msk


# add noise to trigger the instability
noise = np.random.normal(size=np.shape(yr)) * grid.msk
noise -= grid.domain_integration(noise)*grid.msk/grid.area
grid.fill_halo(noise)

buoy[xr < grid.Lx*0.5] = -1
buoy[xr >= grid.Lx*0.5] = +1
buoy *= grid.msk

model.set_psi_from_vorticity()

f2d.loop()
