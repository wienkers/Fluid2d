from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np

param = Param('default.xml')
param.modelname = 'boussinesq'
param.expname = 'icecube_1'

# domain and resolution
ratio = 4
param.nx = 64*ratio
param.ny = param.nx/ratio
param.npx = 1
param.Lx = 1.*ratio
param.Ly = param.Lx/ratio
param.geometry = 'square'

# time
param.tend = 4.
param.cfl = 0.4
param.adaptable_dt = True
param.dt = 1e-2
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
param.plot_psi = False
param.freq_plot = 10
param.colorscheme = 'imposed'
param.cax = [-1e2, 1e2]
param.generate_mp4 = True

# physics
param.gravity = 1.
param.forcing = True
param.forcing_module = 'forcing_icecube'
param.diffusion = True
param.noslip = False

grid = Grid(param)
# Prandtl number is Kvorticity / Kbuoyancy
xr, yr = grid.xr, grid.yr
xx = np.abs(xr-param.Lx*0.5)
nh = grid.nh

grid.msk[(yr+xx) < 0.4] = 0
grid.msk[xx > 0.3] = 1
grid.msk[:nh, :] = 0
grid.msk[:, :nh] = 0
grid.msk[:, -nh:] = 0
grid.finalize_msk()
grid.finalize_msk()

prandtl = 1./7  # prandtl = 7 for Water, cf wikipedia


deltab = 5.  # this is directly the Rayleigh number is L=visco=diffus=1

param.deltab = deltab  # make it known to param

L = param.Ly
visco = .1e-2
diffus = visco / prandtl

#diffus = 2e-3
#visco = diffus * prandtl

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

xr, yr = grid.xr, grid.yr
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

vor[:] = + 1e1*noise*grid.msk

model.set_psi_from_vorticity()

f2d.loop()
