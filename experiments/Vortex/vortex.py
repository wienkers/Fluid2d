from param import Param
from grid import Grid
from fluid2d import Fluid2d
import numpy as np
import ana_profiles as ap


param = Param('default.xml')
param.modelname = 'euler'
param.expname = 'Vortex_00'

# domain and resolution
param.nx = 64*2
param.ny = param.nx
param.Ly = param.Lx
param.npx = 2
param.npy = 1
param.geometry = 'closed'

# time
param.tend = 30.
param.cfl = 0.9
param.adaptable_dt = True
param.dt = 0.01
param.dtmax = 100

# discretization
param.order = 5
param.timestepping = 'RK3_SSP'

# output
param.var_to_save = ['vorticity', 'psi']
param.list_diag = ['ke', 'vorticity', 'enstrophy']
param.freq_plot = 10
param.freq_his = 1.
param.freq_diag = 0.1

# plot
param.freq_plot = 10
param.plot_interactive = True
param.plot_psi = True
param.plot_var = 'vorticity'
param.cax = np.array([-1., 1.])*4
param.colorscheme = 'imposed'
param.generate_mp4 = True

# physics
param.forcing = False
param.noslip = False
param.diffusion = False

grid = Grid(param)

param.Kdiff = 5e-2*grid.dx

xr, yr = grid.xr, grid.yr

# it's time to modify the mask and add obstacles  if you wish, 0 is land

msk_config = 'bay'

if msk_config == 'bay':
    x0, y0, radius = 0.5, 0.3, 0.2
    y1 = 0.49
    msk2 = ap.vortex(xr, yr, param.Lx, param.Ly,
                     x0, y0, radius, 'step')

    grid.msk[yr < y1] = 0
    grid.msk += np.asarray(msk2, dtype=int)
    grid.msk[grid.msk < 0] = 0
    grid.msk[grid.msk > 1] = 1
    grid.msk[0:1, :] = 0
    grid.finalize_msk()

elif msk_config == 'T-wall':
    i0, j0 = param.nx//2, param.ny//2
    di = int(0.25*param.Lx/grid.dx)
    grid.msk[:j0, i0] = 0
    grid.msk[j0, i0-di:i0+di] = 0
    grid.finalize_msk()


f2d = Fluid2d(param, grid)
model = f2d.model


vor = model.var.get('vorticity')


def vortex(param, grid, x0, y0, sigma,
           vortex_type, ratio=1):
    """Setup a compact distribution of vorticity

    at location x0, y0 vortex, width is sigma, vortex_type controls
    the radial vorticity profile, ratio controls the x/y aspect ratio
    (for ellipses)

    """
    xr, yr = grid.xr, grid.yr
    # ratio controls the ellipticity, ratio=1 is a disc
    x = np.sqrt((xr-param.Lx*x0)**2+(yr-param.Ly*y0)**2*ratio**2)

    y = x.copy()*0.

    if vortex_type in ('gaussian', 'cosine', 'step'):
        if vortex_type == 'gaussian':
            y = np.exp(-x**2/(sigma**2))

        elif vortex_type == 'cosine':
            y = np.cos(x/sigma*np.pi/2)
            y[x > sigma] = 0.

        elif vortex_type == 'step':
            y[x <= sigma] = 1.
    else:
        print('this kind of vortex (%s) is not defined' % vortex_type)

    return y


# 2/ set an initial tracer field
vtype = 'gaussian'
# vortex width
sigma = 0.0*param.Lx

vortex_config = 'dipole2'

if vortex_config == 'single':
    vtype = 'gaussian'
    sigma = 0.03*param.Lx
    vor[:] = vortex(param, grid, 0.4, 0.54, sigma,
                    vtype, ratio=1)

elif vortex_config == 'dipolebay':
    vtype = 'gaussian'
    sigma = 0.03*param.Lx
    y2 = 0.53
    vor[:] = vortex(param, grid, 0.15, y2, sigma,
                    vtype, ratio=1)
    vor[:] -= vortex(param, grid, 0.75, y2, sigma,
                     vtype, ratio=1)

elif vortex_config == 'dipole2':
    vtype = 'gaussian'
    sigma = 0.03*param.Lx
    y2 = 0.2
    vor[:] = -vortex(param, grid, 0.8, 0.04, sigma,
                     vtype, ratio=1)
    vor[:] += vortex(param, grid, 0.5, 0.55, sigma,
                     vtype, ratio=1)

elif vortex_config == 'rankine':
    vtype = 'step'
    ring_sigma = 0.2*param.Lx
    ring_amp = 1.
    vor[:] = ring_amp * vortex(param, grid, 0.5, 0.5, ring_sigma,
                               vtype, ratio=1)
    # sigma ring, core = 0.2, 0.135 yields a tripole (with step distribution)
    # sigma ring, core = 0.2, 0.12 yields a dipole (with step distribution)
    core_sigma = 0.173*param.Lx
    core_amp = ring_amp*(ring_sigma**2-core_sigma**2.)/core_sigma**2.
    vor[:] -= (core_amp+ring_amp)*vortex(param, grid, 0.5, 0.5, core_sigma,
                                         vtype, ratio=1)

elif vortex_config == 'dipole':
    vtype = 'gaussian'
    sigma = 0.04*param.Lx
    vor[:] = vortex(param, grid, 0.3, 0.52, sigma, vtype)
    vor[:] -= vortex(param, grid, 0.3, 0.48, sigma, vtype)

elif vortex_config == 'chasing':
    sigma = 0.03*param.Lx
    vtype = 'step'
    vor[:] = vortex(param, grid, 0.3, 0.6, sigma, vtype)
    vor[:] -= vortex(param, grid, 0.3, 0.4, sigma, vtype)
    vor[:] += vortex(param, grid, 0.1, 0.55, sigma, vtype)
    vor[:] -= vortex(param, grid, 0.1, 0.45, sigma, vtype)

elif vortex_config == 'corotating':
    sigma = 0.06*param.Lx
    dist = 0.25*param.Lx
    vtype = 'gaussian'
    vor[:] = vortex(param, grid, 0.5, 0.5+dist/2, sigma, vtype)
    vor[:] += vortex(param, grid, 0.5, 0.5-dist/2, sigma, vtype)

elif vortex_config == 'collection':
    vtype = 'cosine'
    x0 = [0.3, 0.4, 0.6, 0.8]
    y0 = [0.5, 0.5, 0.5, 0.5]
    amplitude = [1, -2, -1, 2]
    width = np.array([1, 0.5, 1, 0.5])*0.04*param.Lx

    for x, y, a, s in zip(x0, y0, amplitude, width):
        vor[:] += a*vortex(param, grid, x, y, s, vtype)

elif vortex_config == 'unequal':
    # Melander, Zabusky, McWilliams 1987
    # Asymmetric vortex merger in two dimensions: Which vortex is 'victorious'?
    s1 = 0.04*param.Lx
    a1 = 1.
    s2 = 0.1*param.Lx
    a2 = 0.2
    vtype = 'cosine'
    vor[:] = a1*vortex(param, grid, 0.5, 0.6, s1, vtype)
    vor[:] += a2*vortex(param, grid, 0.5, 0.4, s2, vtype)


vor[:] = vor*grid.msk

if False:
    np.random.seed(1)  # this guarantees the results reproducibility
    noise = np.random.normal(size=np.shape(yr))*grid.msk
    noise -= grid.domain_integration(noise)*grid.msk/grid.area
    grid.fill_halo(noise)

    noise_amplitude = 1e-3

    vor += noise*noise_amplitude

model.set_psi_from_vorticity()

# % normalization of the vorticity so that enstrophy == 1.
model.diagnostics(model.var, 0)
enstrophy = model.diags['enstrophy']
# print('enstrophy = %g' % enstrophy)
vor[:] = vor[:] / np.sqrt(enstrophy)
model.set_psi_from_vorticity()


f2d.loop()
