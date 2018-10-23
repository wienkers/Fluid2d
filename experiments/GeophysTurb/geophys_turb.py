from param import Param
from grid import Grid
from fluid2d import Fluid2d
from numpy import exp,sqrt,pi,cos,random,shape

param = Param('default.xml')
param.modelname = 'quasigeostrophic'
param.expname = 'GFD_2'

# domain and resolution
param.ny = 64
param.nx = param.ny*2
param.npy=1
param.Lx = 2.
param.Ly = param.Lx/2
param.geometry='square'

# time
param.tend=4e4
param.cfl=1.
param.adaptable_dt=False
param.dtmax=100.

# discretization
param.order=5


amplitude = 10e-1

param.dt = 0.1/amplitude

# output
param.generate_mp4 = True
param.var_to_save=['pv','psi']
param.list_diag=['ke','pv','pv2']
param.freq_his=50
param.freq_diag=10

# plot
param.plot_var='vorticity'
param.freq_plot=10
a=amplitude
param.cax=[-a,a]
param.plot_interactive=True
param.colorscheme='imposed'

# physics
param.beta=1.
param.Rd=0.5#2*grid.dx
param.forcing = False
param.noslip = True
param.diffusion=False

grid  = Grid(param)
param.Kdiff=0.5e-4*grid.dx


# add an island
#grid.msk[60:70,30:40]=0
#grid.finalize_msk()


f2d = Fluid2d(param,grid)
model = f2d.model

xr,yr = grid.xr,grid.yr
vor = model.var.get('pv')


# set an initial tracer field


vor[:] = amplitude*random.normal(size=shape(vor))*grid.msk
y=vor[:]*1.
model.ope.fill_halo(y)
vor[:]=y

model.add_backgroundpv()

model.set_psi_from_pv()

f2d.loop()
