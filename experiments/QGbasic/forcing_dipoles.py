from param import Param
import ana_profiles as ap

class Forcing(Param):
    """ define the forcing """

    def __init__(self, param, grid):

        self.list_param = ['deltab', 'dt']
        param.copy(self, self.list_param)

        self.list_param = ['j0', 'npy', 'nh']
        grid.copy(self, self.list_param)

        xr, yr = grid.xr, grid.yr
        Lx, Ly = param.Lx, param.Ly

        sigma = 0.02*Lx
        d = 4*sigma/Lx
        vtype = 'cosine'
        self.forc = -ap.vortex(xr, yr, Lx, Ly, 0.5-d/2, 0.2, sigma, vtype)
        self.forc += ap.vortex(xr, yr, Lx, Ly, 0.5+d/2, 0.2, sigma, vtype)

        self.forc *= grid.msk

        self.t0 = 0.
        self.forc_period = 20.

    def add_forcing(self, x, t, dxdt):
        """ add the forcing term on x[0]=the vorticity """
        if t > (self.t0+self.forc_period):

            dxdt[0][:, :] -= self.forc*10.
            self.t0 += self.forc_period
            print('add forcing!!!')
