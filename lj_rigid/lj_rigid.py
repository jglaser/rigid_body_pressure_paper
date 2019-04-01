from hoomd import *
from hoomd import md

import numpy as np
import math

import BlockAverage

import unittest

context.initialize()

p = int(option.get_user()[0])
rho_list = [0.1,0.2]
rho = rho_list[p]
n = 10
a = (1/rho)**(1./3.)

nsteps = 2.5e6

class npt_rigid_validation(unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(unitcell=lattice.sc(a=a),n=n)
        N = len(self.system.particles)

        # generate a system of N=8 AB diblocks
        nl = md.nlist.cell()

        offset = 1
        sigma = 1.0
        #I_sphere = 2/5.*0.25*sigma*sigma
        #I = (I_sphere + offset*offset, I_sphere + offset*offset, I_sphere)

        # for compatibility with HOOMD 1.3's moment of inertia and mass
        I = (1.0,1.0,0)

        for p in self.system.particles:
            p.moment_inertia = I
            p.mass = 2

        # create constituent particle types
        self.system.particles.types.add('const')

        md.integrate.mode_standard(dt=0.005)

        wca = md.pair.lj(r_cut=False, nlist = nl)

        # central particles
        wca.pair_coeff.set('A', self.system.particles.types, epsilon=0, sigma=0, r_cut=False)

        # constituent particle coefficients (WCA)
        wca.pair_coeff.set('const','const', epsilon=1.0, sigma=sigma, r_cut=sigma*2**(1./6))
        wca.set_params(mode="shift")

        # monoatomic rigid body with constituent particle offset from central sphere along z
        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['const'], positions=[(0,0,offset)])
        rigid.create_bodies()

        #rigid.disable()
        self.center = group.rigid_center()


    def test_compare_nvt_npt(self):
        # thermalize
        langevin = md.integrate.langevin(group=self.center,kT=1.0,seed=123)
        langevin.set_gamma('A',1.0)
        run(10000)

        langevin.disable()

        # write out XML file for hoomd v1.3
        from hoomd import deprecated
        deprecated.dump.xml(group=group.all(),
                                  filename='init-v1_{:.2f}.xml'.format(rho),
                                  period=None,
                                  position=True,
                                  type=True,
                                  body=True,
                                  velocity=True)

        # store system data for later re-initialization
        snap = self.system.take_snapshot()

        # run system in NVT
        nvt = md.integrate.nvt(group=self.center,kT=1.0,tau=1.0)
        #nve = md.integrate.nve(group=self.center)

        log = analyze.log(filename='log-v2.dat',quantities=['volume','pressure','pair_lj_energy','pressure_xx','pressure_yy','pressure_zz','pressure_xy','pressure_yz','pressure_xz'],period=100,overwrite=True)

        Pval = []
        def accumulate_P(timestep):
            Pval.append(log.query('pressure'))

        run(10000, profile=True)
        run(nsteps,callback=accumulate_P, callback_period=100)

        block = BlockAverage.BlockAverage(Pval)
        P_avg = np.mean(np.array(Pval))
        i, P_err = block.get_error_estimate()

        context.msg.notice(1,'rho={:.3f} P = {:.5f}+-{:.5f}\n'.format(rho,P_avg,P_err))
        #nve.disable()
        nvt.disable()
        #langevin.disable()

        # re-initialize
        self.system.restore_snapshot(snap)

        # set up NPT at measured pressure from NVT
        npt = md.integrate.npt(group=self.center,P=P_avg,tauP=0.5,kT=1.0,tau=1.0)

        rho_val = []
        N = len(self.center)
        def accumulate_rho(timestep):
            rho_val.append(N/log.query('volume'))

        run(nsteps,callback=accumulate_rho, callback_period=100)

        block = BlockAverage.BlockAverage(rho_val)
        rho_avg = np.mean(np.array(rho_val))
        i, rho_err = block.get_error_estimate()

        context.msg.notice(1,'P={:.5f} rho_avg = {:.5f}+-{:.5f}\n'.format(P_avg,rho_avg,rho_err))

        # max error 0.5 %
        self.assertLessEqual(rho_err/rho_avg,0.005)

        # confidence interval, 0.95 quantile of the normal distribution
        ci = 1.96

        # compare density in NVT vs average density in NPT
        self.assertLessEqual(math.fabs(rho_avg-rho),ci*(rho+rho_err))

    def tearDown(self):
        del self.center
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
