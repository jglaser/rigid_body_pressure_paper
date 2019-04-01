import hoomd_script
import numpy as np

hoomd_script.context.initialize()

rho = 0.1
system = hoomd_script.init.read_xml('init-v1_{:.2f}.xml'.format(rho))

# pair interactions
nl = hoomd_script.nlist.cell()
wca = hoomd_script.pair.lj(r_cut=False, nlist = nl)
sigma = 1.0
wca.pair_coeff.set('A',['A','const'], epsilon=0, sigma=0, r_cut=False)
wca.pair_coeff.set('const','const', epsilon=1.0, sigma=sigma, r_cut=sigma*2**(1./6))
wca.set_params(mode="shift")

# integrate
hoomd_script.integrate.mode_standard(dt=0.005)

rigid = hoomd_script.group.rigid();
bdnvt = hoomd_script.integrate.bdnvt_rigid(group=rigid, T=1.0, seed=5)
hoomd_script.run(10000)
bdnvt.disable()

print(system.bodies[0])

# dump output
all=hoomd_script.group.all()
log = hoomd_script.analyze.log(filename='log-v1.dat',quantities=['volume','pressure', 'pair_lj_energy', 'pressure_xx','pressure_yy','pressure_zz','pressure_xy','pressure_yz','pressure_xz'],period=100,overwrite=True)

Pval = []
def accumulate_P(timestep):
    Pval.append(log.query('pressure'))

#hoomd_script.integrate.nve_rigid(group=rigid)
hoomd_script.integrate.nvt_rigid(group=rigid,T=1.0,tau=1.0)
hoomd_script.run(2.5e6,callback=accumulate_P,callback_period=100)

import BlockAverage
block = BlockAverage.BlockAverage(Pval)
P_avg = np.mean(np.array(Pval))
i, P_err = block.get_error_estimate()

print('rho={:.3f} P = {:.5f}+-{:.5f}\n'.format(rho,P_avg,P_err))

