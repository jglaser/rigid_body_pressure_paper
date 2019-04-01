import math

import numpy as np

import rowan

from flow import *
from flow import environments

class Project(FlowProject):
    pass

def done(job):
    return 'done' in job.doc

nrank=1
@Project.operation
@aggregate()
@directives(np=lambda *jobs: nrank*len(jobs))
@directives(ngpu=lambda *jobs: nrank*len(jobs))
@directives(nranks=lambda *jobs: nrank*len(jobs))
def sample(*jobs):
    import hoomd
    from hoomd import md

    hoomd.context.initialize('--nrank=1')

    job = jobs[hoomd.comm.get_partition()]
    with job:
        sp = job.statepoint()
        import os
        restart = os.path.exists(job.fn('restart.gsd'))
        if not restart:
            system = hoomd.init.create_lattice(n=sp['n'],unitcell=hoomd.lattice.sc(a=(1/33.456)**(1./3.),type_name='H2O'))
        else:
            system = hoomd.init.read_gsd('restart.gsd')
        system.particles.types.add('OW')
        system.particles.types.add('H')

        theta_spce = 109.47*math.pi/180
        roh = 0.1
        pos_spce = [(roh*math.sin(theta_spce/2), roh*math.cos(theta_spce/2),0),
            (roh*math.sin(-theta_spce/2), roh*math.cos(theta_spce/2), 0),
            (0,0,0)]
        mass_spce = np.array([1,1,16])
        rcm = np.sum(mass_spce*pos_spce,axis=0)/np.sum(mass_spce)
        pos_spce -= rcm
        types_spce = ['H','H','OW']
        sigma_spce = 0.3166
        eps_spce = 0.650
        m_spce = np.sum(mass_spce)
        I_spce = np.zeros((3,3),dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i == j:
                        I_spce[i,j] += mass_spce[k]*np.linalg.norm(pos_spce[k])**2.0
                    I_spce[i,j] -= mass_spce[k]*pos_spce[k,i]*pos_spce[k,j]

        q = 0.8476
        charges_spce = [q/2,q/2,-q]

        np.random.seed(123)
        for p in system.particles:
            p.mass = m_spce
            p.moment_inertia = (I_spce[0,0],I_spce[1,1],I_spce[2,2])
            #p.orientation = rowan.random.rand(1)

        reduced_T_unit = 0.008314510
        T = sp['T_kelvin']*reduced_T_unit
        f = 138.935458
        rigid = md.constrain.rigid()
        rigid.set_param('H2O',positions=pos_spce, types=types_spce, charges=math.sqrt(f)*np.array(charges_spce))
        rigid.create_bodies()

        my_nlist = md.nlist.cell()
        lj = md.pair.lj(r_cut=sp['r_cut_lj'], nlist=my_nlist)
        lj.pair_coeff.set(['H2O','H'],system.particles.types,epsilon=0,sigma=0,r_cut=False)
        lj.pair_coeff.set('OW','OW',epsilon=eps_spce,sigma=sigma_spce)

        pppm = md.charge.pppm(nlist=my_nlist,group=hoomd.group.all())
        pppm.set_params(Nx=64,Ny=64,Nz=64,order=5,rcut=1.1)

        if not restart:
            def format_param_pos(diameters,centers,colors):
                # build up shape_def string in a loop
                centers = centers
                colors = colors
                N = len(diameters);
                shape_def = 'sphere_union {0} '.format(N);
                if colors is None:
                    # default
                    colors = ["ff5984ff" for c in centers]


                for d,p,c in zip(diameters, centers, colors):
                    shape_def += '{0} '.format(d);
                    shape_def += '{0} {1} {2} '.format(*p);
                    shape_def += '{0} '.format(c);

                return shape_def

            from hoomd import deprecated
            pos = deprecated.dump.pos(filename='out.pos',period=100000,unwrap_rigid=True)
            union_def_W = format_param_pos(diameters = [0.24,0.24,0.3],
                centers = pos_spce, colors = ['ffffffff','ffffffff','ffff0000'])
            pos.set_def('H2O',union_def_W)

        integrator = md.integrate.mode_standard(dt=0.001)

        center = hoomd.group.rigid_center()
        log = hoomd.analyze.log(filename='out.log', quantities=['N','volume','density_kgm3','potential_energy','temperature','pair_lj_energy','pppm_energy','pair_ewald_energy','pressure'],period=100,overwrite=not restart)

        def density(timestep):
            return len(center)*18.0/(6.022*10**23)*(10**24)/system.box.get_volume()

        log.register_callback('density_kgm3', density)

        hoomd.dump.gsd(filename='out.gsd', group=center, period=100000,overwrite=not restart)
        hoomd.dump.gsd(filename="restart.gsd", group=center, truncate=True, period=10000, phase=0)

        reduced_P_unit = 0.0602214
        P = sp['P_bar']*reduced_P_unit

        nvt = md.integrate.nvt(kT=T,group=center,tau=10.0)
        nvt.randomize_velocities(seed=123)
        hoomd.run(10000)
        nvt.disable()

        npt = md.integrate.npt(kT=T,P=P,group=center,tau=0.1,tauP=0.1)

    #    if not restart:
            # currently buggy, gives xyz components independent velocities
    #        npt.randomize_velocities(seed=123)

        if not restart:
            hoomd.run(1e3,profile=True)
            pos.disable()

        hoomd.run(5e6)
        job.doc['done'] = True

if __name__ == '__main__':
    Project().main()
