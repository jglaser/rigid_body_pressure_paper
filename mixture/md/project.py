import flow
from flow import environments, directives
from flow import aggregate
from flow import FlowProject

class Project(FlowProject):
    pass

@Project.operation
@aggregate()
@directives(np=lambda *jobs: len(jobs))
@directives(nranks=lambda *jobs: len(jobs))
@directives(ngpu=lambda *jobs: len(jobs))
def sample(*jobs):
    import hoomd
    from hoomd import md
    import numpy as np

    hoomd.context.initialize('--mode=gpu --nrank=1')

    job = jobs[hoomd.comm.get_partition()]

    with job:
        sp = job.statepoint()

        n = sp['n_stoch']
        system = hoomd.init.create_lattice(n=sp['n'],unitcell=hoomd.lattice.unitcell(N=n+1, a1=[2,0,0], a2=[0,2,0], a3=[0,0,n+2], dimensions=3,
            position = [(1,1,1)]+[(0,0,2+i) for i in range(n)], type_name = ['A']+['B']*n, moment_inertia = [(1,1,1)]+[(0,0,0)]*n))

        r_B = sp['r_B']
        r_C = sp['r_C']
        const_pos = np.array([(-r_C,-r_C,-r_C),(r_C,-r_C,-r_C),(-r_C,r_C,-r_C),(r_C,r_C,-r_C),
                     (-r_C,-r_C,r_C),(r_C,-r_C,r_C),(-r_C,r_C,r_C),(r_C,r_C,r_C)])

        system.particles.types.add('C')
        rigid = md.constrain.rigid()

        rigid.set_param('A', positions = const_pos, types = ['C']*len(const_pos))
        rigid.create_bodies()

        nl = md.nlist.tree()
        lj = md.pair.lj(nlist = nl, r_cut = 2**(1./6.))
        lj.pair_coeff.set('B','B', sigma=r_B+r_B, epsilon=1,r_cut=(r_B+r_B)*2**(1./6.))
        lj.pair_coeff.set('B','C', sigma=r_B+r_C, epsilon=1,r_cut=(r_B+r_C)*2**(1./6.))
        lj.pair_coeff.set('C','C', sigma=r_C+r_C, epsilon=1,r_cut=(r_C+r_C)*2**(1./6.))
        lj.pair_coeff.set(['A','B','C'], 'A',sigma=0, epsilon=0)

        md.integrate.mode_standard(dt=0.0025)
        center = hoomd.group.rigid_center()
        g = hoomd.group.union(a=center, b=hoomd.group.nonrigid(), name='int_group')

        def format_param_pos(diameters,centers,colors=None):
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
        pos = deprecated.dump.pos(filename='out.pos', period=10000)
        pos.set_def('A',format_param_pos([2*r_C]*len(const_pos), const_pos))
        pos.set_def('B','sphere {} ffff0000'.format(2*r_B))
        pos.set_def('C','sphere 0.0 ffff0000')

        nvt = md.integrate.langevin(kT=1.0, seed=123,group=g)
        hoomd.run(1000)
        nvt.disable()

        pos.disable()

        gsd = hoomd.dump.gsd(filename='out.gsd',period=10000,dynamic=['property','momentum'], group=g)
        restart = hoomd.dump.gsd(filename='restart.gsd',period=10000,truncate=True, group=g)
        log = hoomd.analyze.log(period=100,quantities=['volume','temperature','pressure','potential_energy'], filename='out.log')
     
        npt = md.integrate.npt(P=sp['P'], kT=1.0, tauP=0.5, tau=0.5, group=g)

        hoomd.run(1e8)

if __name__ == '__main__':
    Project().main()
 
