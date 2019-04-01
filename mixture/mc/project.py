import flow
from flow import environments, directives
from flow import aggregate
from flow import FlowProject

class Project(FlowProject):
    pass

nrank=8

@Project.operation
@aggregate()
@directives(np=lambda *jobs: nrank*len(jobs))
@directives(nranks=lambda *jobs: nrank*len(jobs))
def sample(*jobs):
    import hoomd
    from hoomd import hpmc
    from hoomd import jit
    import numpy as np

    hoomd.context.initialize('--nrank={}'.format(nrank))

    job = jobs[hoomd.comm.get_partition()]
    sp = job.statepoint()

    with job:
        n = sp['n_stoch']
        system = hoomd.init.create_lattice(n=sp['n'],unitcell=hoomd.lattice.unitcell(N=n+1, a1=[2,0,0], a2=[0,2,0], a3=[0,0,n+2], dimensions=3,
            position = [(1,1,1)]+[(0,0,2+i) for i in range(n)], type_name = ['A']+['B']*n))

        r_B = sp['r_B']
        r_C = sp['r_C']
        const_pos = np.array([(-r_C,-r_C,-r_C),(r_C,-r_C,-r_C),(-r_C,r_C,-r_C),(r_C,r_C,-r_C),
                     (-r_C,-r_C,r_C),(r_C,-r_C,r_C),(-r_C,r_C,r_C),(r_C,r_C,r_C)])

        mc = hpmc.integrate.sphere(seed=sp['seed'])
        mc.shape_param.set('A',diameter=0, orientable=True)
        mc.shape_param.set('B',diameter=0, orientable=False)

        code = """
        float rsq = dot(r_ij,r_ij);

        const float eps = {};
        const float sigma_sphere = {};
        const float sigma_cube = {};
        const float rcut_wca_sq = {};

        float r6inv = 1/(rsq*rsq*rsq);

        float sigma;

        if ((type_i == 0 && type_j == 1) || (type_i == 1 && type_j == 0))
            sigma = 0.5*(sigma_sphere+sigma_cube);
        else if (type_i == 1 && type_j == 1)
            sigma = sigma_sphere;
        else if (type_i == 0 && type_j == 0)
            sigma = sigma_cube;

        float sigma2 = sigma*sigma;
        if (rsq < rcut_wca_sq*sigma2)
            {{
            float sigma6 = sigma2*sigma2*sigma2;
            return 4.0*eps*(sigma6*sigma6*r6inv*r6inv-sigma6*r6inv) + eps;
            }}

        return 0.0;
        """.format(1.0, 2*r_B, 2*r_C, 2**(1./3.));

        rcut_max = 2*2**(1./6.)*max(r_C,r_B)
        patch = jit.patch.user_union(mc,r_cut=rcut_max, code=code)

        patch.set_params('A', positions=const_pos, typeids=[0]*len(const_pos))
        patch.set_params('B', positions=[(0,0,0)], typeids=[1])

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

        mc_tune_A = hpmc.util.tune(mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.3,type='A')
        mc_tune_B = hpmc.util.tune(mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.3,type='B')

        boxmc = hpmc.update.boxmc(mc, betaP=sp['P'], seed=sp['seed'])
        boxmc.ln_volume(0.1,weight=1)
        npt_tune = hpmc.util.tune_npt(boxmc, tunables=['dlnV'], target = 0.3, gamma = 1.0)

        log = hoomd.analyze.log(period=100,quantities=['volume','hpmc_d','hpmc_a','hpmc_translate_acceptance','hpmc_rotate_acceptance','hpmc_patch_energy'], filename='out.log')

        hoomd.run(1)
        pos.disable()

        gsd = hoomd.dump.gsd(filename='out.gsd',period=10000,dynamic=['property','momentum'], group=hoomd.group.all())
        restart = hoomd.dump.gsd(filename='restart.gsd',period=10000,truncate=True, group=hoomd.group.all())

        for i in range(20):
            hoomd.run(1000)
            mc_tune_A.update()
            mc_tune_B.update()
            npt_tune.update()
        hoomd.run(1e8)

if __name__ == '__main__':
    Project().main()

