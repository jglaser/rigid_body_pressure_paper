import signac
import math
import numpy as np
import sys
import BlockAverage

def process_job(job):
    with job:
        print(job)
        try:
            log = np.loadtxt('out.log',skiprows=1)
            sp = job.statepoint()
            V_cube = 8*math.pi*4/3*sp['r_C']**3
            V_sphere = math.pi*4/3*sp['r_B']**3
            volume = log[:,1]
            density = V_cube*(sp['n']**3)/volume
            _, density_err = BlockAverage.BlockAverage(density).get_error_estimate()
            return np.mean(density), density_err
        except Exception as e:
            print(e)
            return np.nan, np.nan


if __name__ == '__main__':
    project = signac.get_project()
    jobs = list(project.find_jobs())
    res = [process_job(job) for job in jobs]
    with open('eos_mc.txt', 'w') as f:
        f.write('mean_density err_density P jobid\n')
        for j,r in zip(jobs,res):
            sp = j.statepoint()
            f.write('{} {} {} {}\n'.format(r[0],r[1],sp['P'], j))

