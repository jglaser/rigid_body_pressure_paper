import signac
import numpy as np
import sys
import BlockAverage

def process_job(job):
    with job:
        print(job)
        try:
            log = np.loadtxt('out.log',skiprows=1)
            sp = job.statepoint()
            density = log[:,3]
            _, density_err = BlockAverage.BlockAverage(density).get_error_estimate()
            potential_energy_LJ = log[:,6]
            _, potential_err_LJ = BlockAverage.BlockAverage(potential_energy_LJ).get_error_estimate()
            potential_energy = log[:,4]
            _, potential_err = BlockAverage.BlockAverage(potential_energy).get_error_estimate()
            return np.mean(density), density_err, np.mean(potential_energy_LJ), potential_err_LJ, np.mean(potential_energy), potential_err
        except Exception as e:
            print(e)
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


if __name__ == '__main__':
    project = signac.get_project()
    jobs = list(project.find_jobs({'r_cut_lj': 1.5}))
    res = [process_job(job) for job in jobs]
    with open('eos_rc1.5.txt', 'w') as f:
        f.write('mean_density err_density mean_U_LJ err_U_LJ mean_U err_U T_kelvin P_bar jobid\n')
        for j,r in zip(jobs,res):
            sp = j.statepoint()
            f.write('{} {} {} {} {} {} {} {} {}\n'.format(r[0],r[1],r[2],r[3],r[4],r[5],sp['T_kelvin'],
                sp['P_bar'], j))

