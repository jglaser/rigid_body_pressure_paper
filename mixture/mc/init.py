import signac
import numpy as np
P_list = np.arange(0.5,10.0,0.5)
for P in P_list:
    sp = {'P': P, 'r_B': 0.7*0.5, 'r_C': 0.5, 'n': 10, 'seed': 123, 'n_stoch': 3}
    project = signac.get_project()
    job = project.open_job(sp)
    job.init()
    print(job)
