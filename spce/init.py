import signac

project = signac.get_project()

T_list = [300,400,500]
P_list = [1,2,5,10,20,30,40,50,100,200,500,1000,2000,5000]

for T in T_list:
    for P in P_list:
        sp = {'T_kelvin': T, 'P_bar': P, 'n': 16 }
        job = project.open_job(sp)
        job.init()

for T in T_list:
    for P in P_list:
        sp = {'T_kelvin': T, 'P_bar': P, 'n': 16, 'r_cut_lj': 1.5}
        job = project.open_job(sp)
        job.init()
 
