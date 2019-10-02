import physical_validation as pv
import numpy as np

system = pv.data.SystemData(
    natoms=1000,
    nconstraints=0,
    ndof_reduction_tra=3
)

units = pv.data.UnitData(
    kb=1.0,
    energy_str='k_B T',
    energy_conversion=1.0,
    length_str='\sigma',
    length_conversion=1.0,
    volume_str='\sigma^3',
    volume_conversion=1.0,
    temperature_str='',
    temperature_conversion=1.0,
    pressure_str='k_BT \sigma^-3',
    pressure_conversion=1/6.022140857e-2, # relative to GROMACS
    time_str='\\tau',
    time_conversion=1.0
)

P1 = 0.5
P2 = 0.55

ensemble_1 = pv.data.EnsembleData(
    natoms=1000,
    ensemble='NPT',
    pressure=P1,
    temperature=1.0
)

ensemble_2 = pv.data.EnsembleData(
    natoms=1000,
    ensemble='NPT',
    pressure=P2,
    temperature=1.0
)


# load logs for both pressures
log_1 = np.loadtxt('log_P1.dat',skiprows=1)
volume_1 = log_1[:,1]
pe_1 = log_1[:,3]

observables_1 = pv.data.ObservableData(
    volume = volume_1,
    potential_energy = pe_1
)

res_1 = pv.data.SimulationData(
    units=units, ensemble=ensemble_1,
    system=system, observables=observables_1
)

log_2 = np.loadtxt('log_P2.dat',skiprows=1)
volume_2 = log_2[:,1]
pe_2 = log_2[:,3]

observables_2 = pv.data.ObservableData(
    volume = volume_2,
    potential_energy = pe_2
)

res_2 = pv.data.SimulationData(
    units=units, ensemble=ensemble_2,
    system=system, observables=observables_2
)

quantiles = pv.ensemble.check(res_1, res_2, filename='ensemble')
