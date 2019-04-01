# Data for Fig. 1

Run `lj_rigid.py` with argument `--user=0` to generate data for phi=0.1, in `log-v2.dat`. An XML file named
`init-v1_0.10.xml` will be created.  Afterwards, run

```
hoomd hoomdv1-rigid.py
```

with HOOMD-blue 1.3., to generate (incorrect) data from HOOMD-blue 1.x. for comparison, in `log-v1.dat`.

