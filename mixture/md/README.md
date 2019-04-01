Prerequisites:

[HOOMD-blue](https://github.com/glotzerlab/hoomd-blue)

[signac](https://github.com/glotzerlab/signac)

[signac-flow](https://github.com/glotzerlab/signac-flow) (version 0.8.0)

HOOMD-blue needs to be compiled with `-DENABLE_CUDA=ON -D ENABLE_MPI=ON`.

To run the MD simulations, first initialize the signac workspace

```
signac init my_workspace
```

Then initialize the state points

```
python3 init.py
```

and submit, e.g. on a cluster, with

```
python3 project.py submit
```

The job script currently assumes the presence of GPUs.

Generated output will be stored in `out.log` in every job directory under `workspace/`. See the
[signac documentation](https://docs.signac.io/en/latest/) for more information on how to
navigate the workspaces.

Summarize data in a file `eos_mc.txt` with

```
python3 eos.py
```
