#!/usr/bin/env python

import subprocess
from tssep_data.util.slurm import cmd_to_hpc
cmd = cmd_to_hpc('python -m mpi4py.bench helloworld', mpi=4, time='00:30:00')
print(f'cmd={cmd}')
subprocess.run(cmd, shell=False)
