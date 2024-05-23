#!/usr/bin/env python3

import sys
import subprocess
import shlex
import datetime
import os
import re

class c:  # noqa
    Color_Off = '\033[0m'  # Text Reset
    Black = '\033[0;30m'  # Black
    Red = '\033[0;31m'  # Red
    Green = '\033[0;32m'  # Green
    Yellow = '\033[0;33m'  # Yellow
    Blue = '\033[0;34m'  # Blue
    Purple = '\033[0;35m'  # Purple
    Cyan = '\033[0;36m'  # Cyan
    White = '\033[0;37m'  # White


def insert(
        argv: 'list[str]',
        short_name,
        long_name,
        default,
        msg=None,
):
    if short_name and short_name in argv:
        pass
    elif long_name and any([a.startswith(long_name) for a in argv]):
        pass
    else:
        # argv.insert(1, default)  # Doesn't support list
        if isinstance(default, str):
            default = [default]
        argv[1:1] = default

        if msg is not None:
            print(msg)

    return argv


def clean_env(cmd):
    # icdiff <(env) <(srun -n 2 -t 30:00 --pty env)

    env = None
    if cmd in ['sbatch', 'salloc']:
        env = os.environ.copy()

        deleted = []
        for k in list(env.keys()):
            if k.startswith('PMIX'):
                del env[k]
                deleted.append(k)

            if k.startswith('OMPI'):
                if k not in [
                    'OMPI_MCA_mtl',
                    'OMPI_MCA_hwloc_base_report_bindings',
                    'OMPI_MCA_btl',
                ]:
                    del env[k]
                    deleted.append(k)

            if k.startswith('SLURM'):
                if k not in [
                    'SLURM_CPU_BIND',
                    'SLURM_QOS',
                    'SLURM_PARTITION',
                    'SLURM_MPI_TYPE',
                    'SLURM_ACCOUNT',
                    'SLURM_TIME_FORMAT',
                    'SLURM_OUTPUT',
                    'SLURM_CONF',
                ]:
                    del env[k]
                    deleted.append(k)

        if deleted:
            print(f'{c.Red}Removed from env: {" ".join(deleted)}{c.Color_Off}')

    elif cmd in ['srun']:
        pass
    else:
        raise ValueError(cmd)

    return env


if __name__ == '__main__':
    cmd = sys.argv[0].split('/')[-1]

    if cmd == 'srun.py':
        cmd = 'srun'
        if 'mpirun' in sys.argv:
            raise RuntimeError(
                '"srun ... mpirun ..."\n'
                'executes quadratic number of processes\n'
                'Probably you wanted\n'
                '"salloc ... mpirun ..."'
            )
    elif cmd == 'salloc.py':
        cmd = 'salloc'
    elif cmd == 'sbatch.py':
        cmd = 'sbatch'
    else:
        raise AssertionError(sys.argv)

    argv = [cmd] + sys.argv[1:]

    # nodelist: These nodes will be included + maybe other nodes
    # (i.e. not a whitelist).
    # i.e. the job should run at least on these nodes.
    # insert(argv, '-w', '--nodelist', '--nodelist=node13')

    exclude = [
    ]

    if exclude:
        exclude = ','.join(sorted(set(exclude)))
        exclude = f'--exclude={exclude}'
        insert(argv, '-x', '--exclude', exclude)

    # insert(argv, '-p', '--partition', '--partition=normal')
    # insert(argv, '', '--mem', ['--mem', '2G'])
    # insert(argv, '-t', '', ['-t', '8:00:00'])

    # if cmd == 'sbatch' and 'SBATCH_OUTPUT' in os.environ:
    #     insert(argv, '-o', '--output', f'--output={os.environ["SBATCH_OUTPUT"]}',
    #            msg='SLURM of Noctua ignores env variable SBATCH_OUTPUT, hence add it manually.')

    # if '--wrap' in argv:
    #     m = re.match('(?:srun (?:--unbuffered )?)?python -m ([^ ]+).*', argv[argv.index('--wrap') + 1])
    #     if m:
    #         name, = m.groups()
    #         insert(argv, '-J', '--job-name', ['--job-name', name])

    env = clean_env(cmd)

    print(c.Green, '$ ', shlex.join(argv), c.Color_Off,
          sep='', flush=True)

    # with open(".history", "a") as fd:
    #     print(shlex.join(argv), ' #', datetime.datetime.today().strftime('%Y.%m.%d %H:%M:%S'), file=fd)

    cp = subprocess.run(argv, env=env, check=False)
    sys.exit(cp.returncode)
