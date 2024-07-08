import operator
import os
import shutil
import dataclasses
import datetime
import re
import subprocess
import math
import shlex
import psutil


from tssep_data.util.bash import location


@dataclasses.dataclass
class SlurmResources:
    cpus: int  # cpus per task/MPI process
    mem: 'str | None' = None  # e.g. '25GB'  # mem per task/MPI process
    gpus: 'int | None' = 0  # gpus per job, --gpus-per-task is for gpus per task/MPI process
    gputype: str = None
    mpi: int = 1  # Number of tasks/MPI processes. Note: The total number of requested cpus is cpus times mpi.
    cpu_partition: str = 'cpu'
    gpu_partition: str = 'gpu'
    time: 'str | int | None' = None
    # qos: [str] = {'MERL': None, 'Noctua2': 'cont', 'unknown': None}[location]

    # Depending on the installation, either mpirun or srun is used to
    # launch an mpi script.
    mpi_cmd: str = 'srun'
    job_name: str = None

    if location == 'MERL':
        cpu_partition = 'cpu,cpu_extra'
        gpu_partition = 'gpu,gpu_24'
        mpi_cmd = 'mpirun'
    elif location == 'Noctua2':
        gputype = 'a100'
        cpu_partition = 'normal'
        time = '8 hours'
        mpi_cmd = 'srun.py'
    elif location == 'Noctua1':
        # gputype = 'rtx2080ti'
        gputype = 'a40'
        cpu_partition = 'normal'
        time = '8 hours'
        mpi_cmd = 'srun.py'
    elif location == 'unknown':
        pass
    else:
        raise NotImplementedError(location)

    def __post_init__(self):
        assert int(self.cpus) > 0, self.cpus
        assert int(self.mpi) > 0, self.mpi
        assert int(self.gpus) >= 0, self.gpus
        # if isinstance(self.time, int):
        #     if self.time < 60 * 60:
        #         self.time = ''

    @staticmethod
    def _timestr_to_seconds(timestr):
        """
        Slurm formats (see https://slurm.schedmd.com/sbatch.html):
            "minutes",  # Here, not supported.
            "minutes:seconds",
            "minutes:seconds",
            "hours:minutes:seconds",
            "days-hours",
            "days-hours:minutes",
            "days-hours:minutes:seconds",

        Additional human readable timeformats are supported by pytimeparse,
        e.g.:
            "1d", "1h", "1m"


        >>> SlurmResources._timestr_to_seconds('1m')
        60
        >>> SlurmResources._timestr_to_seconds('1:0')
        60
        >>> SlurmResources._timestr_to_seconds('1:0:0')
        3600
        >>> SlurmResources._timestr_to_seconds('1d')
        86400
        >>> SlurmResources._timestr_to_seconds('1-1')
        90000
        >>> SlurmResources._timestr_to_seconds('1-1:1')
        90000
        >>> SlurmResources._timestr_to_seconds('1-1:1:1')
        90000

        """

        # Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds".

        m = re.match('(\d+):(\d+):(\d+)', timestr)
        if m:
            hours, minutes, seconds = m.groups()
            return int(hours) * 3600 + int(minutes) * 60 + int(seconds)

        m = re.match('(\d+):(\d+)', timestr)
        if m:
            minutes, seconds = m.groups()
            return int(minutes) * 60 + int(seconds)

        m = re.match('(\d+)-(\d+)', timestr)
        if m:
            days, hours = m.groups()
            return int(days) * 24 * 3600 + int(hours) * 3600

        m = re.match('(\d+)-(\d+):(\d+)', timestr)
        if m:
            days, hours, minutes = m.groups()
            return int(days) * 24 * 3600 + int(hours) * 3600 + int(minutes) * 60

        m = re.match('(\d+)-(\d+):(\d+):(\d+)', timestr)
        if m:
            days, hours, minutes, seconds = m.groups()
            return int(days) * 24 * 3600 + int(hours) * 3600 + int(minutes) * 60 + int(seconds)

        import pytimeparse
        seconds = pytimeparse.timeparse.timeparse(timestr)
        if seconds is not None:
            return seconds

        raise ValueError(timestr)

    def _time(self, style='slurm'):
        """
        >>> SlurmResources(1, time=300)._time()
        '0:5:0'
        >>> SlurmResources(1, time='300m')._time()
        '5:0:0'
        >>> SlurmResources(1, time='5h')._time()
        '5:0:0'
        >>> SlurmResources(1, time='5h')._time(style='s')
        18000
        >>> SlurmResources(1, time='5h')._time(style='h')
        5.0

        >>> SlurmResources(1, time='60')._time(style='h')
        >>> SlurmResources(1, time='1:0:0')._time(style='h')
        1.0
        >>> SlurmResources(1, time='1-0')._time(style='h')
        >>> SlurmResources(1, time='1-1:1')._time(style='h')
        >>> SlurmResources(1, time='1-1:1:1')._time(style='h')
        """
        # SLURM time formats are messed up.
        # Use pytimeparse with reasonable formats.
        #
        # https://slurm.schedmd.com/sbatch.html:
        # Acceptable time formats include
        # "minutes": Bad idea. Default is normally second. Not supported.
        # "minutes:seconds"
        # "hours:minutes:seconds": Not supported by pytimeparse.
        # "days-hours": Not supported by pytimeparse.
        # "days-hours:minutes": Conflict with "minutes:seconds"
        # "days-hours:minutes:seconds"
        assert self.time is not None, self.time
        if isinstance(self.time, str):
            seconds = self._timestr_to_seconds(self.time)
        else:
            seconds = self.time

        if style == 'slurm':
            minutes, seconds = seconds // 60, seconds % 60
            hours, minutes = minutes // 60, minutes % 60
            return f'{hours}:{minutes}:{seconds}'
        elif style in (int, 'seconds', 'second', 'sec', 's'):
            return seconds
        elif style in (float, 'hours', 'hour', 'h'):
            return seconds / 3600
        else:
            raise ValueError(style)

    def _mem(self):
        """
        >>> SlurmResources(mem='25GB')._mem()
        ['--mem', '25GB']
        >>> SlurmResources(mem='25GB', cpus=4)._mem()
        ['--mem', '25GB']
        >>> SlurmResources(mem='25GB', cpus=4, mpi=2)._mem()  # SLURM doesn't support float, hence round 6.25 to 7
        ['--mem-per-cpu', '7GB']
        """
        if f'{self.mpi}' == '1':
            return ['--mem', f'{self.mem}']
        else:
            # Slurm has no --mem-per-cpu, hence
            m = re.fullmatch('(\d+)([tTgGmMkK]?[bB]?i?t?)', self.mem)
            assert m, (m, self.mem)
            value, unit = m.groups()

            value = math.ceil(float(value) / int(self.cpus))

            return ['--mem-per-cpu', f'{value}{unit}']

    def to_list(self):
        """
        >>> print(SlurmResources(6, '4GB').to_str())
        -N 1 -n 1 -c 6 --mem 4GB -p cpu
        >>> print(SlurmResources(6, '25GB', 1).to_str())
        -N 1 -n 1 -c 6 --mem 25GB -p gpu --gres=gpu:1
        >>> print(SlurmResources(1, None, 0, mpi=20).to_str())
        -n 20 -c 1 -p cpu
        >>> print(SlurmResources(1, None, 0, mpi=20).to_list())
        ['-n', '20', '-c', '1', '-p', 'cpu']
        >>> print(SlurmResources(1, time=48*60*60).to_str())
        -N 1 -n 1 -c 1 --time 48:0:0 -p cpu
        >>> print(SlurmResources(1, time='2 days').to_str())
        -N 1 -n 1 -c 1 --time 48:0:0 -p cpu
        >>> print(SlurmResources(cpus=1, mpi=2, gpus=1).to_str())
        -N 1 -n 2 -c 1 -p gpu --gres=gpu:1
        >>> print(SlurmResources(cpus=1, mpi=2, gpus=1, gpu_partition=None).to_str())
        -N 1 -n 2 -c 1 --gres=gpu:1

        """
        s = []
        if f'{self.mpi}' == '1':
            s += [
                '-N', '1',
                # Number of nodes. Can be used to control scatter.
                # Rarely useful.
                # Bug: I had one task that was scattered across two nodes.
                #      Hopefully this will prevent the scatter.
                #      i.e.: GPU job request 8 CPUs, but gets 6 CPUs on one
                #      Node and 6 CPUs on other node.
                #      Was not able to reproduce this, probably depends on
                #      cluster utilization.
            ]
        if f'{self.mpi}' != '1' and self.gpus > 0:
            # gpus is gpus per job and all tasks share the gpu.
            # No idea, how a scattering would work, hence limit the job
            # to one node. Also I have no idea, when a scatter would be
            # interesting (Maybe 1 GPU per task, but the the --gpu option is
            # wrong.)
            s += ['-N', '1']

        s += [
            '-n', f'{self.mpi}',
            '-c', f'{self.cpus}',
        ]
        if self.mem:
            s += self._mem()
        if self.time:
            s += ['--time', f'{self._time()}']
        # if self.qos:
        #     s += ['--qos', f'{self.qos}']
        if self.gpus:
            assert isinstance(self.gpus, int), (type(self.gpus), self.gpus)

            if self.gpu_partition is not None:
                s += ['-p', f'{self.gpu_partition}']

            if self.gputype is None:
                s += [f'--gres=gpu:{self.gpus}']
            else:
                s += [f'--gres=gpu:{self.gputype}:{self.gpus}']
        else:
            if self.cpu_partition is not None:
                s += ['-p', f'{self.cpu_partition}']

        if self.job_name:
            s += ['--job-name', f'{self.job_name}']
        return s

    def to_str(self):
        return ' '.join(self.to_list())

    @property
    def sbatch_executable(self):
        if shutil.which('sbatch'):
            if shutil.which('sbatch.py'):
                # I use a python wrapper around sbatch/srun/salloc to inject
                # some options.
                # e.g. --exclude to ignore some nodes.
                return 'sbatch.py'
            else:
                return 'sbatch'
        else:
            raise RuntimeError('Cannot find sbatch executable to run a slurm job.')

    @property
    def salloc_executable(self):
        if shutil.which('salloc'):
            if shutil.which('salloc.py'):
                # I use a python wrapper around sbatch/srun/salloc to inject
                # some options.
                # e.g. --exclude to ignore some nodes.
                return 'salloc.py'
            else:
                return 'salloc'
        else:
            raise RuntimeError('Cannot find salloc executable to run a slurm job.')

    @property
    def srun_executable(self):
        if shutil.which('srun'):
            if shutil.which('srun.py'):
                # I use a python wrapper around sbatch/srun/salloc to inject
                # some options.
                # e.g. --exclude to ignore some nodes.
                return 'srun.py'
            else:
                return 'srun'
        else:
            raise RuntimeError('Cannot find srun executable to run a slurm job.')

    def to_sbatch_str(self):
        return f'{self.sbatch_executable} {self.to_str()}'

    def to_salloc_str(self):
        return f'{self.salloc_executable} {self.to_str()}'

    def to_srun_str(self):
        return f'{self.srun_executable} {self.to_str()}'


def shutdown_soon(debug=False):
    """
    >>> shutdown_soon(debug=True)  # doctest: +ELLIPSIS
    $ scontrol update jobid=None TimeLimit=...
    """
    import dlp_mpi

    if not dlp_mpi.IS_MASTER:
        return

    if not debug:
        SLURM_JOBID = os.environ.get('SLURM_JOBID')

    if debug or SLURM_JOBID:
        env = os.environ.copy()
        _ = env.pop('SLURM_TIME_FORMAT', '')

        if debug:
            # stdout = dummy
            stdout = ' StartTime=2023-01-10T10:35:44 '
            SLURM_JOBID = None
        else:
            cp = subprocess.run(
                ['scontrol', 'show', 'jobid', f'{SLURM_JOBID}'],
                stdout=subprocess.PIPE, env=env, universal_newlines=True)
            stdout = cp.stdout

        m = re.search('StartTime=([^ ]+)', stdout)
        if m:
            StartTime, = m.groups()

            StartTime = datetime.datetime.fromisoformat(StartTime)

            delta = datetime.datetime.now() - StartTime + datetime.timedelta(minutes=10)

            minutes = int(delta.total_seconds() // 60) + 5

            cmd = ['scontrol', 'update', f'jobid={SLURM_JOBID}', f'TimeLimit={minutes}']
            print(f'$', shlex.join(cmd))
            if debug:
                pass
            else:
                subprocess.run(cmd, env=env)
            return
        else:
            print(stdout)
            reason = 'Cannot predict TimeLimit, because StartTime not found.'
    else:
        reason = 'Missing SLURM_JOBID in env.'

    print(f'SLURM shutdown_soon failed, i.e. kept TimeLimit of SLURM JOB (if its a slurm job). Reason: {reason}')


def set_memory_limit(mem_per_cpu_in_GB):
    import dlp_mpi

    # Note: This is virtual memory
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # 8 * 1024**3 : 8 GBytes
    soft = (
                   mem_per_cpu_in_GB * int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
           ) * 1024 ** 3

    if dlp_mpi.IS_MASTER:
        print(f'### Limit memory usage to: {soft / 1024 ** 3} GiBytes ###')
        print(
            f'$SLURM_CPUS_PER_TASK={os.environ.get("SLURM_CPUS_PER_TASK", 1)}')
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


@dataclasses.dataclass
class CMD2HPC:
    cpus: int  # cpus per task/MPI process
    mem: 'str | None' = None  # e.g. '25GB'  # mem per task/MPI process
    gpus: 'int' = 0  # gpus per job, --gpus-per-task is for gpus per task/MPI process
    gputype: 'str | None' = None
    mpi: 'int | str' = 1  # Number of tasks/MPI processes. Note: The total number of requested cpus is cpus times mpi.
    cpu_partition: str = None
    gpu_partition: str = None
    time: 'str | int | None' = None
    job_name: 'str | None' = None
    mpi_cmd: 'callable' = SlurmResources.mpi_cmd
    # force_local: 'bool' = False
    pty: 'bool' = False
    block: 'bool' = True
    shell: 'bool' = False
    shell_wrap: 'bool' = False
    backend: 'str' = None  # slurm, mpirun

    def __post_init__(self):
        self.mpi = int(self.mpi)
        self.sr = SlurmResources(
            cpus=self.cpus, mem=self.mem, gpus=self.gpus, mpi=self.mpi,
            gputype=self.gputype,
            cpu_partition=self.cpu_partition,
            gpu_partition=self.gpu_partition,
            time=self.time, job_name=self.job_name,
            mpi_cmd=self.mpi_cmd,
        )

        if self.backend is None:
            try:
                _ = self.sr.salloc_executable
                self.backend = 'slurm'
            except RuntimeError:
                self.backend = 'mpirun'

        if self.backend == 'mpirun' and self.sr.mpi > 1:
            self._apply_local_limits()

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

    def _apply_local_limits(self):
        import psutil
        import humanfriendly  # ESPnet dependency, hence anyway installed
        total_mem = psutil.virtual_memory().available
        if self.sr.mem is not None:
            max_mem_limit_jobs = total_mem // humanfriendly.parse_size(self.sr.mem)
            if max_mem_limit_jobs < self.sr.mpi:
                if max_mem_limit_jobs == 0:
                    print(f'ERROR: Not enough memory for a single job (requested {self.sr.mpi}): {self.sr.mem}*{self.sr.mpi}>{humanfriendly.format_size(total_mem)}.\n'
                          f'WARNING: Ignoring above error and set the number of jobs to one.')
                    max_mem_limit_jobs = 1
                else:
                    print(f'WARNING: Not enough memory for {self.sr.mpi} jobs: {self.sr.mem}*{self.sr.mpi}>{humanfriendly.format_size(total_mem)}. '
                          f'Limiting to {max_mem_limit_jobs} jobs.')
                self.sr.mpi = max_mem_limit_jobs
        max_cpu_limit_jobs = max([
            1,
            len(os.sched_getaffinity(0)) // self.cpus,
            # psutil.cpu_count(logical=False),  # some mpi implementations allow only number physical cores
        ])
        if max_cpu_limit_jobs < self.sr.mpi:
            print(f'WARNING: Not enough cpus (#cpus={len(os.sched_getaffinity(0))}) '
                  f'for {self.sr.mpi} jobs. '
                  f'Limiting to {max_cpu_limit_jobs} jobs.')
            self.sr.mpi = max_cpu_limit_jobs
        self.mpi = self.sr.mpi

    def __call__(self, cmd):
        if self.backend == 'slurm':
            return self._slurm(cmd)
        elif self.backend == 'mpirun':
            return self._mpirun(cmd)
        else:
            raise ValueError(self.backend)

    class _CMD:
        def __init__(self, cmd, shell_wrap):
            if shell_wrap:
                self.str = bash_wrap(cmd, shell=True)
                self.lst = bash_wrap(cmd, shell=False)
            else:
                self.str = cmd if isinstance(cmd, str) else shlex.join(cmd)
                self.lst = shlex.split(cmd) if isinstance(cmd, str) else cmd

    def _slurm(self, cmd):
        slurm_cmd = self.sr.salloc_executable if self.block else self.sr.sbatch_executable
        if location not in ['Noctua2', 'Noctua1', 'unknown', 'MERL']:
            raise NotImplementedError(location)

        cmd = self._CMD(cmd, shell_wrap=self.shell_wrap)

        mpi_cmd = self.sr.mpi_cmd
        if self.sr.mpi == 1 and self.sr.mpi_cmd in ['mpirun', 'mpiexec']:
            mpi_cmd = ''

        if self.block:
            mpi_cmd = [mpi_cmd] if mpi_cmd else []
            if self.shell:
                return f'{shlex.join([slurm_cmd] + self.sr.to_list() + mpi_cmd)} {cmd.str}'
            else:
                return [slurm_cmd] + self.sr.to_list() + mpi_cmd + cmd.lst
        else:
            if self.shell:
                wrap = f'{mpi_cmd} {cmd.str}'
            else:
                if mpi_cmd:
                    wrap = f'{mpi_cmd} {cmd.str}'
                else:
                    wrap = cmd.str
            cmd = [slurm_cmd] + self.sr.to_list() + ['--wrap', wrap]
            return shlex.join(cmd) if self.shell else cmd

    def _mpirun(self, cmd):
        cmd = self._CMD(cmd, shell_wrap=self.shell_wrap)
        if self.sr.mpi == 1:
            return cmd.str if self.shell else cmd.lst

        # From OpenMPI help text, when request to many mpi processes:
        #     In all the above cases, if you want Open MPI to default to the number
        #     of hardware threads instead of the number of processor cores, use the
        #     --use-hwthread-cpus option.
        # MPICH has also this option. Hence, activate it by default.

        if self.shell:
            return f'mpirun --use-hwthread-cpus -np {self.sr.mpi} {cmd.str}'
        else:
            return ['mpirun', '--use-hwthread-cpus', '-np', f'{self.sr.mpi}'] + cmd.lst


def cmd_to_hpc_v2(
        cmd,
        cpus: int = 1,  # cpus per task/MPI process
        mem: 'str | None' = None,  # e.g. '25GB'  # mem per task/MPI process
        gpus: 'int' = 0,  # gpus per job, --gpus-per-task is for gpus per task/MPI process
        gputype: 'str | None' = None,
        mpi: 'int | str' = 1,  # Number of tasks/MPI processes. Note: The total number of requested cpus is cpus times mpi.
        cpu_partition: str = None,
        gpu_partition: str = None,
        time: 'str | int | None' = None,
        job_name = None,
        mpi_cmd = SlurmResources.mpi_cmd,
        force_local=False,
        pty=False,
        block=True,
        shell=False,
        shell_wrap=False,
):
    """
    Convert the command to a command that can be run on a HPC cluster,
    if slurm is installed. Otherwise, the command is modified to run locally,
    i.e.:
     - Single process (i.e. mpi==1): Return the command unchanged
     - Multi process (i.e. mpi>1): Return command prefixed by mpirun.
       Note: The requested resources will be adjusted to the available
             resources. e.g.
              - requesting 10 process with each 10GB memory, but only 32 GB
                are free, it will be reduced to 3 processes.
              - requesting 10 process but the machine has only 8 cores, it will
                be reduced to 8 processes.

    Args:
        cmd:
        cpus:
        mem:
        gpus:
        gputype:
        mpi:
        cpu_partition:
        gpu_partition:
        time:
        job_name:
        mpi_cmd:
        force_local:
        pty:
        block:
        shell:
        shell_wrap:

    Returns:


    >>> import mock, contextlib, psutil, humanfriendly


    # For doctest reproducibility, we mock the available memory and cpus
    >>> @contextlib.contextmanager
    ... def mock_local():
    ...     t = type(psutil.virtual_memory())
    ...     with mock.patch(f'{t.__module__}.{t.__qualname__}.available', new=humanfriendly.parse_size('11.88GB')):
    ...         with mock.patch('os.sched_getaffinity', new=lambda x: [0, 1, 2, 3]):
    ...             with mock.patch('shutil.which', new=lambda x: None):
    ...                 yield
    >>> mock_slurm = mock.patch('shutil.which', new=lambda x: x)

    # Normal command, that gets executed on the local machine
    >>> with mock_local():
    ...     shlex.join(cmd_to_hpc_v2('echo hello', 1, '4GB', 0, None, 1, 'cpu', 'gpu', '1:0:0'))
    'echo hello'

    # Request parallel execution will add mpirun
    >>> with mock_local():
    ...     cmd_to_hpc_v2('echo hello', 1, '0.1GB', 0, None, 4, 'cpu', 'gpu', '1:0:0', shell=True)
    'mpirun --use-hwthread-cpus -np 4 echo hello'

    # Request too much cpus for the local machine will reduce the number of processes.
    >>> with mock_local():
    ...      shlex.join(cmd_to_hpc_v2('echo hello', 2, '0.1GB', 0, None, 10, 'cpu', 'gpu', '1:0:0'))
    WARNING: Not enough cpus (#cpus=4) for 10 jobs. Limiting to 2 jobs.
    'mpirun --use-hwthread-cpus -np 2 echo hello'

    # Request too much memory for the local machine will reduce the number of processes.
    >>> with mock_local():
    ...      shlex.join(cmd_to_hpc_v2('echo hello', 1, '10GB', 0, None, 2, 'cpu', 'gpu', '1:0:0'))
    WARNING: Not enough memory for 2 jobs: 10GB*2>11.88 GB. Limiting to 1 jobs.
    'echo hello'

    # Request too much memory for the local machine for a single job: Print a warning.
    >>> with mock_local():
    ...     shlex.join(cmd_to_hpc_v2('echo hello', 1, '100GB', 0, None, 2, 'cpu', 'gpu', '1:0:0'))
    ERROR: Not enough memory for a single job (requested 2): 100GB*2>11.88 GB.
    WARNING: Ignoring above error and set the number of jobs to one.
    'echo hello'

    # Shell && is preserved, whell shell is True or (block is False and slurm is used)
    >>> with mock_local():
    ...     cmd_to_hpc_v2('echo hello && echo world', shell=True)
    'echo hello && echo world'
    >>> with mock_local():
    ...     cmd_to_hpc_v2('echo hello && echo world', shell=True, mpi=2)
    'mpirun --use-hwthread-cpus -np 2 echo hello && echo world'
    >>> with mock_slurm:
    ...     print(cmd_to_hpc_v2('echo hello && echo world', shell=True))
    salloc.py -N 1 -n 1 -c 1 srun echo hello && echo world
    >>> with mock_slurm:
    ...     print(cmd_to_hpc_v2('echo hello && echo world', shell=True, block=False))
    sbatch.py -N 1 -n 1 -c 1 --wrap 'srun echo hello && echo world'
    >>> with mock_slurm:
    ...     print(cmd_to_hpc_v2('echo hello && echo world', shell=False, block=False))
    ['sbatch.py', '-N', '1', '-n', '1', '-c', '1', '--wrap', 'srun echo hello && echo world']
    >>> with mock_slurm:
    ...     print(cmd_to_hpc_v2('echo hello && echo world', shell=True, block=False, gpus=1))
    sbatch.py -N 1 -n 1 -c 1 -p None --gres=gpu:1 --wrap 'srun echo hello && echo world'


    """
    return CMD2HPC(
        cpus=cpus, mem=mem, gpus=gpus, gputype=gputype, mpi=mpi,
        cpu_partition=cpu_partition, gpu_partition=gpu_partition, time=time,
        job_name=job_name, mpi_cmd=mpi_cmd, pty=pty, block=block, shell=shell,
        shell_wrap=shell_wrap,
    )(cmd)





def cmd_to_hpc(
        cmd,
        cpus: int = 1,  # cpus per task/MPI process
        mem: 'str | None' = None,  # e.g. '25GB'  # mem per task/MPI process
        gpus: 'int' = 0,  # gpus per job, --gpus-per-task is for gpus per task/MPI process
        gputype: 'str | None' = None,
        mpi: 'int | str' = 1,  # Number of tasks/MPI processes. Note: The total number of requested cpus is cpus times mpi.
        cpu_partition: str = None,
        gpu_partition: str = None,
        time: 'str | int | None' = None,
        job_name = None,
        mpi_cmd = SlurmResources.mpi_cmd,
        force_local=False,
        pty=False,
        block=True,
        shell=False,
        shell_wrap=False,
):
    """
    Convert the command to a command that can be run on a HPC cluster,
    if slurm is installed. Otherwise, the command is modified to run locally,
    i.e.:
     - Single process (i.e. mpi==1): Return the command unchanged
     - Multi process (i.e. mpi>1): Return command prefixed by mpirun.
       Note: The requested resources will be adjusted to the available
             resources. e.g.
              - requesting 10 process with each 10GB memory, but only 32 GB
                are free, it will be reduced to 3 processes.
              - requesting 10 process but the machine has only 8 cores, it will
                be reduced to 8 processes.

    
    Args:
        cmd: 
        cpus: Number of cpus per task/MPI process
        mem: 
        gpus: 
        gputype: 
        mpi: 
        cpu_partition: 
        gpu_partition: 
        time: 
        force_local: 
        block:
            If True, the generated command will be blocking and wait for "cmd"
            to finish.
            If False and slurm is used, the "cmd" will the submitted to slurm
            and the generated command will return immediately.
        shell: 

    Returns:

    >>> import mock, contextlib, psutil, humanfriendly


    # For doctest reproducibility, we mock the available memory and cpus
    >>> @contextlib.contextmanager
    ... def mock_local():
    ...     t = type(psutil.virtual_memory())
    ...     with mock.patch(f'{t.__module__}.{t.__qualname__}.available', new=humanfriendly.parse_size('11.88GB')):
    ...         with mock.patch('os.sched_getaffinity', new=lambda x: [0, 1, 2, 3]):
    ...             with mock.patch('shutil.which', new=lambda x: None):
    ...                 yield
    >>> mock_slurm = mock.patch('shutil.which', new=lambda x: x)

    # Normal command, that gets executed on the local machine
    >>> with mock_local():
    ...     shlex.join(cmd_to_hpc('echo hello', 1, '4GB', 0, None, 1, 'cpu', 'gpu', '1:0:0'))
    'echo hello'

    # Request parallel execution will add mpirun
    >>> with mock_local():
    ...     cmd_to_hpc('echo hello', 1, '0.1GB', 0, None, 4, 'cpu', 'gpu', '1:0:0', shell=True)
    'mpirun -np 4 echo hello'

    # Request too much cpus for the local machine will reduce the number of processes.
    >>> with mock_local():
    ...      shlex.join(cmd_to_hpc('echo hello', 2, '0.1GB', 0, None, 10, 'cpu', 'gpu', '1:0:0'))
    WARNING: Not enough cpus (#cpus=4) for 10 jobs. Limiting to 2 jobs.
    'mpirun -np 2 echo hello'

    # Request too much memory for the local machine will reduce the number of processes.
    >>> with mock_local():
    ...      shlex.join(cmd_to_hpc('echo hello', 1, '10GB', 0, None, 2, 'cpu', 'gpu', '1:0:0'))
    WARNING: Not enough memory for 2 jobs: 10GB*2>11.88 GB. Limiting to 1 jobs.
    'mpirun -np 1 echo hello'

    # Request too much memory for the local machine for a single job: Print a warning.
    >>> with mock_local():
    ...     shlex.join(cmd_to_hpc('echo hello', 1, '100GB', 0, None, 2, 'cpu', 'gpu', '1:0:0'))
    WARNING: Not enough memory for a single job (requested2): 100GB*2>11.88 GB. Try it anyway with 1 job.
    'mpirun -np 1 echo hello'

    # Shell && is preserved, whell shell is True or (block is False and slurm is used)
    >>> with mock_local():
    ...     cmd_to_hpc('echo hello && echo world', shell=True)
    'echo hello && echo world'
    >>> with mock_local():
    ...     cmd_to_hpc('echo hello && echo world', shell=True, mpi=2)
    'mpirun -np 2 echo hello && echo world'
    >>> with mock_slurm:
    ...     print(cmd_to_hpc('echo hello && echo world', shell=True))
    salloc.py -N 1 -n 1 -c 1 srun echo hello && echo world
    >>> with mock_slurm:
    ...     print(cmd_to_hpc('echo hello && echo world', shell=True, block=False))
    sbatch.py -N 1 -n 1 -c 1 --wrap 'srun echo hello && echo world'
    >>> with mock_slurm:
    ...     print(cmd_to_hpc('echo hello && echo world', shell=False, block=False))
    ['sbatch.py', '-N', '1', '-n', '1', '-c', '1', '--wrap', 'srun echo hello && echo world']

    """
    mpi = int(mpi)

    sr = SlurmResources(cpus=cpus, mem=mem, gpus=gpus, mpi=mpi,
                        gputype=gputype,
                        cpu_partition=cpu_partition,
                        gpu_partition=gpu_partition,
                        time=time, job_name=job_name,
                        mpi_cmd=mpi_cmd,
                        )
    try:
        _ = sr.salloc_executable
        slurm = True
    except RuntimeError:
        slurm = False

    if force_local:
        slurm = False

    class CMD:
        def __init__(self, cmd):
            if shell_wrap:
                self.str = bash_wrap(cmd, shell=True)
                self.lst = bash_wrap(cmd, shell=False)
            else:
                self.str = cmd if isinstance(cmd, str) else shlex.join(cmd)
                self.lst = shlex.split(cmd) if isinstance(cmd, str) else cmd

    cmd = CMD(cmd)

    if slurm:
        slurm_cmd = sr.salloc_executable if block else sr.sbatch_executable
        if location in ['Noctua2', 'Noctua1', 'unknown', 'MERL']:
            mpi_cmd = sr.mpi_cmd
            if sr.mpi == 1 and sr.mpi_cmd in ['mpirun', 'mpiexec']:
                mpi_cmd = ''

            def to_cmd(cmd: CMD):
                nonlocal mpi_cmd
                if block:
                    mpi_cmd = [mpi_cmd] if mpi_cmd else []
                    if shell:
                        return f'{shlex.join([slurm_cmd] + sr.to_list() + mpi_cmd)} {cmd.str}'
                    else:
                        return [slurm_cmd] + sr.to_list() + mpi_cmd + cmd.lst
                else:
                    if shell:
                        wrap = f'{mpi_cmd} {cmd.str}'
                    else:
                        if mpi_cmd:
                            wrap = f'{mpi_cmd} {cmd.str}'
                        else:
                            wrap = cmd.str
                    cmd = [slurm_cmd] + sr.to_list() + ['--wrap', wrap]
                    return shlex.join(cmd) if shell else cmd
        else:
            raise NotImplementedError(location)
    else:
        if sr.mpi > 1:
            import psutil
            import humanfriendly  # ESPnet dependency, hence anyway installed
            total_mem = psutil.virtual_memory().available
            if sr.mem is not None:
                max_mem_limit_jobs = total_mem // humanfriendly.parse_size(sr.mem)
                if max_mem_limit_jobs < sr.mpi:
                    if max_mem_limit_jobs == 0:
                        print(f'ERROR: Not enough memory for a single job (requested{sr.mpi}): {sr.mem}*{sr.mpi}>{humanfriendly.format_size(total_mem)}. '
                              f'WARNING: Ignoring above error and set the number of jobs to one.')
                        max_mem_limit_jobs = 1
                    else:
                        print(f'WARNING: Not enough memory for {sr.mpi} jobs: {sr.mem}*{sr.mpi}>{humanfriendly.format_size(total_mem)}. '
                              f'Limiting to {max_mem_limit_jobs} jobs.')
                    sr.mpi = max_mem_limit_jobs
            max_cpu_limit_jobs = max([
                1,
                len(os.sched_getaffinity(0)) // cpus,
                # psutil.cpu_count(logical=False),  # some mpi implementations allow only number physical cores
            ])
            if max_cpu_limit_jobs < sr.mpi:
                print(f'WARNING: Not enough cpus (#cpus={len(os.sched_getaffinity(0))}) '
                      f'for {sr.mpi} jobs. '
                      f'Limiting to {max_cpu_limit_jobs} jobs.')
                sr.mpi = max_cpu_limit_jobs
            def to_cmd(cmd):
                # From OpenMPI help text, when requeste to many mpi processes:
                #     In all the above cases, if you want Open MPI to default to the number
                #     of hardware threads instead of the number of processor cores, use the
                #     --use-hwthread-cpus option.
                # MPICH has also this option. Hence, activate it by default.

                if shell:
                    return f'mpirun --use-hwthread-cpus -np {sr.mpi} {cmd.str}'
                else:
                    return ['mpirun', '--use-hwthread-cpus', '-np', f'{sr.mpi}'] + cmd.lst
        else:
            def to_cmd(cmd):
                return cmd.str if shell else cmd.lst

    return to_cmd(cmd)


def bash_wrap(cmd, shell=True):
    """
    Wraps a commad to be executed by bash.
    Motivation:
     - The command can contain pipes and other bash features.
     - The new command can always be executed by srun.

    >>> print(bash_wrap('echo "hello"'))
    bash -c 'echo "hello"'
    >>> print(bash_wrap("echo 'hello'"))
    bash -c 'echo '"'"'hello'"'"''
    >>> print(bash_wrap('echo "hello" | grep "hello"'))
    bash -c 'echo "hello" | grep "hello"'
    >>> print(subprocess.check_output(bash_wrap('echo "hello" | grep "hello"'), shell=True, universal_newlines=True))
    hello
    <BLANKLINE>
    >>> print(bash_wrap('echo "hello" && echo "World"'))
    bash -c 'echo "hello" && echo "World"'
    >>> print(subprocess.check_output(bash_wrap('echo "hello" && echo "world"'), shell=True, universal_newlines=True))
    hello
    world
    <BLANKLINE>
    >>> print(bash_wrap(['echo', 'hello']))
    bash -c 'echo hello'

    >>> print(bash_wrap('echo "hello" && echo "World"', shell=False))
    ['bash', '-c', 'echo "hello" && echo "World"']
    """
    if shell:
        if isinstance(cmd, str):
            return f'bash -c {shlex.quote(cmd)}'
        else:
            return f'bash -c {shlex.quote(shlex.join(cmd))}'
    else:
        if isinstance(cmd, str):
            return ['bash', '-c', cmd]
        else:
            return ['bash', '-c', shlex.join(cmd)]
