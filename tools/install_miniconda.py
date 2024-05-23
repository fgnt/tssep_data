"""
python install_miniconda.py  # Install miniconda
python install_miniconda.py dry  # Only print commands that would be executed.

"""

import sys
import subprocess
import shlex
import os
from pathlib import Path


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


def run(cmd, check=True, dry=False, stdout=None):
    print(f'{c.Green}$ {shlex.join(cmd) if isinstance(cmd, list) else cmd}{c.Color_Off}')
    if not dry:
        subprocess.run(
            cmd,
            shell=not isinstance(cmd, list),
            universal_newlines=True,
            stdout=stdout,
            check=check,
            executable="/bin/bash"
        )


def ask_yes_no(question):
    response = input(f'{question} y or n:')
    if response in ['y', 'Y']:
        return True
    else:
        return False


def check_cache_folder():
    print(f'Expect an empty cache dir:')
    run(f'ls ~/.cache/pip/', check=False)

    try:
        num_files = len(list(os.scandir(os.path.expanduser('~/.cache/pip/'))))
    except FileNotFoundError:
        num_files = 0

    if 0 == num_files:
        print(f'~/.cache/pip/ is empty that is fine')
    else:
        print(
            f'{c.Red}Found files in ~/.cache/pip/, delete them{c.Color_Off}\n'
            f'Do not forget to update ~/.config/pip/pip.conf with\n'
            '[global]\n'
            'no-cache-dir = false'
        )
        # tee.log('Do you want to ignore this? y or n:')
        response = input('Do you want to ignore this? y or n:')
        if response in ['y', 'Y']:
            pass
        else:
            raise AssertionError('~/.cache/pip/ is not empty')


def main(dry=None):
    dry: bool = {None: False, 'dry': True}[dry]

    check_cache_folder()

    here = Path(__file__).parent
    conda_dir = here / 'conda'
    activate = f'source {conda_dir}/bin/activate'


#    run('wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh', dry=dry)
#    run(f'bash Miniconda3-latest-Linux-x86_64.sh -b -p {conda_dir}', dry=dry)
    run('wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh', dry=dry)
    run(f'bash Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -b -p {conda_dir}', dry=dry)
    run(f'{activate} && conda install -y gxx_linux-64', dry=dry)

    # By default torch is installed with GPU support.
    # If you need to install torch with a specific CUDA version, please check
    # https://pytorch.org/get-started/locally/
    run(f'{activate} && pip install torch torchvision torchaudio', dry=dry)
    # run(f'{activate} && pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113', dry=dry)

    run(f'{activate} && conda env update -f {here}/environment.yml', dry=dry)

    run(f'{activate} && pip install -e ..', dry=dry)

    if not (Path(__file__).parent.parent.parent / 'tssep' / 'setup.py').exists():
        print(
            f'{c.Red}'
            'Could not find tssep/setup.py. '
            'Please clone the tssep repository such, that tssep_data and tssep are in the same folder and try again.'
            f'{c.Color_Off}'
        )
        return
    run(f'{activate} && pip install -e ../../tssep', dry=dry)

    if ask_yes_no(f'Should MPI be tested with slurm (recommended, but requires slurm)?'):
        run(f'{activate} && python {Path(__file__).parent / "mpi_test.py"}', dry=dry)
        if not ask_yes_no(f'Has the command printed "... of 4 on ..." four times?'):
            print(
                f'{c.Red}'
                'Installation of mpi4py failed.'
                'You have to manually fix this.'
                'Suggestions (Might work, but not guaranteed):\n'
                ' - Uninstall mpi4py (pip uninstall mpi4py)\n'
                ' - Activate or install openmpi (e.g. module add openmpi or sudo apt-get install libopenmpi-dev openmpi-bin openmpi-doc)\n'
                ' - Install mpi4py (e.g. pip install --no-cache --force mpi4py)\n'
                ' - Try again "salloc -n 4 mpirun python -m mpi4py.bench helloworld"\n'
                f'{c.Color_Off}'
            )

if __name__ == '__main__':
    main(*sys.argv[1:])
