"""

python -m tssep.eval.run init with config.yaml libricss

python -m tssep.eval.run sbatch with config.yaml libricss
python -m tssep.eval.run sbatch with config.yaml libricss eeg.ckpt=$(realpath checkpoints/ckpt_12000.pth)

python -m tssep.eval.run init with config.yaml libricss_tsvad_oracle
python -m tssep.eval.run sbatch with config.yaml libricss_tsvad_espnet

"""


import socket
import sys
import os
import datetime
import shutil
import tempfile
import subprocess
import shlex
import subprocess
from pathlib import Path

import torch

import sacred.commands
from sacred.observers import FileStorageObserver, RunObserver

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False
# sacred.SETTINGS.CONFIG.DICT_INSERTION_ORDER_PRESERVATION = True
sacred.SETTINGS.CAPTURE_MODE = 'sys'

# sacred.SETTINGS.HOST_INFO.CAPTURED_ENV: list
sacred.SETTINGS.HOST_INFO.CAPTURED_ENV.extend([
    k
    for k in os.environ.keys()
    if 'cuda' in k.lower() or 'slurm' in k.lower() or 'srun' in k.lower() or k in [
        'OMP_NUM_THREADS'
    ]
])

import padertorch as pt
import paderbox as pb
from paderbox.utils.nested import deflatten

import tssep
import tssep_data.data
import tssep.train.enhancer
from tssep_data.util.bash import c
from tssep_data.util.slurm import SlurmResources
from tssep.train.run import dump_config
from tssep.train.experiment import Experiment
from tssep_data.eval.experiment import EvalExperiment
from tssep_data.data.constants import eg_dir

ex = sacred.Experiment('extract_eval')


@ex.config
def config():
    # Dummys from train config:
    script_python_path = ''
    train_slurm_resources = dict(factory=SlurmResources)  # dummy for sacred
    pt.Configurable.get_config(train_slurm_resources)  # Fill defaults

    eval_script_python_path = pt.configurable.resolve_main_python_path()  # Info inside the config to find, which script produces the config

    eg = {'trainer': {'storage_dir': None, 'model': {'fe': {'shift': 256}}}}
    eeg = {'eval_dir': None, 'probability_to_segments': {'shift': eg['trainer']['model']['fe']['shift']}}
    eeg['ckpt'] = (Path(eg['trainer']['storage_dir']) / 'checkpoints' / 'ckpt_best_loss.pth').resolve()

    if eeg['eval_dir'] is None:
        eeg['eval_dir'] = pt.io.get_new_subdir(
            Path(eg['trainer']['storage_dir']) / 'eval' / Path(eeg['ckpt']).with_suffix('').name.replace('ckpt_', ''),
            consider_mpi=True, mkdir=True)

    Experiment.get_config(eg)  # Fill defaults
    tssep_data.eval.experiment.EvalExperiment.get_config(eeg)  # Fill defaults

    ex.observers.append(FileStorageObserver(Path(eeg['eval_dir']) / 'sacred'))

    # eval_slurm_resources = dict(factory=SlurmResources, mem='40GB', gpus=0, cpus=1, mpi=20)
    eval_slurm_resources = dict(factory=SlurmResources, mem='20GB', gpus=1, cpus=2, mpi=1)
    pt.Configurable.get_config(eval_slurm_resources)  # Fill defaults

    asr_slurm_resources = dict(factory=SlurmResources, mem='5GB', gpus=0, cpus=1, mpi=80)
    pt.Configurable.get_config(asr_slurm_resources)  # Fill defaults


@ex.named_config
def default():
    eg = deflatten({
        # 'trainer.model.reader.eval_dataset_name': 'libri_css',
        # 'trainer.model.reader.datasets.libri_css.observation': '["observation"][:]',
        'trainer.model.reader.data_hooks.tasks.auxInput.estimate': {
            'SimLibriCSS-dev': True,
            # Disable assert, that checks, whether the speaker_ids between ivector and dataset match.
            # At least one speaker has no non-overlapping activity and hence no ivector.
        },
        'trainer.model.reader.data_hooks.tasks.auxInput.json': [
            eg_dir / 'data/ivector/simLibriCSS_oracle_ivectors.json',
            # tssep.git_root / 'egs/libri_css/data/ivector/libriCSS_oracle_ivectors.json',
            eg_dir / 'data/ivector/libriCSS_espnet_ivectors.json',
        ],

    })
    eeg = deflatten({
        'feature_statistics_domain_adaptation': 'mean_std',
        'aux_feature_statistics_domain_adaptation': 'mean_std',
        'wpe.factory': tssep.train.enhancer.WPE,
        'probability_to_segments.subsegment.algorithm': 'partial_gready',  # 'optimal' or 'partial_gready'. 'optimal' is sometimes slow.
        'probability_to_segments.thresh': 0.3,
        'probability_to_segments.max_kernel': 161,
        'probability_to_segments.min_kernel': 81,
        'probability_to_segments.min_frames': 40,  # No effect, if max_kernel - min_kernel >= min_frames
        'nn_segmenter.factory': tssep_data.eval.experiment.Segmenter,  # Reduce memory consumption
        'nn_segmenter.length': 4000,
        'nn_segmenter.shift': 2000,
        'enhancer.factory': tssep.train.enhancer.ClassicBF_np,
        'enhancer.bf': 'mvdr_souden',  # 'ch0', 'mvdr_souden', ...
        'enhancer.masking': True,
        'enhancer.masking_eps': 0.5,
        'channel_reduction': 'median',  # 'median', 'reference'
    })


@ex.command(unobserved=True)
def diff(_config, eg, eeg):
    """
    Show the missmatch between the config in the 'storage_dir' and the config that would be used.
    Usually they should be equal, but when new parameters are introduced, after the last executing
    this function can be used to see if the new parameters have the right value.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_tmp = Path(tmpdir) / 'config.yaml'
        config_yaml = Path(eeg['eval_dir']) / 'config.yaml'
        pb.io.dump(_config, config_tmp)
        subprocess.run(['icdiff', str(config_yaml), str(config_tmp)], env=os.environ)


@ex.command(unobserved=True)
def meld(_config, eg, eeg):
    """
    Show the missmatch between the config in the 'storage_dir' and the config that would be used.
    Usually they should be equal, but when new parameters are introduced, after the last executing
    this function can be used to see if the new parameters have the right value.

    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config_tmp = Path(tmpdir) / 'config.yaml'
        config_yaml = Path(eeg['eval_dir']) / 'config.yaml'
        pb.io.dump(_config, config_tmp)
        subprocess.run(['meld', str(config_yaml), str(config_tmp)], env=os.environ)


@ex.command
def init(eg, eeg, _config, _run):
    storage_dir = Path(eg['trainer']['storage_dir'])
    eval_dir = Path(eeg['eval_dir'])

    with open(eval_dir / 'python_history.txt', 'a') as fd:
        print(f'{psutil.Process().cmdline()}'
              f'  # {datetime.datetime.today().strftime("%Y.%m.%d %H:%M:%S")}'
              f'  # {Path.cwd()}', file=fd)

    dump_config(storage_dir=eval_dir, _config=_config)
    sacred.commands.print_config(_run)

    makefile()

    print(f'Initialized {eval_dir}')


@ex.command
def sbatch(eg, eeg, _config, _run, eval_slurm_resources: SlurmResources):
    storage_dir = Path(eg['trainer']['storage_dir'])
    eval_dir = Path(eeg['eval_dir'])

    init()

    eval_slurm_resources = pt.Configurable.from_config(eval_slurm_resources)
    name = f"{storage_dir.name}_{eval_dir.parent.name.replace('ckpt_', '')}_{eval_dir.name}"

    cmd = f"""{eval_slurm_resources.to_sbatch_str()} --job-name {name} --wrap '{eval_slurm_resources.mpi_cmd} python -m tssep.eval.run -c eval{name} with config.yaml'"""

    print(f'{c.Green}$ {cmd}{c.Color_Off}')

    cp = subprocess.run(cmd, shell=True, cwd=eval_dir, stdout=subprocess.PIPE, universal_newlines=True, env=os.environ)
    print(cp.stdout)
    print(f'See {eval_dir}')

    jobid = cp.stdout
    with open(eval_dir / 'slurm_history.txt', 'a') as fd:
        print(f'{jobid} {cmd}', file=fd)


@ex.command
def srun_debug(eg, eeg, _config, _run, eval_slurm_resources: SlurmResources):
    storage_dir = Path(eg['trainer']['storage_dir'])
    eval_dir = Path(eeg['eval_dir'])

    eval_slurm_resources: SlurmResources = pt.Configurable.from_config(eval_slurm_resources)
    eval_slurm_resources.mpi = 1
    eval_slurm_resources.cpus = max(4, eval_slurm_resources.cpus)

    eval_slurm_resources.time = f"{min(1, eval_slurm_resources._time(style='hours'))}h"

    name = f"{storage_dir.name}_{eval_dir.parent.name.replace('ckpt_', '')}_{eval_dir.name}"
    cmd = f"{eval_slurm_resources.to_srun_str()} --job-name {name} --pty python -m tssep_data.eval.run --pdb -c eval{name} with config.yaml"
    cmd = cmd.split()
    print(f'{c.Green}$ {subprocess.list2cmdline(cmd)}{c.Color_Off}')

    cp = subprocess.run(cmd, shell=False, cwd=eval_dir, env=os.environ)
    if cp.returncode:
        sys.exit(cp.returncode)
    print(f'See {eval_dir}')


@ex.main
def main(_run, _config, eg, eeg):
    """

    """
    eg: Experiment = Experiment.from_config(eg)
    eeg: EvalExperiment = pt.Configurable.from_config(eeg)
    eval_dir = Path(eeg.eval_dir)

    assert eval_dir == Path.cwd(), f'eval_dir: {eval_dir} != {Path.cwd()}'

    dump_config(storage_dir=eeg.eval_dir, _config=_config)
    sacred.commands.print_config(_run)

    if tssep_data.util.bash.location == 'MERL':
        # Note: This is virtual memory
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # 8 * 1024**3 : 8 GBytes
        soft = (8 * int(os.environ.get('SLURM_CPUS_PER_TASK', 1))) * 1024**3

        print(f'### Limit memory usage to: {soft / 1024**3} GiBytes ###')
        print(f'$SLURM_CPUS_PER_TASK={os.environ.get("SLURM_CPUS_PER_TASK", 1)}')
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

    eeg.eval(eg=eg)


from tssep_data.eval.makefile import makefile
makefile = ex.command(unobserved=True)(makefile)

if __name__ == '__main__':
    import psutil, shlex
    print(shlex.join(psutil.Process().cmdline()))
    ex.run_commandline()

    from tssep_data.util.slurm import shutdown_soon
    shutdown_soon()  # Sometimes the Job gets stuck.
