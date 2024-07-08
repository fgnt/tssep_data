#!/usr/bin/env python
import inspect
import os
import re
import shlex
import subprocess
import sys
import signal
import logging
import shutil
from pathlib import Path

import paderbox as pb

from tssep_data.util.slurm import cmd_to_hpc
from tssep_data.util.cmd_runner import touch, CLICommands, confirm, run, Green, Color_Off, Red, maybe_execute, env_check, user_select, add_stage_cmd


clicommands = CLICommands()


def launch_training(
        storage_dir,
        name,
        init_fn,
        checkpoint=False,
):
    cwd = Path.cwd()
    storage_dir = storage_dir

    @maybe_execute(
        target=(storage_dir / 'config.yaml').relative_to(cwd),
        target_samples=[
            (storage_dir).relative_to(cwd),
            (storage_dir / 'Makefile').relative_to(cwd),
        ]
    )
    def _():
        init_fn()

    q = f'Start/Submit the {name} training in {storage_dir!r}?'
    if (storage_dir / 'checkpoints').exists():
        q += ' (restart/continue)'

    match sel := user_select(q, ['slurm', 'local'], ['slurm', 'local'][shutil.which('sbatch') is None]):
        case 'slurm':
            run(
                f'{sys.executable} -m tssep_data.train.run sbatch with config.yaml',
                cwd=storage_dir,
            )
        case 'local':
            run(
                f'{sys.executable} -m tssep_data.train.run with config.yaml',
                cwd=storage_dir,
            )
        case 'skip':
            pass
        case _:
            raise RuntimeError(sel)


@clicommands
def tsvad():
    cwd = Path.cwd()
    VAD_STORAGE_DIR = cwd / 'tsvad'
    VAD_TARGET_DIR = cwd / 'data/jsons/sim_libri_css_early/target_vad/v2'

    launch_training(
        VAD_STORAGE_DIR,
        'TS-VAD',
        init_fn=lambda: run([
            sys.executable, '-m', 'tssep_data.train.run', 'init', 'with',
            # f'vad',
            f'{cwd}/cfg/common.yaml',
            f'eg.trainer.storage_dir={VAD_STORAGE_DIR}',
            # f'eg.trainer.model.aux_data.db.json_path={cwd}/data/ivector/simLibriCSS_oracle_ivectors.json',
            # f'eg.trainer.model.reader.db.json_path={cwd}/data/jsons/sim_libri_css.json',
            # f'eg.trainer.model.reader.db_librispeech.json_path={cwd}/data/jsons/librispeech.json',
            # f'eg.trainer.model.reader.vad.files.SimLibriCSS-train={VAD_TARGET_DIR}/SimLibriCSS-train.pkl',
            # f'eg.trainer.model.reader.vad.files.SimLibriCSS-dev={VAD_TARGET_DIR}/SimLibriCSS-dev.pkl',
            # f'eg.trainer.model.reader.vad.files.SimLibriCSS-test={VAD_TARGET_DIR}/SimLibriCSS-test.pkl',
            f'eg.trainer.stop_trigger=[100000,"iteration"]',
        ])
    )


def get_checkpoint(storage_dir):
    assert storage_dir / 'checkpoints', storage_dir / 'checkpoints'
    import natsort

    checkpoints = natsort.natsorted([
        f
        for f in (storage_dir / 'checkpoints').glob('ckpt_*.pth')
        if not f.is_symlink()
    ])
    assert checkpoints, storage_dir / 'checkpoints'

    checkpoint = user_select(
        'Which checkpoint should be used for TS-SEP as initialization?',
        {str(c): c for c in checkpoints},
    )
    assert checkpoint, checkpoint
    return checkpoint


@clicommands
def tssep():
    cwd = Path.cwd()
    VAD_STORAGE_DIR = cwd / 'tsvad'
    SEP_STORAGE_DIR = cwd / 'tssep'

    def init_fn():
        checkpoint = get_checkpoint(VAD_STORAGE_DIR)
        run([
            sys.executable, '-m', 'tssep.train.run', 'init', 'with',
            f'sep',
            f'eg.trainer.storage_dir={SEP_STORAGE_DIR}',
            f'eg.trainer.stop_trigger=[100000,"iteration"]',
            # f'eg.init_ckpt.factory=tssep.train.init_ckpt.InitCheckPointVAD2Sep',
            f'eg.init_ckpt.init_ckpt={checkpoint}',
        ])

    launch_training(
        SEP_STORAGE_DIR,
        'TS-SEP',
        init_fn=init_fn,
    )


@clicommands
def espnet_gss():
    cwd = Path.cwd()
    STORAGE_DIR = cwd / 'espnet'

    dev = cwd / Path('data/espnet_libri_css_diarize_spectral_rttm/orig_id/dev.rttm')
    eval = cwd / Path('data/espnet_libri_css_diarize_spectral_rttm/orig_id/eval.rttm')

    STORAGE_DIR.mkdir()

    run(f'{sys.executable} -m meeteval.io.chime7 from_rttm {dev} > {STORAGE_DIR / "c7_dev.json"}')
    run(f'{sys.executable} -m meeteval.io.chime7 from_rttm {eval} > {STORAGE_DIR / "c7_eval.json"}')
    pb.io.dump(pb.io.load(STORAGE_DIR / "c7_dev.json") + pb.io.load(STORAGE_DIR / "c7_eval.json"), STORAGE_DIR / "c7.json")

    run(f'{sys.executable} -m meeteval.io.chime7 from_rttm {eval} > {STORAGE_DIR / "c7_dev.json"}')

    # /scratch/hpc-prf-nt1/cbj/deploy/css/egs/libri_css/data/jsons/libriCSS_raw_chfiles.json
    run(cmd_to_hpc(
        f"{sys.executable} -m tssep.eval.gss_v2 c7.json {cwd}/data/jsons/libriCSS_raw_chfiles.json --out_folder=gss --channel_slice=: && python -m fire tssep.eval.makefile gss_makefile gss",
        job_name=f'espnet_gss_v2',
        block=False,
        shell=True,
        time='24h',
        mem='4G',
        mpi='40',
        shell_wrap=True,
    ), cwd=STORAGE_DIR)


@clicommands
def tensorboard_symlink_tree():
    makefile = f"""
tree1days:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	{sys.executable} -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents* --max_age=1days

tree7days:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	{sys.executable} -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents* --max_age=7days

tree:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	{sys.executable} -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents*
    """
    file = Path('tensorboard')


def _create_makefile():
    """
    Dummy function to create the Makefile by executing this doctest
    Advantage of Makefile:
     - Compactly define all commands
     - Autocomplete

    >>> _create_makefile()  # doctest: +ELLIPSIS
    Wrote .../egs/libri_css/Makefile
    """
    clicommands.create_makefile(__file__)


if __name__ == '__main__':
    pwd = Path(__file__).parent
    if pwd != Path.cwd():
        print(f'WARNING: This script ({__file__}) should be executed in the parent folder.')
        print(f'WARNING: Changing directory to {pwd}.')
        os.chdir(pwd)

    import fire
    env_check()
    add_stage_cmd(clicommands)
    fire.Fire(clicommands.to_dict(), command=None if sys.argv[1:] else 'stage')
