#!/usr/bin/env python
import inspect
import os
import re
import shlex
import subprocess
import sys
import signal
import logging
import decimal
import shutil
from pathlib import Path

import paderbox as pb

from tssep_data.util.slurm import cmd_to_hpc
from tssep_data.util.cmd_runner import touch, CLICommands, confirm, run, c, maybe_execute, env_check, user_select, add_stage_cmd


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

    if shutil.which('sbatch') is None:
        sel = 'local'
        print(f'{c.yellow}No sbatch command found. Running locally.{c.end}')
    else:
        sel = user_select(q, ['slurm', 'local'], 'slurm')
    match sel:
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

    launch_training(
        VAD_STORAGE_DIR,
        'TS-VAD',
        init_fn=lambda: run([
            sys.executable, '-m', 'tssep_data.train.run', 'init', 'with',
            # f'vad',
            f'{cwd}/cfg/common.yaml',
            f'{cwd}/cfg/tsvad.yaml',
            f'eg.trainer.storage_dir={VAD_STORAGE_DIR}',
            f'eg.trainer.stop_trigger=[100000,"iteration"]',
        ])
    )


def get_checkpoint(storage_dir):
    assert storage_dir / 'checkpoints', storage_dir / 'checkpoints'
    import natsort

    ckpt_dir = storage_dir / 'checkpoints'

    checkpoints = natsort.natsorted([
        f
        for f in ckpt_dir.glob('ckpt_*.pth')
        if not f.is_symlink()
    ])
    assert checkpoints, ckpt_dir

    d = pb.io.load(ckpt_dir / 'ckpt_latest.pth', unsafe=True)
    try:
        ckpt_ranking = pb.utils.mapping.Dispatcher(
            d['hooks']['BackOffValidationHook']['ckpt_ranking'])
    except KeyError:
        raise
        ckpt_ranking = {}
    else:
        ckpt_ranking = dict(ckpt_ranking)

    cwd = Path.cwd()
    checkpoint = user_select(
        'Which checkpoint should be used for TS-SEP as initialization?',
        {f'{os.path.relpath(c, cwd)} (loss: {decimal.Decimal(f"{ckpt_ranking[c.name]:.2g}")})': c for c in checkpoints},
    )
    assert checkpoint, checkpoint
    return checkpoint


@clicommands
def tssep():
    cwd = Path.cwd()

    def init_fn():
        checkpoint = get_checkpoint(VAD_STORAGE_DIR)
        run([
            sys.executable, '-m', 'tssep_data.train.run', 'init', 'with',
            f'{cwd}/cfg/common.yaml',
            f'{cwd}/cfg/tssep.yaml',
            f'eg.trainer.storage_dir={SEP_STORAGE_DIR}',
            f'eg.trainer.stop_trigger=[100000,"iteration"]',
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

    run(cmd_to_hpc(
        f"{sys.executable} -m tssep_data.eval.gss_v2 c7.json {cwd}/data/jsons/libriCSS_raw_chfiles.json --out_folder=gss --channel_slice=: && python -m fire tssep_data.eval.makefile gss_makefile gss",
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
    cwd = Path.cwd()
    VAD_STORAGE_DIR = Path(os.environ.get('VAD_STORAGE_DIR', cwd / 'tsvad')).absolute()
    SEP_STORAGE_DIR = Path(os.environ.get('SEP_STORAGE_DIR', cwd / 'tssep')).absolute()

    file_parent = Path(__file__).parent
    if file_parent != cwd:
        print(f'WARNING: This script ({__file__}) should be executed in its folder.')
        print(f'WARNING: Changing directory to {file_parent}.')
        os.chdir(file_parent)

    import fire
    env_check()
    add_stage_cmd(clicommands)
    fire.Fire(clicommands.to_dict(), command=None if sys.argv[1:] else 'stage')
