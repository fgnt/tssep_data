import os
import functools
import sys
import shutil
from pathlib import Path

import whisper  # pip install openai-whisper
import paderbox as pb
import tqdm
from paderbox.utils.mapping import Dispatcher

import tssep
from tssep.util.cmd_runner import touch, CLICommands, confirm, run, Green, Color_Off, Red, maybe_execute, env_check, user_select, add_stage_cmd
from tssep.util.slurm import cmd_to_hpc


def load_model(model_name):
    load_model = functools.partial(
        whisper.load_model, model_name,
        download_root=Path(tssep.__file__).parent.parent / 'cache')
    model = load_model()
    return model  # noqa


def main(json, out=None, file='audio_path', model_name='tiny.en'):
    json = Path(json)
    if out is None:
        out = json.with_name(json.stem + f'_whisper_{model_name}.json')

    data = pb.io.load(json)
    asr = load_model(model_name)

    def foo(segment):
        result = asr.transcribe(
            segment[file],
            fp16=False,  # without it raises an annoying warning in each call.
        )
        segment['words'] = result["text"]
        return segment

    data = list(map(foo, tqdm.tqdm(data)))

    pb.io.dump(data, out)
    print(f'Wrote {out}')


def launch(json, out=None, file='audio_path', model_name='tiny.en'):
    q = f'Start transcribe for {json} with whisper {model_name}?'

    json = Path(json)

    choices = {
        # Don't change the order here. It has to be slurm than local.
        # It is used later to select the default.
        'slurm/mpirun': 'slurm',
        'local/serial': 'local',
    }

    cmd = f"{sys.executable} -m tssep.eval.transcribe run {json} {out} {file} {model_name}"

    match sel := user_select(
        q,
        choices,
        default=list(choices.keys())[shutil.which('sbatch') is None]
    ):
        case 'slurm':
            # tiny.en takes less than (2 / mpi) hours
            # large-v2 takes less than (200 / mpi) hours
            run(cmd_to_hpc(
                cmd,
                job_name=f'transcribe_{json.name}',
                block=True,  # Set to false, if you use bigger models.
                shell=True,
                time=Dispatcher({'tiny.en': '1h', 'large-v2': '9h'})[model_name],
                mem=Dispatcher({'tiny.en': '2G', 'large-v2': '12G'})[model_name],
                mpi='40',
                shell_wrap=True,
            ))
        case 'local':
            run(cmd)
        case _:
            raise RuntimeError(sel)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'run': main,
        'launch': launch,
    })
