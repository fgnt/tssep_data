"""
from pathlib import Path
from tssep.util.cmd_runner import touch, CLICommands, confirm, run, Green, Color_Off, Red, maybe_execute

clicommands = CLICommands()

@clicommands
def hello_world():
    @maybe_execute(target='hello_world.txt')
    def _():
        run(['echo', 'hello world'])
        run(['touch', 'hello_world.txt'])


if __name__ == '__main__':
    pwd = Path(__file__).parent
    if pwd != Path.cwd():
        print(f'WARNING: This script ({__file__}) should be executed in the parent folder.')
        print(f'WARNING: Changing directory to {pwd}.')
        os.chdir(pwd)

    import fire
    run('just env_check', cwd='..')
    fire.Fire(clicommands.to_dict())

"""
import functools
import inspect
import signal
import subprocess
import sys
import os
import shlex
import re
import logging
import time
from pathlib import Path
from typing import Any

import paderbox.utils.mapping
import tssep_data

Red = '\033[0;31m'
Green = '\033[0;32m'
Blue = '\033[0;34m'
Purple = '\033[0;35m'
Color_Off = '\033[0m'


class c:
    red = '\033[0;31m'
    green = '\033[0;32m'
    yellow = '\033[0;33m'
    blue = '\033[0;34m'
    purple = '\033[0;35m'
    cyan = '\033[0;36m'
    white = '\033[0;37m'
    end = '\033[0m'



class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        if self.signal_received:
            # Second SIGINT: raise KeyboardInterrupt
            self.old_handler(*self.signal_received)
            self.signal_received = False
        else:
            self.signal_received = (sig, frame)
            logging.warning('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def run(cmd, cwd=None, env=None):
    cmd_str = cmd if isinstance(cmd, str) else shlex.join(cmd)
    print(f'Running {Green}{cmd_str}{Color_Off} in {Green}{cwd or "."}{Color_Off} with env={Green}{env}{Color_Off}')
    env_ = env if env is None else {**os.environ, **env}

    start = time.time()
    with DelayedKeyboardInterrupt():
        cp = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                            check=False, env=env_)
    if cp.returncode != 0:
        # https://linuxconfig.org/list-of-exit-codes-on-linux
        exitmsg = {
            126: '126: Command found but is not executable',
            127: '127: Command not found, usually the result of a missing directory in $PATH variable',
            130: '130: Command terminated with signal 2 (SIGINT) (ctrl+c on keyboard). 128+2',
            137: '137: Command terminated with signal 9 (SIGKILL) (kill command). 128+9',
            143: '143: Command terminated with signal 15 (SIGTERM) (kill command). 128+15',
        }.get(cp.returncode, cp.returncode)
        if isinstance(exitmsg, int) and 1 <= cp.returncode - 128 <= 15:
            sig = signal.Signals(cp.returncode - 128)
            msg = f'{cp.returncode}: (Command terminated with signal {sig.value} ({sig.name}). 128+{sig.value})'

        print(
            f'{Red}Cmd\n\t{cmd_str}\nin {cwd or os.getcwd()} with env={env}\nfailed ({exitmsg}).{Color_Off}')
        sys.exit(cp.returncode)
    else:
        print(f'Finished {Green}{cmd_str}{Color_Off} in {Green}{cwd or "."}{Color_Off} in {(time.time() - start) / 3600:.2f}h')

    return cp


def confirm(question):
    import questionary
    return questionary.confirm(question).unsafe_ask()


def text(question):
    import questionary
    return questionary.text(question).unsafe_ask()


def user_select(message, choices: 'tuple[Any] | list[Any] | dict', default=None):
    """


    python -c 'from tssep.util.cmd_runner import user_select; print(user_select("Question",{"a": 1, "b": 2}, "a"))'


    >>> user_select('Question', ['a', 'b'], 'a')
    """
    import questionary
    if isinstance(choices, dict):
        keys = tuple(choices.keys())
        choices = {None: None, **choices}
        if isinstance(default, int) and default not in choices and default >= 0:
            default = keys[default]
        sel = questionary.select(message, keys, default).unsafe_ask()
        return choices[sel]
    else:
        if isinstance(default, int) and default not in choices and default >= 0:
            default = choices[default]
        return questionary.select(message, choices, default).unsafe_ask()


def touch(file):
    file = Path(file)
    try:
        file.touch()
    except FileNotFoundError:
        file.parent.mkdir(exist_ok=True, parents=True)
        file.touch()
    print(f'Touch {file}')


class CLICommands:
    def __init__(self):
        self._commands = {}

    def to_dict(self):
        return {k: v for k, v in self._commands.items()}

    def __call__(self, func):
        self._commands[func.__name__] = func
        return func

    def create_makefile(self, __file__):
        __file__ = Path(__file__)
        m = __file__.parent / 'Makefile'
        d = [
            f'# This file is autogenerated by {os.path.relpath(__file__, tssep_data.git_root)}.\n',
        ]
        d += [
            '.PHONY: help\nhelp:\n\t@cat Makefile\n'
        ]
        d = d + [
            f'.PHONY: {k}\n{k}:\n\tpython {__file__.name} {k}\n'
            for k in self._commands
        ]
        m.write_text('\n'.join(d))
        print('Wrote', m)


def add_stage_cmd(clicommands):
    @clicommands
    def stage(stage=None):
        cmds = clicommands.to_dict()
        cmds.pop('stage')
        cmds = list(cmds.values())

        for i, v in enumerate(cmds):
            print(f'Stage {i+1} is {v.__name__}')

        def match(stage):
            if isinstance(stage, str):
                if (m := re.match('(\d*)[-](\d*)', stage)):
                    start, end = m.groups()
                    start = int(start) if start else 1
                    end = int(end) if end else len(cmds)
                    if 1 <= start <= end <= len(cmds):
                        return cmds[start - 1:end]
                raise ValueError(f'Expected {stage!r} to be "start-end" with start, end in {{1, ..., {len(cmds)}}}. e.g. 2-3 (end is included).')

        if stage is None:
            stage, cmds = user_select(
                'Select stage(s) to run',
                {
                    'None': ['None', []],
                    **{
                        f'Stage {i + 1} is {v.__name__}': [i+1, [v]]
                        for i, v in enumerate(cmds)
                    },
                    'All': ['All', cmds],
                },
                default='None',
            )
        elif isinstance(stage, int):
            if 1 <= stage <= len(cmds):
                cmds = [cmds[stage - 1]]
            else:
                raise ValueError(f'Expected {stage} in {{1, ..., {len(cmds)}}}')
        elif isinstance(stage, tuple) and all(isinstance(s, int) for s in stage):
            if (1 <= s <= len(cmds) for s in stage):
                cmds = [cmds[s - 1] for s in stage]
            else:
                raise ValueError(f'Expected {stage} in {{1, ..., {len(cmds)}}}')
            stage = ','.join([str(s) for s in stage])
        elif m := match(stage):
            cmds = m
        else:
            raise ValueError(stage, f'Expected <int> or <int>-<int>, where <int> in {{1, ..., {len(cmds)}}}')

        print(f'Run Stage(s) {stage}: {", ".join([c.__name__ for c in cmds])}')
        for c in cmds:
            c()
        print(f'Ran Stage(s) {stage}: {", ".join([c.__name__ for c in cmds])}')


def maybe_execute(
        name=None,
        done_file=None,
        target=None,
        target_samples=None,
        overwrite_works=False,  # Whether the target(s) have to be (manually) deleted or not.
        call_immediately=True,
        *,
        source=None,
        sources=None,
):
    """

    Args:
        name: Default: Name if the function
        done_file: Default: True if target is None and target_samples is None else f'done/{name}'
            A file that indicates, whether the function was already executed.
            It will be created after the function was successfully executed.
        target:
            Like in Makefile, the target file that should be created.
            If it exists, it means the execution is not necessary.
        target_samples:
            Some files of all files that should be created.
            They are used to estimate, whether the execution was successfully.
            Checking all files is often too expensive.
            Note: For target_samples, glob is supported, e.g. `*.wav` and
            `ch?.wav`, where `*` and `?` are placeholders for
            any string (0 to inf characters) and any character.
        overwrite_works:
            Whether the target(s) have to be deleted or not before the execution..

    Returns:

    """
    def wrapper(func):
        nonlocal target, done_file, target_samples, name

        sig = inspect.signature(func)
        parameters = sig.parameters

        if name is None:
            if func.__name__ != '_':
                name = func.__qualname__
            elif target is not None:
                name = Path(target).name
            elif done_file is not None:
                name = Path(done_file).name
            else:
                raise RuntimeError(name, func, target, done_file)

        if done_file is None:
            done_file = target is None  # and target_samples is None
        if done_file is True:
            done_file = re.sub('[^A-Za-z0-9]', '_', name)
            done_file = f'done/{done_file}'

        name = f'{Purple}{name}{Color_Off}'

        execute = None

        criterions = []
        if done_file:
            done_file = Path(done_file)
            criterions.append(done_file.exists() and done_file.read_text() == func.__qualname__)
        if target:
            target = Path(target)
            criterions.append(target.exists())

        def exists_with_glob(t):
            return t.exists() or any(t.parent.glob(t.name))

        if target_samples:
            target_samples = [Path(t) for t in target_samples]
            criterions.extend([exists_with_glob(t) for t in target_samples])

        assert source is None or sources is None, ('Use either source or sources, not both.', source, sources)
        if source is not None:
            if not Path(source).exists():
                execute = False
                print(f'WARNING:{name}: Skip, since source {source!r} doesn\'t exist.')
        elif sources is not None:
            if not all(Path(s).exists() for s in sources):
                execute = False
                m = {True: 'exists', False: "doesn't exists"}
                print(f'WARNING:{name}: Skip, since at least one source doesn\'t exist:',
                      {f'{s}: {m[Path(s).exists()]}' for s in sources})

        if execute is False:
            # At least one source doesn't exist.
            pass
        elif all(criterions):
            execute = False
            if done_file:
                print(f'{name}: Skip, since {done_file!r} exists. Delete it to rerun.')
            elif target:
                print(f'{name}: Skip, since target {os.fspath(target)!r} exists. Delete it to rerun.')
            else:
                print(f'{name}: Skip, since target_samples {list(map(os.fspath, target_samples))!r} exists. Delete them to rerun.')
        elif not any(criterions):
            execute = True
        # elif overwrite_works:
        else:
            print(f'{name}: Issue some but not all files exists:')
            tmp = paderbox.utils.mapping.Dispatcher({True: f'{Green}exists{Color_Off}', False: f"{Red}doesn't exists{Color_Off}"})
            if done_file:
                print(' '*4, f'- {done_file}: {tmp[done_file.exists()]}')
                if done_file.exists() and done_file.read_text() != func.__qualname__:
                    print(' ' * 8, f'- done_file.read_text() == {done_file.read_text()!r} != {func.__qualname__!r} == func.__qualname__')
            if target:
                print(' '*4, f'- Target: {os.fspath(target)!r}: {tmp[target.exists()]}')
            print(' '*4, f'- Target samples:')
            if target_samples:
                for t in target_samples:
                    print(' '*8, f'- {os.fspath(t)!r}: {tmp[exists_with_glob(t)]}')

            # if not confirm(f'{name}: Rerun?'):
            #     sys.exit(1)
            # execute = True

            sel = user_select(f'{name}: Rerun?', [
                'No',
                'Yes' if overwrite_works else 'Yes (disabled for this task, you have to clean up manually)',
                '(Debug option) Skip and touch done file',
            ])
            if sel == 'No':
                execute = False
            elif sel == 'Yes':
                execute = True
            elif sel == '(Debug option) Skip and touch done file':
                touch(done_file)
                done_file.write_text(func.__qualname__)
                execute = False
            else:
                raise ValueError(sel)

        if execute:
            kwargs = {}
            if 'target' in parameters:
                kwargs['target'] = target
            if 'source' in parameters:
                assert source is not None, (source, func, parameters)
                kwargs['source'] = source
            if 'sources' in parameters:
                assert sources is not None, (sources, func, parameters)
                kwargs['sources'] = sources

            _ = func(**kwargs)
            if target:
                assert target.exists(), (target, func, kwargs)
            if target_samples:
                missing = [f'{Red}Error: {t} doesn\'t exist.{Color_Off}' for t in target_samples if not exists_with_glob(t)]
                if missing:
                    missing = '\n'.join(missing)
                    assert missing, f'\n{missing}\ntarget_samples: {target_samples}\nfunc: {func}\nkwargs: {kwargs}'
            if done_file:
                touch(done_file)
                done_file.write_text(func.__qualname__)
        else:
            pass
            # print(f'{name}: Skipped.')
        return func
    if call_immediately:
        return wrapper
    else:
        def wrapper2(func):
            @functools.wraps(func)
            def wrapper3(*args, **kwargs):
                return wrapper(func)(*args, **kwargs)
            return wrapper3
        return wrapper2


def env_check():
    import re, subprocess, sys, json, os
    import paderbox as pb
    import tssep_data
    git_root = Path(tssep_data.__file__).parent.parent  # .../tssep_data/__init__.py/../..
    environment = pb.io.load_yaml(git_root / 'tools/environment.yml')

    def normalize(package_name):
        # git+https://github.com/boeddeker/rirgen.git#egg=rirgen
        try:
            if package_name.startswith('git+http'):
                package_name = package_name.split('#egg=', maxsplit=1)[1].split('[')[0]
            return re.split('[><=]=?', package_name)[0].lower().replace('_', '-')
        except Exception:
            raise Exception(package_name)

    packages = [
        normalize(p)
        for e in environment['dependencies']
        for p in ([e] if isinstance(e, str) else e['pip'])
        if p not in ['python==3.11']
    ]

    # import check doesn't work, since some packages use different names
    # e.g. pyyaml vs. yaml (PyPI/Install name vs Python/import name)
    installed_packages = subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], universal_newlines=True)
    installed_packages = json.loads(installed_packages)
    installed_packages = [normalize(r['name']) for r in installed_packages]

    import importlib
    Red = '\033[0;31m'
    Color_Off = '\033[0m'
    fail = False

    init_msg = "The following packages aren't installed in the python environment:\n"
    for p in packages:
        if p not in installed_packages:
            fail = True
            print(f'{init_msg}\t{Red}✘ {p}{Color_Off}')
            init_msg = ''

    KALDI_ROOT = os.environ.get('KALDI_ROOT')
    if not KALDI_ROOT:
        print(f'{Red}✘ KALDI_ROOT not set{Color_Off}')
    elif not os.path.exists(KALDI_ROOT):
        print(f"{Red}✘ KALDI_ROOT={KALDI_ROOT} doen't exist{Color_Off}")
    if fail:
        sys.exit(1)
