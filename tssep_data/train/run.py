import subprocess
import tempfile
from tssep.train.run import *


@ex.command(unobserved=True)
def meld(_config, eg):
    """
    Similar to the diff command, but will open meld, so you can fix the config.
    """
    storage_dir = Path(eg['trainer']['storage_dir'])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_tmp = tmpdir / 'config.yaml'
        pb.io.dump(_config, config_tmp)

        config_yaml = Path(storage_dir / 'config.yaml')
        if config_yaml.exists():
            backup_config(config_yaml)

        subprocess.run(['meld', str(config_yaml), str(config_tmp)], env=os.environ)


@ex.command(unobserved=True)
def diff(_config, eg, storage_dir=None):
    """
    Show the missmatch between the config in the 'storage_dir' and the config that would be used.
    Usually they should be equal, but when new parameters are introduced, after the last executing
    this function can be used to see if the new parameters have the right value.

    """
    if storage_dir is None:
        storage_dir = Path(eg['trainer']['storage_dir'])
    else:
        storage_dir = Path(storage_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        config_tmp = tmpdir / 'config.yaml'
        pb.io.dump(_config, config_tmp)

        config_yaml = storage_dir / 'config.yaml'
        # if config_yaml.exists():
        #     backup_config(config_yaml)
        subprocess.run(['icdiff', str(config_yaml), str(config_tmp)], env=os.environ)

# @ex.named_config
# def vad():

from tssep_data.train.makefile import makefile

makefile = ex.command(unobserved=True)(makefile)

if __name__ == "__main__":
    import shlex
    import psutil

    print(shlex.join(psutil.Process().cmdline()))

    # To debug, use --pdb commandline option from sacred
    ex.run_commandline()
