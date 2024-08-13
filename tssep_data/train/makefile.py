from pathlib import Path

from padertorch.contrib.cb.io import SimpleMakefile
import padertorch as pt

from tssep.train.makefile import makefile as train_makefile


def makefile(_config, eg, dump=True):
    storage_dir = Path(eg["trainer"]["storage_dir"])

    main_python_path = pt.configurable.resolve_main_python_path()
    eval_python_path = main_python_path.replace('train', 'eval')
    assert main_python_path != eval_python_path, (main_python_path, eval_python_path)

    m = train_makefile(_config, eg, dump=False)

    m.phony["run_pdb"] = f"python -m {main_python_path} --pdb with config.yaml"

    m.phony['sbatch'] = f'python -m {main_python_path} sbatch with config.yaml'

    m.phony['eval_init'] = f'python -m {eval_python_path} init with config.yaml default'
    m.phony['eval_init_interactive'] = f'python -m tssep_data.eval.init_interactive'

    if dump:
        m.dump(storage_dir / 'Makefile')
    return m
