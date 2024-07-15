"""

Licence MIT
Origin: Communications Department, Paderborn University, Germany

"""

from pathlib import Path

import tssep
from tssep_data.util.cmd_runner import run

def main(file='c7_words_nemo.json'):
    """
    python -m tssep.eval.c7_frame_to_second c7_words_nemo.json
	python -m tssep.eval.normalize_words c7_words_nemo_fix.json
	python -m meeteval.io.chime7 to_stm c7_words_nemo_fix.json > c7_words_nemo_fix.stm
	python -m meeteval.der md_eval_22 -r /scratch/hpc-prf-nt1/cbj/deploy/css/egs/libri_css/data/libri_css_ref.stm -h c7_words_nemo_fix.stm
	meeteval-wer cpwer -r /scratch/hpc-prf-nt1/cbj/deploy/css/egs/libri_css/data/libri_css_ref.stm -h c7_words_nemo_fix.stm
	cat c7_words_nemo_fix_md_eval_22.json
	cat c7_words_nemo_fix_cpwer.json

    """
    ref = tssep.git_root / 'egs/libri_css/data/libri_css_ref.stm'

    file = Path(file)
    run(f'python -m tssep.eval.c7_frame_to_second {file}')
    file_fix = file.with_stem(f'{file.stem}_fix')
    run(f'python -m tssep.eval.normalize_words {file_fix}')
    file_stm = file_fix.with_suffix('.stm')
    run(f'python -m meeteval.io.chime7 to_stm {file_fix} > {file_stm}')
    run(f'python -m meeteval.der md_eval_22 -r {ref} -h {file_stm}')
    run(f'meeteval-wer cpwer -r {ref} -h {file_stm}')
    file_der = file_fix.with_suffix('_md_eval_22.json')
    file_cpwer = file_fix.with_suffix('_cpwer.json')
    run(f'cat {file_der}')
    run(f'cat {file_cpwer}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
