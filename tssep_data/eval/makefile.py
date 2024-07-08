import dataclasses
import os
from pathlib import Path

import padertorch as pt
from padertorch.contrib.cb.io import SimpleMakefile
import tssep

from tssep.util.slurm import SlurmResources
from tssep.util.slurm import cmd_to_hpc, bash_wrap


def get_eval_id(eg, eeg):
    storage_dir = Path(eg['trainer']['storage_dir'])
    eval_dir = Path(eeg['eval_dir'])
    return f"{storage_dir.name}_{eval_dir.parent.name.replace('ckpt_', '')}_{eval_dir.name}"


def makefile(_config, eg, eeg, eval_slurm_resources, asr_slurm_resources):

    storage_dir = Path(eg['trainer']['storage_dir'])
    eval_dir = Path(eeg['eval_dir'])

    _config = pt.configurable.recursive_class_to_str(_config)

    eval_id = get_eval_id(eg, eeg)

    slurm: SlurmResources = pt.Configurable.from_config(eval_slurm_resources)
    asr_slurm: SlurmResources = pt.Configurable.from_config(asr_slurm_resources)

    if slurm.job_name is None:
        slurm.job_name = eval_id

    m = SimpleMakefile()
    m += 'SHELL := /bin/bash'
    m.phony['help'] = 'cat Makefile'
    m.phony['run'] = 'python -m tssep.eval.run with config.yaml'
    m.phony['run_pdb'] = 'python -m tssep.eval.run --pdb with config.yaml'
    # m.phony['sbatch'] = 'python -m tssep.eval.run sbatch with config.yaml'
    m.phony['sbatch'] = cmd_to_hpc(
        'make run', shell=True, block=False, **dataclasses.asdict(slurm))
    m.phony['srun_debug'] = 'python -m tssep.eval.run srun_debug with config.yaml'

    # Diff commands only work for train:
    m.phony['meld'] = 'python -m tssep.eval.run meld with config.yaml'
    # m.phony['meld_breaking_change'] = 'python -m css.egs.extract.run meld with eg.trainer.storage_dir=.'  # Fix a breaking change
    m.phony['diff'] = 'python -m tssep.eval.run diff with config.yaml'

    # m.phony['der'] = [
    #     'python -m css.egs.extract.find_optimal_der .',
    #     'cat md-eval.pl_*'
    # ]

    # m.phony['c7json'] = [
    #     '# python -m css.egs.extract.eval_util.c7json audio',
    #     "# python -m meeteval.io.chime7 filter dia_c7.json '.*ch0'",
    #     "# python -m meeteval.io.chime7 resub dia_c7_filtered.json '(.*)_ch0' '\1'",
    #     'python -m css.egs.extract.eval_util.dia_soft_to_hard .',
    #     "python -m meeteval.io.chime7 add_missing dia_est_v3/dia_c7.json",
    #     'make gss',
    # ]

    # tmp = '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed.json'
    tmp = tssep.git_root / 'egs/libri_css/data/jsons/libriCSS_raw_chfiles.json'
    m.phony['gss'] = [
        'python -m tssep.eval.c7_frame_to_second c7.json',
        cmd_to_hpc(
            f"python -m tssep.eval.gss_v2 c7_fix.json {tmp} --out_folder=gss --channel_slice=:",
            job_name=f'{eval_id}_gss_v2',
            block=False,
            shell=True,
            time='24h',
            mem='4G',
            mpi='40',
        ),
        'make gss_makefile',
        # f"cd gss && sbatch.py -t 12h -n 200 --mem-per-cpu 4G --job-name {eval_id}_gss_v2 --wrap "
        # f"'srun python -m tssep.eval.gss_v2 ../c7.json {tmp} --out_folder=. --channel_slice=:",
    ]
    m.phony['gss_makefile'] = [
        'python -m fire tssep.eval.makefile gss_makefile gss',
    ]

    ref = tssep.git_root / 'egs/libri_css/data/libri_css_ref.stm'
    m['asr/ref.stm'] = [
        'mkdir -p asr',
        f'ln -s {ref} asr/ref.stm',
    ]
    m['asr/hyp.stm'] = [
        'mkdir -p asr',
        f'python -m tssep.eval.c7_frame_to_second c7.json --out=asr/hyp.json',
    ]

    m.phony['transcribe_tiny.en'] = [
        'mkdir -p asr',
        f'make asr/ref.stm asr/hyp.stm',
        'python -m tssep.eval.transcribe launch asr/hyp.json',
        f"meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r asr/ref.stm -h asr/hyp_whisper_tiny.en.json",
        f'cat asr/hyp_whisper_tiny.en_cpwer.json'
    ]

    m.phony['transcribe_large-v2'] = [
        f'mkdir -p asr',
        f'make asr/ref.stm asr/hyp.stm',
        f'python -m tssep.eval.transcribe launch asr/hyp.json --model_name="large-v2"',
        f"meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r asr/ref.stm -h asr/hyp_whisper_large-v2.json",
        f'cat asr/hyp_whisper_large-v2_cpwer.json'
    ]

    m.phony['transcribe_nemo'] = [
        f'mkdir -p asr',
        f'python -m tssep.eval.c7_frame_to_second c7.json --out=asr/hyp.json',
        f'make asr/ref.stm',
        cmd_to_hpc(
            f'python -m cbj.transcribe.cli chime7 asr/hyp.json --model_tag=nemo --key=audio_path',
            mem='3G', time='1h', mpi='20',
            job_name=f'nemo_asr',
            block=True,
            shell=True,
            shell_wrap=True,
        ),
        f"meeteval-wer cpwer --normalize='lower,rm(.?!,)' -r asr/ref.stm -h asr/hyp_nemo.json",
        f'cat asr/hyp_nemo.json'
    ]

    # add_asr_wer_to_makefile(m, eval_id)

    # for model_tag, resources in [
    #     ('espnet', dict(mem='5G', time='6h', mpi='20')),  # Noctua1: (2700 / mpi) minutes
    #     ('wavlm', dict(mem='8G', time='12h', mpi='40')),  # Noctua1: (1600 / mpi) minutes  (Something is wrong. WavLM was slower than espnet)
    #     ('nemo', dict(mem='3G', time='1h', mpi='20')),  # Noctua1: Very fast: (100 / mpi) minutes
    #     ('nemo_xxl', dict(mem='14G', time='6h', mpi='20')),  # Noctua1: (480 / mpi) minutes
    # ]:
    #     m.phony[f'asr_{model_tag}: c7.json'] = [
    #         cmd_to_hpc(
    #             f'python -m cbj.transcribe.cli chime7 c7.json --model_tag={model_tag} --key=audio_path && make wer_{model_tag}',
    #             **resources,
    #             job_name=f'{eval_id}_asr_{model_tag}',
    #             block=False,
    #             shell=True,
    #             shell_wrap=True,
    #         ),
    #     ]
    #     ref = tssep.git_root / 'egs/libri_css/data/libri_css_ref.stm'
    #     m.phony[f'wer_{model_tag}: c7_words_{model_tag}.json'] = [
    #         f'python -m tssep.eval.c7_frame_to_second c7_words_{model_tag}.json',
    #         f'python -m tssep.eval.normalize_words c7_words_{model_tag}_fix.json',
    #         f'python -m meeteval.io.chime7 to_stm c7_words_{model_tag}_fix.json > c7_words_{model_tag}_fix.stm',
    #         f'python -m meeteval.der md_eval_22 -r {ref} -h c7_words_{model_tag}_fix.stm',
    #         f'meeteval-wer cpwer -r {ref} -h c7_words_{model_tag}_fix.stm',
    #         f'cat c7_words_{model_tag}_fix_md_eval_22.json',
    #         f'cat c7_words_{model_tag}_fix_cpwer.json',
    #     ]


    # sbatch.py -t 12h -n 200 --mem-per-cpu 4G --wrap 'srun python -m css.egs.extract.gss_v2 dia_c7_removed_dummys.json /scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2.json --out_folder=mask_mul --gss_postfilter=mask_mul.2 --channel_slice=:'
    # sbatch.py -n 80 --time 12h -p largemem --mem-per-cpu 7G --wrap 'srun.py python -m cbj.transcribe.cli chime7 dia_est_v3/gss_c7.json --model_tag=espnet --key=audio_path'
    # m.phony['transcribe_c7json'] = [
    #    f"sbatch.py -n 80 --time 12h -p largemem --mem-per-cpu 7G --job-name {eval_id}_transcribe --wrap "
    #    "'srun.py python -m cbj.transcribe.cli chime7 dia_est_v3/gss_c7.json --model_tag=espnet --key=audio_path'"
    # ]
    # m.phony['c7json_wer'] = [
    #     'python -m meeteval.io.chime7 add_missing dia_est_v3/gss_c7_words.json',
    #     'meeteval-wer cpwer -r /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/all/*_ref.stm -h dia_est_v3/gss_c7_words.stm',
    #     'cat dia_est_v3/gss_c7_words_cpwer.json',
    # ]

    # m.phony['sbatch_asr_wavlm'] = [
    #     'sbatch.py -n 80 -c 1 --mem-per-cpu 7GB --time 8:0:0 -p normal --job-name 77_62000_55_ASR_enh --wrap "make mpirun_asr_wavlm"'
    # ]
    # m.phony['mpirun_asr_wavlm'] = [
    #     f'{asr_slurm.mpi_cmd} python -m css.egs.extract.json_to_transcription --output_folder=audio/wavlm --model=wavlm audio/*.json'
    # ]

    # m.phony['echo_der_wer'] = [
    #     f'@echo "    - {eval_dir.relative_to(storage_dir.parent)}"',
    #     '@echo "        - $$(tail -n 3 md-eval.pl_0L_0S_OV10_OV20_OV30_OV40.txt | head -n 1)"',
    #     '@echo "        - $$(tail -n 1 md-eval.pl_0L_0S_OV10_OV20_OV30_OV40.txt)"',
    #     '@echo "        - $$(tail -n 1 wer_enh_dev.txt)"',
    #     '@echo "        - $$(tail -n 1 wer_enh_eval.txt)"',
    # ]

    # m.phony['json_to_transcription_fast'] = [
    #     f'ls audio/*.json | xargs -I % sbatch -n 20 -p cpu,cpu_extra --job-name {name} --wrap "{slurm.mpi_cmd} python -m css.egs.extract.json_to_transcription %"'
    #     # f'sbatch -n 20 -p cpu,cpu_extra --job-name {name} --wrap "mpirun python -m css.egs.extract.json_to_transcription audio/*.json"'
    # ]

    # m.phony['get_wer_dia'] = [
    #     '# First execute rttm_to_transcription and wait for the job to finish',
    #     """python -m css.egs.extract.get_wer --folder=. --template='sad/{}_hyp/text'""",
    # ]

    # try:
    #     reader = _config['eeg']['reader']
    #     reader_factory: str = reader['factory'].split('.')[-1]
    # except (TypeError, KeyError):
    #     # TypeError: 'NoneType' object is not subscriptable
    #     reader = _config['eg']['trainer']['model']['reader']
    #     reader_factory: str = reader['factory'].split('.')[-1]

    # if reader_factory in ['LibriCSSRaw', 'LibriCSSNTlab']:
    #     wer_named_config = 'libricss'
    # elif reader_factory == 'SimLibriCSS':
    #     if reader['num_speakers'] == 8:
    #         wer_named_config = 'simlibricss'
    #     elif reader['num_speakers'] == 4:
    #         wer_named_config = 'simlibricss_4spk'
    #     else:
    #         # I don't like an exception here. A placeholder might be better.
    #         raise NotImplementedError(reader)
    # elif reader_factory in ['Sampler'] and 'chime6' in reader['readers']:
    #     wer_named_config = 'chime6'
    # elif reader_factory in ['SimpleEvalReader']:
    #     wer_named_config = 'chime6'
    # elif 'CHiME6'.lower() in reader_factory.lower():
    #     wer_named_config = 'chime6'
    # elif reader_factory == 'SMSWSJ':
    #     wer_named_config = 'smswsj  # ToDo: support WER for smsSMS-WSJ'
    # else:
    #     # I don't like an exception here. A placeholder might be better.
    #     raise NotImplementedError(reader)

    # m.phony['get_wer_enh'] = [
    #     '# First execute rttm_to_transcription and wait for the job to finish',
    #     # """python -m css.egs.extract.get_wer --folder=. --template='audio/{}/text'""",
    #     f"python -m css.egs.extract.get_wer with {wer_named_config}",
    # ]
    # m.phony['get_wer_enh_segments'] = [
    #     '# First execute rttm_to_transcription and wait for the job to finish',
    #     """python -m css.egs.extract.get_wer --folder=. --template='audio/{}/text' --ref_json=/mm1/boeddeker/libriCSS/libriCSS_segments_220706_compressed.json""",
    # ]

    m.phony['clear'] = [
        'rm -rf audio sad embedding md-eval.pl_*.txt details.json summary.yaml',
    ]

    m.phony['makefile'] = 'python -m tssep.eval.run makefile with config.yaml'

    m.dump(eval_dir / 'Makefile')


def gss_makefile(storage_dir='.'):
    m = SimpleMakefile()
    m += 'SHELL := /bin/bash'
    m.phony['help'] = 'cat Makefile'

    storage_dir = Path(storage_dir).resolve()
    parts = list(filter(lambda p: p not in ['eval'],storage_dir.parts[-4:]))
    id_ = '_'.join(parts)
    add_asr_wer_to_makefile(m, id_, file='gss_c7.json')

    m.phony['makefile'] = f'python -m fire tssep.eval.makefile gss_makefile .'

    m.dump(storage_dir / 'Makefile')


def add_asr_wer_to_makefile(
        m: SimpleMakefile,
        name,
        file='c7.json',
):
    file = Path(file)
    assert file.suffix == '.json', file
    name = file.with_suffix('')

    for model_tag, resources in [
        ('espnet', dict(mem='5G', time='6h', mpi='20')),  # Noctua1: (2700 / mpi) minutes
        ('wavlm', dict(mem='8G', time='12h', mpi='40')),  # Noctua1: (1600 / mpi) minutes  (Something is wrong. WavLM was slower than espnet)
        ('nemo', dict(mem='3G', time='1h', mpi='20')),  # Noctua1: Very fast: (100 / mpi) minutes
        ('nemo_xxl', dict(mem='14G', time='6h', mpi='20')),  # Noctua1: (480 / mpi) minutes
    ]:
        m.phony[f'asr_{model_tag}: {name}.json'] = [
            cmd_to_hpc(
                f'python -m cbj.transcribe.cli chime7 {name}.json --model_tag={model_tag} --key=audio_path && make wer_{model_tag}',
                **resources,
                job_name=f'{name}_asr_{model_tag}',
                block=False,
                shell=True,
                shell_wrap=True,
            ),
        ]
        ref = tssep.git_root / 'egs/libri_css/data/libri_css_ref.stm'
        m.phony[f'wer_{model_tag}: {name}_words_{model_tag}.json'] = [
            f'python -m tssep.eval.c7_frame_to_second {name}_words_{model_tag}.json',
            f'python -m tssep.eval.normalize_words {name}_words_{model_tag}_fix.json',
            f'cat {name}_words_{model_tag}_fix.json | python -m meeteval.io.chime7 to_stm > {name}_words_{model_tag}_fix.stm',
            f'python -m meeteval.der md_eval_22 -r {ref} -h {name}_words_{model_tag}_fix.stm',
            f'meeteval-wer cpwer -r {ref} -h {name}_words_{model_tag}_fix.stm',
            f'cat {name}_words_{model_tag}_fix_md_eval_22.json',
            f'cat {name}_words_{model_tag}_fix_cpwer.json',
        ]

    m.phony['libricss_wer_tables'] = [
        'python -m tssep.eval.libricss_wer_table *cpwer_per_reco.json --session=True'
    ]
