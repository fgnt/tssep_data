#!/usr/bin/env python
import os
import shlex
import shutil
import sys
from pathlib import Path

import meeteval.io
from tssep_data.util.slurm import cmd_to_hpc, CMD2HPC
from tssep_data.util.cmd_runner import touch, CLICommands, confirm, run, Green, Color_Off, Red, env_check, maybe_execute, user_select, add_stage_cmd


clicommands = CLICommands()


@clicommands
def libri_css():
    for_release_zip = Path('for_release.zip')
    for_release = Path('for_release')

    @maybe_execute(target=for_release)
    def _():
        # Here, only the for_release folder is used.
        # Hence, it is not necessary to execute the baseline code.

        @maybe_execute(target=for_release_zip)
        def _():
            link = 'https://docs.google.com/uc?export=download&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l'
            if shutil.which('gdown'):
                # Working gdown versions: 4.7.3, 5.2.0
                # Failed version: 3.15.0, 4.6.4
                run(['gdown', link])  # gdown version 3.15.0 and 4.6.4 produces access denied. You may want to update to "pip install 'gdown>=5'"
            else:
                print(f'{Red}Download the `for_release.zip` file manually '
                      f'(open {link!r} and click download) or '
                      f'install gdown (pip install gdown) to automatically '
                      f'download the libri_css data.{Color_Off}')
                sys.exit(1)

        run(['unzip', os.fspath(for_release_zip)])


@clicommands
def sim_libri_css():
    jsalt2020 = Path('jsalt2020_simulate')
    @maybe_execute(
        target=jsalt2020,
    )
    def _():
        run(['git', 'clone', 'https://github.com/boeddeker/jsalt2020_simulate'])

    exproot = jsalt2020 / 'simdata'
    librispeech = exproot / 'data-orig/LibriSpeech'
    if librispeech.exists():
        print(f'{librispeech} already exists, skipping librispeech download')
    else:
        for k, v in os.environ.items():
            if k.lower() == 'librispeech':
                print(f'Found librispeech path in environment (key: {k}) at', v)
                run(['ln', '-s', v, str(librispeech)])
                break
        else:
            env = {'EXPROOT': exproot.relative_to(jsalt2020)}
            if confirm(f"{librispeech} doesn't exist.\nDownload librispeech? (Answer no (or Ctrl-C) to get the command to create a symlink to an existing librispeech folder)"):
                run(['./download.sh'], env=env, cwd=jsalt2020)
            else:
                librispeech.parent.mkdir(exist_ok=True, parents=True)
                print('Create here a symlink to librispeech:')
                print(f'ln -s /path/to/librispeech {shlex.quote(str(librispeech))}')
                sys.exit(1)

    # First process dev, since it fails faster, when something doesn't work
    datasets = ['dev', 'train', 'test']

    data = exproot / 'data'

    @maybe_execute(
        target=data,
        target_samples=[
            data / dataset
            for dataset in datasets
        ]
    )
    def _():
        # if confirm("Preprocess librispeech? (i.e. convert flac to wav)"):
        # The user has no choise. preprocess.sh converts flac to wav,
        # changes folder structure and creates some auxillary files.
        env = {'EXPROOT': exproot.relative_to(jsalt2020)}
        (jsalt2020 / 'path.sh').touch()
        run(['./scripts/preprocess.sh'], env=env, cwd=jsalt2020)

    for dataset in datasets:
        # Note: Peak memory is below 10 GB for the last KALDI job
        # train dataset jobs need between 3 and 6 hours on our (Paderborn) cluster.
        dataset_dir = data / f'SimLibriCSS-{dataset}'
        @maybe_execute(
            done_file=True,
            overwrite_works=True,
            target=dataset_dir,
            target_samples=[
                dataset_dir / 'wav',
                dataset_dir / 'mixspec.json',
                dataset_dir / 'mixlog.json',
            ]
        )
        def _():
            env = {'EXPROOT': exproot.relative_to(jsalt2020)}
            (jsalt2020 / 'path.sh').touch()

            cpus = 32
            cmd2hpc = CMD2HPC(
                mpi=cpus,
                # default was 40, but 32 cores on a node is more likely to be supported on a cluster
                # Note: Local execution will use less cores, based on available cpus and mem.
                cpus=1,
                mem=f'{cpus*4}G',  # shared between all CPUs
                time='24h',  # Was Xh on Noctua2
            )

            cpus = cmd2hpc.mpi  # 32 on cluster and probably less on local machine
            cmd2hpc = cmd2hpc.replace(cpus=cpus, mpi=1)

            run(
                cmd2hpc(
                    [
                        './scripts/run_meetings.sh',
                        '--split', f'{cpus}',
                        '--dyncfg', os.path.relpath('simlibricss_meeting_dynamics.json', jsalt2020),
                        f'{dataset_dir.name}', f'{dataset}'
                    ],
                ),
                env=env, cwd=jsalt2020,
            )


@clicommands
def prepare_sim_libri_css():
    jsalt2020 = Path('jsalt2020_simulate')
    exproot = jsalt2020 / 'simdata'
    librispeech = exproot / 'data-orig/LibriSpeech'

    librispeech_json = Path('jsons/librispeech.json')
    @maybe_execute(target=librispeech_json)
    def _():
        run([
            sys.executable, '-m', 'mms_msg.databases.single_speaker.librispeech.create_json',
            '--json-path', f'{librispeech_json}',
            '--database-path', f'{librispeech}',
        ])

    sim_libri_css_json = Path('jsons/sim_libri_css.json')
    @maybe_execute(target=sim_libri_css_json)
    def _():
        run(
            [
                sys.executable, '-m', 'tssep_data.database.sim_libri_css.create_json',
                f'--librispeech_json={librispeech_json}',
                f'--folder={jsalt2020 / "simdata" / "data"}',
                f'--output_path={sim_libri_css_json}',
            ],
        )

    # First process dev, since it fails faster, when something doesn't work
    datasets = ['dev', 'train', 'test']

    tmp = Path(f'{jsalt2020}/simdata/data')

    for target_sample in [
        tmp / f'SimLibriCSS-train/wav/8271_ch0.wav',
        tmp / f'SimLibriCSS-dev/wav/90_ch0.wav',

        # I don't know why, but in the past I got 101 examples, now I get
        # 96 examples from the sim_libri_css code.
        # Since LibriCSS is the test dataset, it is not that important.
        # LibriCSS uses leading zeros, so the file is named 089_ch0.wav
        # or 89_ch0.wav.
        tmp / f'SimLibriCSS-test/wav/*89_ch0.wav',
    ]:
        @maybe_execute(
            name=f'split_files_{target_sample.relative_to(tmp).parts[0]}',
            target_samples=[target_sample], overwrite_works=True,
        )
        def split_files():
            # Split the original files to one file per channel (i.e. minimize the date that needs to be loaded)
            run(cmd_to_hpc(
                # f'{sys.executable} -m tssep_data.database.sim_libri_css.split_files {jsalt2020}/simdata/data/SimLibriCSS-*/wav/',
                f'{sys.executable} -m tssep_data.database.sim_libri_css.split_files {target_sample.parent}/',
                mpi=40,
                mem='2G',
                time='2h',  # Was 1h on Noctua2 for train
                shell=True,
            ))

    tmp = Path(f'{jsalt2020}/simdata/data')
    sim_libri_css_early_json = Path('jsons/sim_libri_css_early.json')
    @maybe_execute(
        target=sim_libri_css_early_json,
        source=sim_libri_css_json,
        target_samples=[
            tmp / f'SimLibriCSS-train/wav/8271speaker_reverberation_early_ch0.wav',
            tmp / f'SimLibriCSS-dev/wav/90speaker_reverberation_early_ch0.wav',
            tmp / f'SimLibriCSS-test/wav/*89speaker_reverberation_early_ch0.wav',
        ],
        overwrite_works=True,
    )
    def sim_libri_css_add_data(source):
        # Add `speaker_reverberation_early_ch0` signal. Database does not
        # contain a reverberant signal and plain on the fly computation is to
        # expensive (Would need some optimization like in MMS-MSG to be fast
        # enough)
        run(cmd_to_hpc(
            f'{sys.executable} -m tssep_data.database.sim_libri_css.add_data {source}',
            mpi=40,
            mem='6G',  # 4G might be enough, but I am not sure about peaks
            time='2h',  # Was 1.2h on Noctua2
            shell=True,
        ))

    tmp = Path('jsons/sim_libri_css_early/target_vad/v2')
    @maybe_execute(
        target_samples=[
            tmp / 'SimLibriCSS-dev.pkl',
            tmp / 'SimLibriCSS-test.pkl',
            tmp / 'SimLibriCSS-train.pkl',
        ],
        overwrite_works=True,
    )
    def sim_libri_css_create_vad():
        run(cmd_to_hpc(
            [
                sys.executable, '-m', 'tssep_data.database.sim_libri_css.create_vad',
                 'v2',
                 f'{sim_libri_css_early_json}',
                 '--key', '["audio_path"]["speaker_reverberation_early_ch0"]',
            ],
            mpi=10,
            mem='3G',
            time='1h',  # Was Xh on Noctua2
            shell=False,
            # force_local=True,
        ))

    @maybe_execute(target='jsons/sim_libri_css_ch.json')
    def _(target):
        target = Path(target)
        source = target.with_name(target.name.replace('_ch.json', '.json'))
        run([
            sys.executable, '-m', 'tssep_data.data.cli.json_split_multichannel',
            f'{source}',
            '--clear=False',
            '--original_id=original_id',
        ])

    tmp = Path(f'{jsalt2020}/simdata/data')
    sim_libri_css_ch_early_json = Path('jsons/sim_libri_css_ch_early.json')
    @maybe_execute(
        target=sim_libri_css_ch_early_json,
        target_samples=[
            tmp / f'SimLibriCSS-train/wav/8271speaker_reverberation_early_ch6.wav',
            tmp / f'SimLibriCSS-dev/wav/90speaker_reverberation_early_ch6.wav',
            tmp / f'SimLibriCSS-test/wav/*89speaker_reverberation_early_ch6.wav',
        ],
        overwrite_works=True,
    )
    def sim_libri_css_ch_add_data(target):
        # Add `speaker_reverberation_early_ch0` signal. Database does not
        # contain a reverberant signal and plain on the fly computation is to
        # expensive (Would need some optimization like in MMS-MSG to be fast
        # enough)
        target = Path(target)
        source = target.with_name(target.name.replace('_early.json', '.json'))
        run(cmd_to_hpc(
            f'{sys.executable} -m tssep_data.database.sim_libri_css.add_data {source}',
            mpi=80,
            mem='6G',  # 4G might be enough, but I am not sure about peaks
            time='5h',  # Was 3h+20m on Noctua1 with 80 cores
            shell=True,
        ))

    # python -m tssep_data.data.cli.json_split_multichannel sim_libri_css.json --clear=False --original_id=original_id

    for source in [
            'jsons/sim_libri_css.json',
            'jsons/sim_libri_css_ch.json',
            'jsons/sim_libri_css_early.json',
            'jsons/sim_libri_css_ch_early.json',
    ]:
        target = Path(f'{source}.gz')
        @maybe_execute(target=target, source=source)
        def compress_json(target, source):
            run(f'gzip -f -k {source}')


@clicommands
def prepare_libri_css():
    # run('just env_check', cwd='..')

    # libri_css_for_release = Path('libri_css/exp/data-orig/for_release')
    libri_css_for_release = Path('for_release')
    if not (libri_css_for_release / 'all_res.json').exists():
        print(f'{Red}Error: Something is wrong with the libri_css folder ({libri_css_for_release}).\n'
              f'{(libri_css_for_release / "all_res.json")} does not exist.{Color_Off}'
              )
        sys.exit(1)

    libriCSS_raw_json = Path('jsons/libriCSS_raw.json')
    @maybe_execute(target=libriCSS_raw_json)
    def _():
        run([
            sys.executable, '-m', 'tssep_data.database.libri_css.create_json',
            '--output_path', f'{libriCSS_raw_json}',
            f'--folder', f'{libri_css_for_release}',
        ])

    libriCSS_raw_chfiles_json = Path('jsons/libriCSS_raw_chfiles.json')
    @maybe_execute(
        target=libriCSS_raw_chfiles_json,
        target_samples=[
            libri_css_for_release / f'OV40/overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9/record/raw_recording_ch6.wav'
        ],
        overwrite_works=True,
    )
    def _():
        run(cmd_to_hpc(
            f'{sys.executable} -m tssep_data.database.libri_css.split_files True {libriCSS_raw_json}',
            mpi=1,
            mem='1G',
            time='1h',  # Was 2min with mpi=11 on Noctua2
        ))

    @maybe_execute(target='jsons/libriCSS_raw_chfiles_ch.json')
    def _(target):
        target = Path(target)
        source = target.with_name(target.name.replace('_ch.json', '.json'))
        assert source != target, (source, target)
        run([
            sys.executable, '-m', 'tssep_data.data.cli.json_split_multichannel',
            f'{source}',
            '--clear=False',
            '--original_id=original_id',
        ])

    @maybe_execute(target='libri_css_ref.stm')
    def _(target):
        # for_release_folder = Path('libri_css/exp/data-orig/for_release')
        for_release_folder = Path('for_release')
        assert for_release_folder.exists(), for_release_folder
        run(f'{sys.executable} -m tssep_data.database.libri_css.create_stm {for_release_folder} {target}')


@clicommands
def ivector():

    ivector_dir = Path('ivector/librispeech_v1_extractor')
    @maybe_execute(done_file=True, target=ivector_dir)
    def download_pretrained_librispeech_model():
        ivector_dir.mkdir(parents=True, exist_ok=True)
        run(
            'wget -c https://kaldi-asr.org/models/13/0013_librispeech_v1_extractor.tar.gz -O - | tar -xz',
            cwd=ivector_dir,
        )

    KALDI_ROOT = os.environ['KALDI_ROOT']

    cmd_sh = Path(f'{KALDI_ROOT}/egs/librispeech/s5/cmd.sh')
    cmd_sh = Path(f'{KALDI_ROOT}/egs/librispeech/s5/cmd.sh')
    @maybe_execute()
    def ask_about_parallel_kaldi_files():
        assert cmd_sh.exists(), cmd_sh

        if confirm(
                f'Is {os.fspath(cmd_sh)!r} and the conf folder already adjusted to your system? (No will trigger an interactive selection)'
        ):
            pass
        else:
            match user_select(
                    'What do you use? (Ctrl+C to abort and do it manually)',
                    ['run.pl', 'slurm.pl', 'queue.pl'],
            ):
                case 'run.pl':
                    cmd_sh.write_text(cmd_sh.read_text().replace('"queue.pl --', '"run.pl --'))
                case 'slurm.pl':
                    cmd_sh.write_text(cmd_sh.read_text().replace('"queue.pl --', '"slurm.pl --'))
                    print(f'Replaced "queue.pl" with "slurm.pl" in {cmd_sh}')
                    slurm_conf = cmd_sh.parent / 'conf/slurm.conf'
                    if not slurm_conf.exists():
                        import tssep_data
                        example_files = sorted(tssep_data.git_root.glob('tools/kaldi/conf/*slurm.conf'))
                        assert example_files, (tssep_data.git_root, 'tools/kaldi/conf/*slurm.conf')
                        file = user_select(
                            'Do you want to use an example slurm.conf? (Ctrl+C to abort and create it manually)',
                            {'Skip': 'Skip', **{os.fspath(f): f for f in example_files}},
                        )
                        if file != 'Skip':
                            slurm_conf.write_text(file.read_text())
                            print(f'Copied {file} to {slurm_conf}')
                case 'queue.pl':
                    pass
                case otherwise:
                    raise AssertionError(otherwise)

    for target, source in [
        ('ivector/simLibriCSS_oracle_ivectors.json', 'jsons/sim_libri_css.json'),
        ('ivector/simLibriCSS_ch_oracle_ivectors.json', 'jsons/sim_libri_css_ch.json'),
        ('ivector/libriCSS_oracle_ivectors.json', 'jsons/libriCSS_raw_chfiles.json'),  # It is not critical here, that the annotations have a small offset.
        ('ivector/libriCSS_ch_oracle_ivectors.json', 'jsons/libriCSS_raw_chfiles_ch.json'),
    ]:
        assert Path(source).exists(), source
        @maybe_execute(target=Path(target))
        def _(target):
            KALDI_ROOT = os.environ['KALDI_ROOT']
            cwd = Path.cwd()
            import paderbox as pb
            storage_dir = pb.io.new_subdir.get_new_subdir(target.parent)
            run([
                f'{sys.executable}', '-m', 'tssep_data.data.embedding.calculate_ivector_v2', 'with',
                f'eg.ivector_dir={cwd / ivector_dir}/exp/nnet3_cleaned',
                f"eg.ivector_glob='*/ivectors/ivector_online.scp'",
                f'eg.json_path={cwd}/{source}',
                f'eg.kaldi_eg_dir="{KALDI_ROOT}/egs/librispeech/s5"',
                f'eg.storage_dir="{storage_dir}"',
                f'eg.output_json={target.name}'
            ])

    # @maybe_execute(target=Path('ivector/libriCSS_oracle_ivectors.json'))
    # def _(target):
    #     cwd = Path.cwd()
    #     import paderbox as pb
    #     storage_dir = pb.io.new_subdir.get_new_subdir(target.parent)
    #
    #
    #     run([
    #         sys.executable, '-m', 'tssep_data.data.embedding.calculate_ivector_v2', 'with',
    #         f'eg.ivector_dir={cwd}/ivector/librispeech_v1_extractor/exp/nnet3_cleaned',
    #         f'eg.ivector_glob=*/ivectors/ivector_online.scp',
    #         f'eg.json_path={cwd}/jsons/libriCSS_raw_chfiles.json',
    #         f'eg.kaldi_eg_dir="{KALDI_ROOT}/egs/librispeech/s5"',
    #         f'eg.storage_dir="{storage_dir}"',
    #         f'eg.output_json={target.name}',
    #     ])

    cwd = Path.cwd()
    rttm_folder = cwd / Path('espnet_libri_css_diarize_spectral_rttm')

    if not rttm_folder.exists():
        run('git clone https://huggingface.co/datasets/boeddeker/espnet_libri_css_diarize_spectral_rttm')

    for file in [
        rttm_folder / 'dev.rttm',
        rttm_folder / 'eval.rttm',
    ]:
        # assert file.exists(), file
        target = file.parent / 'orig_id' / file.name
        @maybe_execute(target=target, source=file)
        def fix_exampleid(target, source):
            run(f'{sys.executable} -m tssep_data.libricss.fix_exampleid rttm {source} --out {target.parent}')
    # run(f'{sys.executable} -m tssep_data.libricss.fix_exampleid rttm {rttm_folder / "dev.rttm"} --out {rttm_folder / "orig_id"}')
    # run(f'{sys.executable} -m tssep_data.libricss.fix_exampleid rttm {rttm_folder / "eval.rttm"} --out {rttm_folder / "orig_id"}')

    for target, source in [
        (cwd / 'ivector/libriCSS_espnet_ivectors.json', cwd / 'jsons/libriCSS_raw_chfiles.json'),
        (cwd / 'ivector/libriCSS_ch_espnet_ivectors.json', cwd / 'jsons/libriCSS_raw_chfiles_ch.json'),
    ]:
        @maybe_execute(target=target)
        def _(target):
            if 'ch_espnet' in target.name:
                rttms = []
                for deveval in ['dev', 'eval']:
                    file = rttm_folder / 'orig_id' / f'{deveval}.rttm'
                    r = meeteval.io.RTTM.load(file)
                    new = [
                        line.replace(filename=line.filename + f'_ch{ch}')
                        for line in r
                        for ch in range(7)
                    ]
                    file = file.with_name(f'{deveval}_ch.rttm')
                    meeteval.io.RTTM(new).dump(file)
                    rttms.append(os.fspath(file))
            else:
                rttms = [
                    os.fspath(rttm_folder / 'orig_id' / f'dev.rttm'),
                    os.fspath(rttm_folder / 'orig_id' / f'eval.rttm'),
                ]

            import paderbox as pb
            storage_dir = pb.io.new_subdir.get_new_subdir(target.parent)

            run([
                sys.executable, '-m', 'tssep_data.data.embedding.calculate_ivector_v2',
                'with',
                f'eg.ivector_dir={cwd}/ivector/librispeech_v1_extractor/exp/nnet3_cleaned',
                f'eg.ivector_glob=*/ivectors/ivector_online.scp',
                f'eg.json_path={source}',
                f'eg.kaldi_eg_dir="{KALDI_ROOT}/egs/librispeech/s5"',
                f'eg.storage_dir="{storage_dir}"',
                f'eg.output_json={target.name}',
                # f'''eg.activity={{'type':'rttm','rttm':[{str(libriCSS_dev_rttm)!r},{str(libriCSS_eval_rttm)!r}]}}''',
                f'eg.activity={{"type":"rttm","rttm":{rttms!r}}}',
            ])


add_stage_cmd(clicommands)


def _create_makefile():
    """
    Dummy function to create the Makefile by executing this doctest
    Advantage of Makefile:
     - Compactly define all commands
     - Autocomplete

    >>> _create_makefile()  # doctest: +ELLIPSIS
    Wrote .../egs/libri_css/data/Makefile
    """
    clicommands.create_makefile(__file__)


if __name__ == '__main__':
    pwd = Path(__file__).parent
    if pwd != Path.cwd():
        print(f'WARNING: This script ({__file__}) should be executed in the parent folder.')
        print(f'WARNING: Changing directory to {pwd}.')
        os.chdir(pwd)

    import fire
    # run('just env_check', cwd='..')
    env_check()
    fire.Fire(clicommands.to_dict(), command=None if sys.argv[1:] else 'stage')
