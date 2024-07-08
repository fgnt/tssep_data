"""
ToDO: Move this file to css.database.librispeech_ivectors

python -m css.database.librispeech_ivectors.calculate_ivector librispeech
python -m css.database.librispeech_ivectors.calculate_ivector simLibriCSS
python -m css.database.librispeech_ivectors.calculate_ivector simLibriCSS_v2
python -m css.database.librispeech_ivectors.calculate_ivector libriCSS
python -m css.database.librispeech_ivectors.calculate_ivector libriCSS_source

python -m css.database.librispeech_ivectors.calculate_ivector libriCSS_v2

python -m css.database.librispeech_ivectors.calculate_ivector libriCSS_espnet


python -m css.database.librispeech_ivectors.calculate_ivector simLibriCSS_v2 /mm1/boeddeker/db/sim_libri_css_4spk.json


    --ivector_dir=IVECTOR_DIR
        D...
    --storage_dir=STORAGE_DIR
        Default: '/mm1/boed...


"""


from pathlib import Path
import collections
import subprocess
import tempfile
import operator

import numpy as np

from lazy_dataset.database import JsonDatabase

import paderbox as pb
from paderbox.utils.mapping import Dispatcher
from paderbox.utils.iterable import zip as zip_strict
import operator

from tssep_data.io.kaldi import KaldiDataDumper, to_wav_scp_value


def load_rttm(file, example_id_mapper: callable = None):
    """
    >>> load_rttm([f'/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/{k}/rttm' for k in ['dev', 'eval']])  # doctest: +ELLIPSIS
    {'session0_CH0_0L': {'8': ArrayInterval("48000:93120, 1057440:1124160, 3317280:3396000, 4220160:4288800", shape=None), '2': ...}}

    >>> from css.egs.extract.example_id_mapping import LibriCSSIDMapper
    >>> load_rttm([f'/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/{k}/rttm' for k in ['dev', 'eval']], LibriCSSIDMapper().to_folder)  # doctest: +ELLIPSIS
    {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0': {'8': ArrayInterval("48000:93120, 1057440:1124160, 3317280:3396000, 4220160:4288800", shape=None), '2': ...}}


    """
    # One rttm file can describe one dataset, multiple datasets or a part of a dataset.
    # Hence assume the example_ids are unique und load all of them.
    #
    # Multiple datasets:
    #   LibriCSS has subsets (e.g. 0V10). ESPnet puts all in one file
    # Part of a dataset:
    #   Kaldi and ESPnet remove one session from LibriCSS and use it as dev dataset.
    #   This is not an official split, hence the split one dataset in two.
    if isinstance(file, (str, Path)):
        file = [file]
    data = pb.array.interval.from_rttm_str('\n'.join([Path(f).read_text().strip() for f in file]))

    if example_id_mapper:
        length = len(data)
        data = {
            example_id_mapper(k): v
            for k, v in data.items()
        }
        assert len(data) == length, (len(data), length)
    return data


class Foo:
    """
    This code is ugly. calculate_ivector_v2 has a cleaner interface
    and better logging mechanisms.

    ToDo: Cleanup this code. Many code is deprecated, but
          was used to create files, that I used.

    """

    def __init__(
            self,
            ivector_dir='/mm1/boeddeker/librispeech_v1_extractor/exp/nnet3_cleaned',  # Trained model
            storage_dir='/mm1/boeddeker/librispeech_v1_extractor',
            kaldi_eg_dir='/mm1/boeddeker/deploy/kaldi/egs/librispeech/s5',
            kaldi_number_of_jobs=16,
    ):
        self.ivector_dir = ivector_dir
        self.storage_dir = storage_dir
        self.kaldi_eg_dir = kaldi_eg_dir
        self.kaldi_number_of_jobs = kaldi_number_of_jobs

    def script(
            self,
            folder,
            dataset_name,  # folder / 'data' / dataset_name
            # out_dir,
    ):
        """

        >>>
        >>> Foo().script()

        """
        script = []
        # ToDo: Which script in the librispeech folder in Kaldi was the origin for this script?
        script.append(f'''
        
cd {self.kaldi_eg_dir}
. ./path.sh
. ./cmd.sh

nj={self.kaldi_number_of_jobs}

ivector_process_dir={folder / 'data' / dataset_name}
ivector_affix=
# sub_speaker_frames=6000
sub_speaker_frames=0


max_count=75

# Subsampling of the features
ivector_period=10

data_set=doesnotmatterDeleteMe

ivector_dir={self.ivector_dir}

out_dir={folder}/ivectors


#######################################################################
# compute mfcc for extracting ivectors
#######################################################################

steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$train_cmd" ${{ivector_process_dir}} {folder}/exp/make_mfcc {folder}/mfcc

steps/compute_cmvn_stats.sh ${{ivector_process_dir}}
utils/fix_data_dir.sh ${{ivector_process_dir}}

# some sort segments will be ignored when compute mfcc filter those segments' weights
#utils/filter_scp.pl ${{ivector_process_dir}}/utt2spk ${{ivector_process_dir}}/weights > ${{ivector_process_dir}}/weights.filter
#sort ${{ivector_process_dir}}/weights.filter > ${{ivector_process_dir}}/weights
#gzip -c ${{ivector_process_dir}}/weights > ${{ivector_process_dir}}/weights.gz


# tree ${{ivector_process_dir}}

#######################################################################
# extract ivector
#######################################################################

# Kaldi expects that $lang/phones.txt and $lang/phones/silence.csl exists, independent if it is used.
mkdir -p ${{ivector_process_dir}}/lang
mkdir -p ${{ivector_process_dir}}/lang/phones
touch ${{ivector_process_dir}}/lang/phones.txt
touch ${{ivector_process_dir}}/lang/phones/silence.csl


steps/online/nnet2/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
    --ivector-period $ivector_period --sub-speaker-frames $sub_speaker_frames \
    --max-count $max_count --compress false \
    ${{ivector_process_dir}} ${{ivector_process_dir}}/lang $ivector_dir/extractor \
    $out_dir

# ToDo: Use --compress false


#     ${{ivector_process_dir}}/weights.gz \
#    $ivector_dir/ivectors_${{data_set}}${{ivector_affix}}

    
# tree ${{ivector_process_dir}}

        ''')

        return '\n'.join(script)

    def create_ivectors(self, wav_scp: dict, utt2spk: dict, folder, dataset_name):
        spk2utt = collections.defaultdict(list)
        for k, v in utt2spk.items():
            spk2utt[v].append(k)

        script = self.script(folder, dataset_name)

        # folder.mkdir(exist_ok=True, parents=True)
        try:
            folder.mkdir(parents=True)
        except FileExistsError:
            from css.bash import c
            print(f'{c.Red}WARNING: Skip {folder}, because folder exist {c.Color_Off}')
            return

        script_path = folder / 'my_run.sh'
        script_path.write_text(script)

        data_folder = folder / 'data' / dataset_name
        data_folder.mkdir(exist_ok=True, parents=True)

        wav_scp = '\n'.join([f'{k} {v}' for k, v in sorted(wav_scp.items())]) + '\n'
        (data_folder / 'wav.scp').write_text(wav_scp)

        utt2spk = '\n'.join([f'{k} {v}' for k, v in sorted(
            utt2spk.items(),
            key=operator.itemgetter(1, 0),
            # Must be sorted based on speaker id. Kaldi suggests to prefix the utterance id by the speaker id,
            # but a sort based on utterance ID works than only if you used "{skp}-{utt}" and not "{skp}_{utt}".
            # To be more robust here, sort based on spk id followed by utt id.
            #
            # This does not work. Kaldi sorts in utils/fix_data_dir.sh the utt2spk file based on the first column (i.e. utt)
            # And then checks if the second column is sorted. Hence "_" does not always work for the concaternation.
            # Here, some numbers of ASCII letters, they are used to sort:
            # assert ord('-') == 45
            # assert ord('0') == 48
            # assert ord('A') == 65
            # assert ord('_') == 95
            # assert ord('a') == 97
            # So if the length of the speaker_id varies and contain numbers of capital letters, you may get a problem
            # with kaldi.
        )]) + '\n'
        (data_folder / 'utt2spk').write_text(utt2spk)

        spk2utt = '\n'.join([f'{k} {" ".join(sorted(v))}' for k, v in sorted(spk2utt.items())]) + '\n'
        (data_folder / 'spk2utt').write_text(spk2utt)

        subprocess.run(['bash', str(script_path)])

    def librispeech(self):

        db = JsonDatabase('/mm1/boeddeker/db/librispeech.json')

        for dataset_name in db.dataset_names:
            ds = db.get_dataset(dataset_name)

            # wav_scp = {
            #     '2830-3980-0071': '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/test/wav/test-clean/2830/3980/2830-3980-0071.wav',
            #     "6829-68769-0038": "/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/test/wav/test-clean/6829/68769/6829-68769-0038.wav",
            # }
            # utt2spk = {
            #     '2830-3980-0071': '2830',
            #     "6829-68769-0038": '6829',
            #
            # }
            #

            wav_scp = {
                # ex['example_id']: ex['audio_path']['observation']
                ex['example_id']: f"flac -c -d -s {ex['audio_path']['observation']} |"
                # ex['example_id']: Dispatcher({
                #     '.wav': f"flac -c -d -s {ex['audio_path']['observation']} |",
                # })[Path(ex['audio_path']['observation']).suffix]
                for ex in ds
            }
            utt2spk = {
                # https://github.com/kaldi-asr/kaldi/blob/aefbd096ec0c7f1136f669c99be66ac393afe29c/egs/librispeech/s5/local/data_prep.sh#L72
                #   Kaldi uses speaker_id and chapter for utt2spk
                # ex['example_id']: f"{ex['speaker_id']}-{ex['chapter_id']}"
                ex['example_id']: f"{ex['speaker_id']}"
                # ex['example_id']: f"{ex['example_id']}"
                for ex in ds
            }

            # spk2utt = collections.defaultdict(list)
            # for k, v in utt2spk.items():
            #     spk2utt[v].append(k)

            self.create_ivectors(
                wav_scp,
                utt2spk,
                Path(f'{self.storage_dir}/ivectors_per_spk/{dataset_name}'),
                dataset_name,
            )

            # # with tempfile.TemporaryDirectory(dir='/mm1/boeddeker/tmp') as tmpdir:
            # #     tmpdir = Path(tmpdir)
            #
            # # folder = Path(f'/mm1/boeddeker/tmp/ivectors/{dataset_name}')
            # # folder = Path(f'/mm1/boeddeker/librispeech_v1_extractor/ivectors/{dataset_name}')
            # folder = Path(f'/mm1/boeddeker/librispeech_v1_extractor/ivectors_per_spk/{dataset_name}')
            # folder.mkdir(exist_ok=True, parents=True)
            #
            # # ToDo: Use the code below
            # # self.create_ivectors(wav_scp, utt2spk, folder)
            #
            # script = self.script(folder, dataset_name)
            #
            # script_path = folder / 'my_run.sh'
            #
            # script_path.write_text(script)
            #
            # wav_scp = '\n'.join([f'{k} {v}' for k, v in sorted(wav_scp.items())]) + '\n'
            # (folder / 'wav.scp').write_text(wav_scp)
            #
            # utt2spk = '\n'.join([f'{k} {v}' for k, v in sorted(utt2spk.items())]) + '\n'
            # (folder / 'utt2spk').write_text(utt2spk)
            #
            # spk2utt = '\n'.join([f'{k} {" ".join(sorted(v))}' for k, v in sorted(spk2utt.items())]) + '\n'
            # (folder / 'spk2utt').write_text(spk2utt)
            #
            # subprocess.run(['bash', str(script_path)])

    def libriCSS(self):
        db = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json')

        for dataset_name in db.dataset_names:
            print(f'Process: {dataset_name}')
            ds = db.get_dataset(dataset_name)

            wav_scp = {}
            utt2spk = {}

            for ex in ds.sort(lambda ex: ex['example_id']):
                data = collections.defaultdict(lambda: pb.array.interval.zeros())
                for speaker_id, offset, num_samples in zip_strict(ex['speaker_id'], ex['offset'],
                                                                  ex['num_samples']['original_source'], strict=True):
                    offset = offset + ex['start']
                    data[speaker_id][offset:offset + num_samples] = 1

                data = dict(data)

                overlap = pb.array.interval.core._combine(lambda *args: sum(args) > 1, *data.values())

                overlap_free = {k: pb.array.interval.core._combine(operator.__and__, v, ~ overlap)
                                for k, v in data.items()}

                observation = ex['audio_path']['observation'][0]

                for spealer_id, v in overlap_free.items():
                    # v: pb.array.interval.ArrayInterval
                    for i, (s, e) in enumerate(v.normalized_intervals):

                        # 11368 is not enough (7 frames)
                        # 17362 is enough
                        # Maybe: 12800.0 = 8 * 0.010 (stft shift) * 10 (subsample) * 16000 (samplereate)
                        kaldi_utt_id = f"{ex['example_id']}_{spealer_id}_{i:04}"
                        # if (e - s) < 12800:
                        #     print(f'Skip {kaldi_utt_id}, because it has only {e-s} samples.')
                        #     continue

                        wav_scp[kaldi_utt_id] = f'sox -t wav {observation} -t wav - trim {s}s ={e}s |'
                        utt2spk[kaldi_utt_id] = f"{ex['example_id']}_{spealer_id}"

                        # ToDo: Remove small segments, because kaldi_io [1] can only read "large" matrices and fails to
                        #       read small matrices.
                        #       Kaldi changes the compression format for small matrices (around 8 rows (or cols?)).
                        #       [1] https://github.com/vesis84/kaldi-io-for-python

            self.create_ivectors(
                wav_scp, utt2spk,
                folder=Path(f'{self.storage_dir}/ivectors_oracle_libriCSS/{dataset_name}'),
                dataset_name=dataset_name,
            )

    def libriCSS_source(self):
        db = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json')

        for dataset_name in db.dataset_names:
            print(f'Process: {dataset_name}')
            ds = db.get_dataset(dataset_name)

            wav_scp = {}
            utt2spk = {}

            for ex in ds.sort(lambda ex: ex['example_id']):
                data = collections.defaultdict(lambda: pb.array.interval.zeros())
                for speaker_id, offset, num_samples in zip_strict(ex['speaker_id'], ex['offset'],
                                                                  ex['num_samples']['original_source'], strict=True):
                    offset = offset + ex['start']
                    data[speaker_id][offset:offset + num_samples] = 1

                data = dict(data)

                overlap = pb.array.interval.core._combine(lambda *args: sum(args) > 1, *data.values())

                overlap_free = {k: pb.array.interval.core._combine(operator.__and__, v, ~ overlap)
                                for k, v in data.items()}

                speaker_source = ex['audio_path']['speaker_source']

                for skp_idx, (spealer_id, v) in enumerate(overlap_free.items()):
                    # v: pb.array.interval.ArrayInterval
                    for i, (s, e) in enumerate(v.normalized_intervals):

                        # 11368 is not enough (7 frames)
                        # 17362 is enough
                        # Maybe: 12800.0 = 8 * 0.010 (stft shift) * 10 (subsample) * 16000 (samplereate)
                        kaldi_utt_id = f"{ex['example_id']}_{spealer_id}_{i:04}"
                        # if (e - s) < 12800:
                        #     print(f'Skip {kaldi_utt_id}, because it has only {e-s} samples.')
                        #     continue

                        wav_scp[kaldi_utt_id] = f'sox -t wav {speaker_source[skp_idx]} -t wav - trim {s}s ={e}s |'
                        utt2spk[kaldi_utt_id] = f"{ex['example_id']}_{spealer_id}"

                        # ToDo: Remove small segments, because kaldi_io [1] can only read "large" matrices and fails to
                        #       read small matrices.
                        #       Kaldi changes the compression format for small matrices (around 8 rows (or cols?)).
                        #       [1] https://github.com/vesis84/kaldi-io-for-python

            self.create_ivectors(
                wav_scp, utt2spk,
                folder=Path(f'{self.storage_dir}/ivectors_oracle_libriCSS_source/{dataset_name}'),
                dataset_name=dataset_name,
            )

    @staticmethod
    def generic_meeting_wav_scp_utt2spk(db, file_eval, dataset_names=None, rttm=None, example_id_mapper=None, absolut_offset=False, debug=False):
        """
        >>> from paderbox.utils.pretty import pprint
        >>> db = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json')
        >>> pprint(Foo().generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['observation'][0]", debug=True))
        Process: 0L
        ({'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930': 'sox -t wav /mm1/boeddeker/libriCSS/libri_css/data-orig/for_release/0L/overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0/record/raw_recording_ch0.wav -t wav - trim =48000s =89920s =1057214s =1118974s =3316544s =3392384s =4219810s =4285251s |'}},
         {'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930': 'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930'}})
        >>> pprint(Foo().generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['speaker_source'][speaker_idx]", debug=True))
        Process: 0L
        ({'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930': 'sox -t wav /mm1/boeddeker/libriCSS/libri_css/data-orig/for_release/0L/overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0/clean/each_spk_6930.wav -t wav - trim =48000s =89920s =1057214s =1118974s =3316544s =3392384s =4219810s =4285251s |'}},
         {'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930': 'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_6930'}})

        >>> from css.egs.extract.example_id_mapping import LibriCSSIDMapper
        >>> pprint(Foo().generic_meeting_wav_scp_utt2spk(
        ...     db,
        ...     "ex['audio_path']['speaker_source'][speaker_idx]",
        ...     dataset_names=db.dataset_names,
        ...     rttm=[f'/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/{k}/rttm' for k in ['dev', 'eval']],
        ...     example_id_mapper=LibriCSSIDMapper().to_folder,
        ...     debug=True))
        Process: 0L
        ({'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_8': 'sox -t wav /mm1/boeddeker/libriCSS/libri_css/data-orig/for_release/0L/overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0/clean/each_spk_6930.wav -t wav - trim =48000s =93120s =1057440s =1124160s =3317280s =3396000s =4220160s =4288800s |'}},
         {'0L': {'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_8': 'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0_8'}})


        >>> db = JsonDatabase('/mm1/boeddeker/db/sim_libri_css.json')
        >>> pprint(Foo().generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['observation'][0]", debug=True))
        Process: SimLibriCSS-dev
        ({'SimLibriCSS-dev': {'00_1272': 'sox -t wav /mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-dev/wav/00_ch0.wav -t wav - trim =0s =228120s =926725s =987552s =2157421s =2215852s =2771863s =2945740s |'}},
         {'SimLibriCSS-dev': {'00_1272': '00_1272'}})
        >>> pprint(Foo().generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['speaker_source'][speaker_idx]", debug=True))
        ({'SimLibriCSS-dev': {'00_1272': 'sox -t wav /mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-dev/wav/00_source1272.wav -t wav - trim =0s =228120s =926725s =987552s =2157421s =2215852s =2771863s =2945740s |'}},
         {'SimLibriCSS-dev': {'00_1272': '00_1272'}})

        """
        wav_scps = {}
        utt2spks = {}

        if rttm is None:
            if dataset_names is None:
                dataset_names = db.dataset_names
            # assert example_id_mapper is None, example_id_mapper
        else:
            assert dataset_names is not None and rttm is not None, (dataset_names, rttm)
            dataset_names = dataset_names
            rttm = load_rttm(rttm, example_id_mapper=example_id_mapper)

        for dataset_name in dataset_names:
            print(f'Process: {dataset_name}')
            ds = db.get_dataset(dataset_name)

            wav_scp = wav_scps[dataset_name] = {}
            utt2spk = utt2spks[dataset_name] = {}

            for ex in ds.sort(lambda ex: ex['example_id']):
                if rttm is None:
                    data = collections.defaultdict(lambda: pb.array.interval.zeros())
                    for speaker_id, offset, num_samples in zip_strict(ex['speaker_id'], ex['offset'],
                                                                      ex['num_samples']['original_source'], strict=True):
                        if isinstance(offset, int):
                            assert isinstance(num_samples, int), (type(num_samples), num_samples)
                            offset = [offset]
                            num_samples = [num_samples]

                        for o, n in zip_strict(offset, num_samples, strict=True):
                            if absolut_offset:
                                pass
                            else:
                                o = o + ex.get('start', 0)
                            data[speaker_id][o:o + n] = 1

                    data = dict(data)
                else:
                    data = rttm[ex['example_id']]

                overlap = pb.array.interval.core._combine(lambda *args: sum(args) > 1, *data.values())

                overlap_free = {k: pb.array.interval.core._combine(operator.__and__, v, ~ overlap)
                                for k, v in data.items()}

                for speaker_idx, (speaker_id, v) in enumerate(overlap_free.items()):

                    if len(v.normalized_intervals) == 0 and (ex['dataset'], ex['example_id'], speaker_id) in [
                        ('SimLibriCSS-dev', '23', '6345'),
                        ('SimLibriCSS-dev', '23', '7850'),
                        ('SimLibriCSS-dev', '36', '1919'),
                        ('SimLibriCSS-dev', '36', '422'),
                        ('SimLibriCSS-train', '0264', '149'),
                        ('SimLibriCSS-train', '0852', '2388'),
                        ('SimLibriCSS-train', '0887', '2854'),
                        ('SimLibriCSS-train', '1504', '8356'),
                        ('SimLibriCSS-train', '1551', '979'),
                        ('SimLibriCSS-train', '1671', '1901'),
                        ('SimLibriCSS-train', '1828', '8464'),
                        ('SimLibriCSS-train', '2030', '2790'),
                        ('SimLibriCSS-train', '2672', '428'),
                        ('SimLibriCSS-train', '2943', '567'),
                        ('SimLibriCSS-train', '3061', '2989'),
                        ('SimLibriCSS-train', '3121', '3261'),
                        ('SimLibriCSS-train', '3266', '147'),
                        ('SimLibriCSS-train', '3315', '5321'),
                        ('SimLibriCSS-train', '3535', '45'),
                        ('SimLibriCSS-train', '3630', '1161'),
                        ('SimLibriCSS-train', '3734', '1987'),
                        ('SimLibriCSS-train', '3758', '982'),
                        ('SimLibriCSS-train', '4170', '1183'),
                        ('SimLibriCSS-train', '4390', '895'),
                        ('SimLibriCSS-train', '4780', '7946'),
                        ('SimLibriCSS-train', '5127', '2270'),
                        ('SimLibriCSS-train', '5294', '6978'),
                        ('SimLibriCSS-train', '5534', '2003'),
                        ('SimLibriCSS-train', '5766', '151'),
                        ('SimLibriCSS-train', '6049', '6399'),
                        ('SimLibriCSS-train', '6053', '1572'),
                        ('SimLibriCSS-train', '6072', '27'),
                        ('SimLibriCSS-train', '6217', '6333'),
                        ('SimLibriCSS-train', '6279', '2882'),
                        ('SimLibriCSS-train', '6364', '3046'),
                        ('SimLibriCSS-train', '6408', '7739'),
                        ('SimLibriCSS-train', '6462', '3825'),
                        ('SimLibriCSS-train', '6474', '3088'),
                        ('SimLibriCSS-train', '6755', '2401'),
                        ('SimLibriCSS-train', '7163', '594'),
                        ('SimLibriCSS-train', '7215', '4222'),
                        ('SimLibriCSS-train', '7223', '8169'),
                        ('SimLibriCSS-train', '7447', '1161'),
                        ('SimLibriCSS-train', '7511', '6286'),
                        ('SimLibriCSS-train', '7537', '79'),
                        ('SimLibriCSS-train', '7541', '2339'),
                        ('SimLibriCSS-train', '7623', '254'),
                        ('SimLibriCSS-train', '7651', '1152'),
                        ('SimLibriCSS-train', '7668', '3955'),
                        ('SimLibriCSS-train', '7674', '6836'),
                        ('SimLibriCSS-train', '7776', '7320'),
                        ('SimLibriCSS-train', '7797', '7704'),
                        ('SimLibriCSS-train', '7871', '5183'),
                        ('SimLibriCSS-train', '7945', '37'),
                        ('SimLibriCSS-train', '8027', '1603'),
                        ('SimLibriCSS-train', '8097', '8143'),
                        ('SimLibriCSS-train', '8165', '6076'),
                        ('SimLibriCSS-train', '8178', '6088'),
                        ('SimLibriCSS-train', '8193', '1736'),
                        ('SimLibriCSS-train', '8213', '1341'),
                        ('SimLibriCSS-train', '8220', '8291'),
                        ('SimLibriCSS-train', '8233', '7796'),
                        ('SimLibriCSS-train', '8238', '3285'),
                        ('SimLibriCSS-train', '8318', '4179'),
                        ('SimLibriCSS-train', '8327', '6534'),
                        ('SimLibriCSS-train', '8356', '6184'),
                        ('SimLibriCSS-train', '8379', '8005'),
                        ('SimLibriCSS-train', '8434', '730'),
                        ('SimLibriCSS-train', '8460', '5280'),
                        ('SimLibriCSS-train', '8504', '5796'),
                        ('SimLibriCSS-train', '8506', '4090'),
                    ]:
                        pass
                        # assert len(v.normalized_intervals) == 0, v.normalized_intervals
                    elif len(v.normalized_intervals) == 0:
                        print((ex['dataset'], ex['example_id'], speaker_id))
                    else:
                        # v: pb.array.interval.ArrayInterval
                        file = eval(file_eval)

                        kaldi_utt_id = f"{ex['example_id']}_{speaker_id}"
                        wav_scp[kaldi_utt_id] = to_wav_scp_value(file, segments=v.normalized_intervals)
                        utt2spk[kaldi_utt_id] = f"{ex['example_id']}_{speaker_id}"

                    if debug:
                        break
                if debug:
                    break
            if debug:
                break

        return wav_scps, utt2spks

    def simLibriCSS_v2(self, json='/mm1/boeddeker/db/sim_libri_css.json'):
        json = Path(json)
        db = JsonDatabase(json)

        with pb.utils.debug_utils.debug_on(Exception):
            wav_scps, utt2spks = self.generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['observation'][0]")
        assert wav_scps.keys() == utt2spks.keys(), (wav_scps.keys(), utt2spks.keys())

        for dataset_name in wav_scps.keys():
            self.create_ivectors(
                wav_scps[dataset_name], utt2spks[dataset_name],
                folder=Path(f'{self.storage_dir}/ivectors_oracle_{json.name}_v2/{dataset_name}'),
                dataset_name=dataset_name,
            )

    def libriCSS_v2(self):
        db = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json')

        with pb.utils.debug_utils.debug_on(Exception):
            wav_scps, utt2spks = self.generic_meeting_wav_scp_utt2spk(db, "ex['audio_path']['observation'][0]")
        assert wav_scps.keys() == utt2spks.keys(), (wav_scps.keys(), utt2spks.keys())

        for dataset_name in wav_scps.keys():
            self.create_ivectors(
                wav_scps[dataset_name], utt2spks[dataset_name],
                folder=Path(f'{self.storage_dir}/ivectors_oracle_libriCSS_v2/{dataset_name}'),
                dataset_name=dataset_name,
            )

    def libriCSS_espnet(self):
        cfg = {
            'json_path': '/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json',
            'file_eval': "ex['audio_path']['observation'][0]",
            'rttms': [
                '/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/dev/rttm',
                '/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/eval/rttm',
            ],
            'folder': Path(f'{self.storage_dir}/ivectors_espnet_libriCSS_v2'),
        }

        from css.egs.extract.example_id_mapping import LibriCSSIDMapper
        db = JsonDatabase(cfg['json_path'])

        wav_scps, utt2spks = self.generic_meeting_wav_scp_utt2spk(
            db,
            cfg['file_eval'],
            dataset_names=db.dataset_names,
            rttm=cfg['rttms'],
            example_id_mapper=LibriCSSIDMapper().to_folder,
        )

        pb.io.dump(cfg, cfg['folder'] / 'config.yaml', mkdir=True, mkdir_exist_ok=False, mkdir_parents=True)

        for dataset_name in wav_scps.keys():
            self.create_ivectors(
                wav_scps[dataset_name], utt2spks[dataset_name],
                folder=cfg['folder'] / f'{dataset_name}',
                dataset_name=dataset_name,
            )

    def simLibriCSS(self):
        db = JsonDatabase('/mm1/boeddeker/db/sim_libri_css.json')

        for dataset_name in db.dataset_names:
            print(f'Process: {dataset_name}')
            ds = db.get_dataset(dataset_name)

            wav_scp = {}
            utt2spk = {}

            for ex in ds.sort(lambda ex: ex['example_id']):
                data = collections.defaultdict(lambda: pb.array.interval.zeros())
                for speaker_id, offset, num_samples in zip_strict(ex['speaker_id'], ex['offset'],
                                                                  ex['num_samples']['original_source'], strict=True):
                    assert isinstance(speaker_id, str), speaker_id
                    if isinstance(ex['offset'], int):
                        offset = offset + ex.get('start', 0)
                        data[speaker_id][offset:offset + num_samples] = 1
                    else:
                        for o, n in zip_strict(offset, num_samples, strict=True):
                            o = ex.get('start', 0) + o
                            data[speaker_id][o:o + n] = 1

                data = dict(data)

                overlap = pb.array.interval.core._combine(lambda *args: sum(args) > 1, *data.values())

                overlap_free = {k: pb.array.interval.core._combine(operator.__and__, v, ~ overlap)
                                for k, v in data.items()}

                observation = Path(ex['audio_path']['observation'][0])

                for spealer_id, v in overlap_free.items():
                    # v: pb.array.interval.ArrayInterval
                    for i, (s, e) in enumerate(v.normalized_intervals):
                        kaldi_exmaple_id = f"{ex['example_id']}-{spealer_id}-{i:04}"
                        if observation.suffix == '.wav':

                            wav_scp[kaldi_exmaple_id] = to_wav_scp_value(observation, s, e)
                            # wav_scp[kaldi_exmaple_id] = f'sox -t wav {observation} -t wav - trim {s}s {e}s |'
                        elif observation.suffixes[-2:] == ['.wav', '.gz']:
                            observation = observation.with_suffix('')  # remove the '.gz' and assume there is also the uncompressed file.
                            wav_scp[kaldi_exmaple_id] = to_wav_scp_value(observation, s, e)
                            # wav_scp[kaldi_exmaple_id] = f'sox -t wav {observation} -t wav - trim {s}s {e}s |'
                            # Might be slow with gunzip
                            # wav_scp[kaldi_exmaple_id] = f'gunzip -c {observation} | sox -t wav - -t wav - trim {s}s {e}s |'
                        else:
                            raise Exception(observation)
                        utt2spk[kaldi_exmaple_id] = f"{ex['example_id']}-{spealer_id}"

            self.create_ivectors(
                wav_scp, utt2spk,
                folder=Path(f'{self.storage_dir}/ivectors_oracle_simLibriCSS/{dataset_name}'),
                dataset_name=dataset_name,
            )


if __name__ == '__main__':
    import fire

    fire.Fire(Foo)
    # Foo().libriCSS()
    # Foo().libriCSS_source()
    # Foo().simLibriCSS()
