"""


python -m css.database.librispeech_ivectors.calculate_ivector_v2 with libriCSS_espnet


python -m css.database.librispeech_ivectors.calculate_ivector_v2 with simLibriCSS4_oracle_chime6


python -m css.database.librispeech_ivectors.calculate_ivector_v2 with chime6_librispeechExtractor


python -m css.database.librispeech_ivectors.calculate_ivector_v2 with simLibriCSS4CHiME6Noise_oracle

python -m css.database.librispeech_ivectors.calculate_ivector_v2 with simLibriCSS4CHiME6Noise_CHiME6Extractor_oracle

python -m css.database.librispeech_ivectors.calculate_ivector_v2 with chime6_CHiME6Extractor

python -m css.database.librispeech_ivectors.calculate_ivector_v2 with chime6_librispeechExtractor_oracle

python -m css.database.librispeech_ivectors.calculate_ivector_v2 with chime6_CHiME6Extractor_oracle

python -m css.database.librispeech_ivectors.calculate_ivector_v2 with chime6_CHiME6Extractor_humanOracle
"""

import dataclasses
import decimal
import operator
import os

import sacred
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
import padertorch as pt
from pathlib import Path
import paderbox as pb

import lazy_dataset.database
import tssep.io.kaldi
# import tssep_data.io.kaldi

sacred.SETTINGS.CONFIG.READ_ONLY_CONFIG = False  # READ_ONLY causes some strange errors. Disable it.

# Define the experiment. Experiment is the main class for managing experiments.
ex = Experiment('Calculate I-Vectors')


# Create a ConfigScope with the config decorator
@ex.config
def config():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'storage_dir': None,
    }
    if eg['storage_dir'] is None:
        eg['storage_dir'] = pt.io.get_new_subdir(eg.get('json_dir', '/mm1/boeddeker/ivectors'))

    eg['json_dir'] = str(Path(eg['storage_dir']).parent)

    print(eg)
    pt.Configurable.get_config(eg)
    ex.observers.append(FileStorageObserver(Path(eg['storage_dir']) / 'sacred'))


librispeech_v1_extractor = {
    'ivector_dir': '/mm1/boeddeker/librispeech_v1_extractor/exp/nnet3_cleaned',
    'ivector_glob': '*/ivectors/ivector_online.scp',
}
chime6_extractor = {
    # 'ivector_dir': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/exp/nnet3_train_worn_simu_u400k_cleaned_rvb',
    'ivector_dir': '/scratch/hpc-prf-nt2/cbj/deploy/kaldi/egs/chime6/s5c_track2/exp/nnet3_train_worn_simu_u400k_cleaned_rvb/extractor',
    'ivector_glob': '*/ivectors/ivector_online.scp',
}

espnet_spectral_clustering = {
    'activity': {
        'type': 'rttm',
        'rttm': ['/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/dev/rttm',
                 '/mm1/boeddeker/deploy/espnet/egs/libri_css/asr1/exp/diarize_spectral/eval/rttm'],
    }
}

chime6_oracle = {
    'kaldi_number_of_jobs': 8,
    'json_path': '/mm1/boeddeker/db/chime6_beamformit_dereverb.json',
    'file_eval': "ex['audio_path']['observation']",  # This is single channel data
    'activity': {  # forced aligment targets for eval.
        'type': 'rttm',
        'rttm': ['/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/chime6_rttm/dev_rttm.scoring',
                 '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/chime6_rttm/eval_rttm.scoring'],
    }
}

chime6_oracle_human = {
    'kaldi_number_of_jobs': 8,
    'json_path': '/mm1/boeddeker/db/chime6_U06.json',  # Use the annotations inside this json file
    'file_eval': "ex['audio_path']['observation'][0]",
}

chime6_spectral_clustering = {
    'json_path': None,
    'file_eval': None,
    'kaldi_number_of_jobs': 8,
    'activity': {
        'type': 'kaldi',
        'data_dir': {
            'dev': {
                'segments': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/segments',
                'wav_scp': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/wav.scp',
                'utt2spk': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/utt2spk',
            },
            'eval': {
                'segments': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/eval_beamformit_dereverb_diarized_hires/segments',
                'wav_scp': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/eval_beamformit_dereverb_diarized_hires/wav.scp',
                'utt2spk': '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/eval_beamformit_dereverb_diarized_hires/utt2spk',
            },
        },
    },
}


@ex.named_config
def libriCSS_espnet():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json',
        'file_eval': "ex['audio_path']['observation'][0]",
        'output_json': 'libriCSS_meeting_espnet_ivectors.json',
        **espnet_spectral_clustering,
        **librispeech_v1_extractor,
    }


@ex.named_config
def libriCSS_espnet_chime6():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json',
        'file_eval': "ex['audio_path']['observation'][0]",
        'output_json': 'libriCSS_meeting_espnet_chime6_ivectors.json',
        **espnet_spectral_clustering,
        **chime6_extractor,
    }


@ex.named_config
def libriCSS_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json',
        'output_json': 'libriCSS_meeting_oracle_ivectors.json',
        **librispeech_v1_extractor,
    }


@ex.named_config
def simLibriCSS_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/db/sim_libri_css.json',
        'output_json': 'simLibriCSS_meeting_oracle_ivectors.json',
        **librispeech_v1_extractor,
    }


@ex.named_config
def simLibriCSS4_oracle_chime6():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/db/sim_libri_css.json',
        'output_json': 'simLibriCSS_meeting_oracle_chime6_ivectors.json',
        **chime6_extractor,
    }


@ex.named_config
def simLibriCSS4_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/db/sim_libri_css_4spk.json',
        'output_json': 'simLibriCSS4_meeting_oracle_ivectors.json',
        **librispeech_v1_extractor,
    }


@ex.named_config
def simLibriCSS4CHiME6Noise_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/db/sim_libri_css_4spk_plus_chime6_noise.json',
        'output_json': 'simLibriCSS4CHiME6Noise_meeting_oracle_ivectors.json',
        'file_eval': "ex['audio_path']['observation']",  # This is single channel data
        **librispeech_v1_extractor,
    }


@ex.named_config
def simLibriCSS4CHiME6Noise_CHiME6Extractor_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'json_path': '/mm1/boeddeker/db/sim_libri_css_4spk_plus_chime6_noise.json',
        'output_json': 'simLibriCSS4CHiME6Noise_meeting_oracle_chime6Extractor_ivectors.json',
        'file_eval': "ex['audio_path']['observation']",  # This is single channel data
        **chime6_extractor,
    }


@ex.named_config
def chime6_librispeechExtractor():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'output_json': 'chime6_spectralClustering_librispeechExtractor_ivectors.json',
        **chime6_spectral_clustering,
        **librispeech_v1_extractor,
    }


@ex.named_config
def chime6_librispeechExtractor_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'output_json': 'chime6_spectralClustering_oracle_librispeechExtractor_ivectors.json',
        **chime6_oracle,
        **librispeech_v1_extractor,
    }


@ex.named_config
def chime6_CHiME6Extractor():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'output_json': 'chime6_spectralClustering_chime6Extractor_ivectors.json',
        **chime6_spectral_clustering,
        **chime6_extractor,
    }


@ex.named_config
def chime6_CHiME6Extractor_oracle():
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        # Spectral clustering is wrong inside the name
        'output_json': 'chime6_spectralClustering_oracle_chime6Extractor_ivectors.json',
        **chime6_oracle,
        **chime6_extractor,
    }


@ex.named_config
def chime6_CHiME6Extractor_humanOracle():
    # chime6_CHiME6Extractor_oracle might be better, but has no training information.
    eg = {
        'factory': NNet3MeetingIVectorCalculator,
        'output_json': 'chime6_humanOracle_chime6Extractor_ivectors.json',
        **chime6_oracle_human,
        **chime6_extractor,
    }


def _combine_segment_wav_scp(
        segments_file,
        wav_scp_file,
        utt2spk_file,
):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> file1 = '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/segments'
    >>> file2 = '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/wav.scp'
    >>> file3 = '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/utt2spk'
    >>> pprint(_combine_segment_wav_scp(file1, file2, file3)[0])  # doctest: +ELLIPSIS
    {'S02_U06.ENH-1': 'sox -t wav /mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/enhan/dev_beamformit_u06/S02_U06.wav -t wav - trim =111.47 =112.52 =277.68 =279.18 =279.83 =291.43 ... =8871.80 =8875.92 =8884.15 =8889.02 =8890.06 =8891.18 |',
     ...}

    """
    from tssep.io.kaldi import to_wav_scp_value, load_utt2spk, load_segments, load_wav_scp

    utt2spk = load_utt2spk(utt2spk_file)
    segments = load_segments(segments_file)
    wav_scp = load_wav_scp(wav_scp_file)
    # return {
    #     utterance_id: (wav_scp[recording_id], segment_begin, segment_end)
    #     for utterance_id, (recording_id, segment_begin, segment_end) in segments.items()
    # }

    def recording_speaker_id(recording_id: str, speaker_id):
        if speaker_id.startswith(recording_id):
            return speaker_id
        else:
            return f'{recording_id}-{speaker_id}'

    out = {
        ((recording_id, utt2spk[utterance_id]), utterance_id): (decimal.Decimal(segment_begin), decimal.Decimal(segment_end))
        for utterance_id, (recording_id, segment_begin, segment_end) in segments.items()
    }
    assert len(out) == len(segments), ('Something went wrong, utterance_ids in segments not unique', len(out), len(segments))
    out = pb.utils.nested.deflatten(out, sep=None)

    out = {
        recording_speaker_id(recording_id, speaker_id): to_wav_scp_value(
            wav_scp[recording_id],
            segments=[
                (segment_begin, segment_end)
                for segment_begin, segment_end in segments.values()
            ], unit='second'
        )
        for (recording_id, speaker_id), segments in out.items()
    }

    # Convert utterance_id (i.e. segment of recording for one speaker) to recording id for a speaker.
    # This will drop some lines.
    utt2spk = {
        recording_speaker_id(segments[utterance_id][0], speaker_id): speaker_id
        for utterance_id, speaker_id in utt2spk.items()
    }

    return out, utt2spk


@dataclasses.dataclass
class NNet3MeetingIVectorCalculator(pt.Configurable):
    """
    A Kaldi nnet3 I-Vector calculator
    """
    storage_dir: Path  # The folder, where all the I-Vectors and anything else will be stored
    json_dir: Path  # At the end, create a symlink in this directory
    output_json: Path = 'meeting_ivectors.json'  # Relative to storage dir.

    ivector_dir: str = '/mm1/boeddeker/librispeech_v1_extractor/exp/nnet3_cleaned'  # Trained model
    json_path: str = '/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json'  # Contains information, where the wav files are, if activity is None, also use the segment information
    absolut_offset: bool = False  # The offset in the json can be absolut or relative to the start. e.g. LibriCSS has relative, while CHiME-6 has absolut values.
    file_eval: str = "ex['audio_path']['observation'][0]"  # How to get the observation, e.g. "ex['audio_path']['observation'][0]" or "ex['audio_path']['speaker_source'][speaker_idx]"
    ivector_glob: str = '*/ivectors/ivector_online.scp'  # How to find the calculated I-Vectors
    activity: dict = None  # Estimates for the activity of the speakers
    dataset_names: str = None  # Default all

    kaldi_eg_dir: str = '/mm1/boeddeker/deploy/kaldi/egs/librispeech/s5'  # cwd to run the kaldi commands

    kaldi_number_of_jobs: int = 16

    def __post_init__(self):
        self.storage_dir = Path(self.storage_dir)
        self.output_json = self.storage_dir / self.output_json

        assert os.path.isabs(self.ivector_dir), self.storage_dir
        assert os.path.isabs(self.json_path), self.json_path
        assert os.path.isabs(self.storage_dir), self.storage_dir
        assert os.path.isabs(self.kaldi_eg_dir), self.kaldi_eg_dir

    def __call__(self):
        self.calculate_ivectors()
        self.write_json()

    def calculate_ivectors(self):
        from tssep_data.data.embedding.calculate_ivector import Foo
        foo = Foo(
            ivector_dir=self.ivector_dir,
            storage_dir=self.storage_dir,
            kaldi_eg_dir=self.kaldi_eg_dir,
            kaldi_number_of_jobs=self.kaldi_number_of_jobs,
        )

        if self.activity is None or self.activity['type'] == 'rttm':
            from tssep_data.database.libri_css.example_id_mapping import LibriCSSIDMapper
            db = lazy_dataset.database.JsonDatabase(self.json_path)

            if self.dataset_names is None:
                dataset_names = db.dataset_names
            else:
                dataset_names = self.dataset_names
            if self.activity is None:
                # Oracle
                wav_scps, utt2spks = foo.generic_meeting_wav_scp_utt2spk(
                    db, self.file_eval, dataset_names=dataset_names,
                    example_id_mapper=lambda ex_id: LibriCSSIDMapper().to_folder(ex_id, default=ex_id),
                    absolut_offset=self.absolut_offset,
                )
            elif self.activity['type'] == 'rttm':
                wav_scps, utt2spks = foo.generic_meeting_wav_scp_utt2spk(
                    db, self.file_eval, dataset_names=dataset_names,
                    rttm=self.activity['rttm'],
                    example_id_mapper=lambda ex_id: LibriCSSIDMapper().to_folder(ex_id, default=ex_id),
                    absolut_offset=self.absolut_offset,
                )
            else:
                raise Exception('Should never happen', self.activity)
        elif self.activity['type'] == 'kaldi':
            assert self.json_path is None, self.json_path
            assert self.dataset_names is None, self.dataset_names
            assert self.file_eval is None, self.file_eval

            wav_scps, utt2spks = {}, {}
            for dataset_name, data_dir in self.activity['data_dir'].items():
                assert set(data_dir.keys()) <= {'segments', 'wav_scp', 'utt2spk'}
                segments = data_dir.get('segments', None)
                wav_scp = data_dir['wav_scp']
                utt2spk = data_dir['utt2spk']
                if segments:
                    wav_scp, utt2spk = _combine_segment_wav_scp(segments, wav_scp, utt2spk)
                else:
                    wav_scp = tssep.io.kaldi.load_wav_scp(wav_scp)
                    utt2spk = tssep.io.kaldi.load_utt2spk(utt2spk)

                wav_scps[dataset_name] = wav_scp
                utt2spks[dataset_name] = utt2spk
        else:
            raise NotImplementedError(self.activity)

        for dataset_name in wav_scps.keys():
            foo.create_ivectors(
                wav_scps[dataset_name], utt2spks[dataset_name],
                folder=self.storage_dir / 'ivectors' / f'{dataset_name}',
                dataset_name=dataset_name,
            )

    def write_json(self):
        # from css.database.librispeech_ivectors.create_meeting_json import load_ivectors
        dss = load_ivectors(self.storage_dir / 'ivectors', self.ivector_glob)

        datasets = {}
        for dataset_name, ds in dss.items():
            print(f'Process {dataset_name}')
            datasets[dataset_name] = ds

        print('Write', self.output_json)
        pb.io.dump({'datasets': datasets}, self.output_json)

        link_name = Path(self.json_dir) / self.output_json.name

        if link_name.is_symlink():
            import questionary
            if questionary.confirm(f"Change symlink {link_name} from {link_name.resolve()} to {self.output_json}?").ask():
                # unlink removes a file or symlink.
                # But I only want to replace a symlink,
                # hence first a check if it is a symlink.
                link_name.unlink()
        pb.io.symlink(self.output_json, link_name)


def load_ivectors(folder, glob):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> db = load_ivectors()
    >>> db.keys()
    dict_keys(['OV10', 'OV20', '0L', 'OV40', 'OV30', '0S'])
    >>> ds = db['OV10']
    >>> pprint(ds['overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1'])
    {'speaker_embedding_path': ['/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.1.ark:55',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.1.ark:399325',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.1.ark:649794',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.1.ark:875464',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.1.ark:1222733',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.2.ark:55',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.2.ark:201725',
      '/mm1/boeddeker/librispeech_v1_extractor/ivectors_oracle_libriCSS_v2/OV10/ivectors/ivector_online.2.ark:466195'],
     'example_id': 'overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1',
     'dataset': 'OV10',
     'speaker_id': ['1320', '1995', '260', '4992', '672', '6930', '8455', '8463'],
     'duration': [99.80038,
      62.62012,
      56.41862,
      86.79156,
      78.8495,
      50.33138,
      66.096,
      55.61412]}

    """
    from tssep.io.kaldi import load_wav_scp, load_utt2spk, load_utt2dur
    import numpy as np
    import kaldi_io

    folder = Path(folder)

    files = {}
    for path in folder.glob(glob):
        dataset_name = path.relative_to(folder).parts[0]

        data_folder = folder / dataset_name / 'data' / dataset_name

        utt2spk = load_utt2spk(data_folder / 'utt2spk')
        utt2dur = load_utt2dur(data_folder / 'utt2dur')

        value = {}
        for k, v in load_wav_scp(path).items():
            example_id, speaker_id = utt2spk[k].rsplit('_', maxsplit=1)
            if (example_id, speaker_id) not in value:
                value[example_id, speaker_id] = {
                    'speaker_embedding_path': v,
                    'example_id': utt2spk[k].rsplit('_', maxsplit=1)[0],
                    'dataset': dataset_name,
                    'speaker_id': utt2spk[k].rsplit('_', maxsplit=1)[1],
                    'duration': utt2dur[k],
                }

            else:
                print(v)
                print(value[example_id, speaker_id]['speaker_embedding_path'], flush=True)
            #     tmp1 = kaldi_io.read_mat(v)[0]
            #     tmp2 = kaldi_io.read_mat(value[example_id, speaker_id]['speaker_embedding_path'])[0]
            #
                np.testing.assert_equal(
                    kaldi_io.read_mat(v)[0],
                    kaldi_io.read_mat(value[example_id, speaker_id]['speaker_embedding_path'])[0],
                    err_msg=f'{example_id} {speaker_id} {k}'
                )

        value = pb.utils.nested.deflatten(value, sep=None)

        def convert(ex):
            keys = {tuple(v.keys()) for v in ex.values()}
            assert len(keys) == 1, keys
            keys = keys.pop()

            new = {}

            for k in keys:
                new[k] = [v[k] for v in ex.values()]

            for k in ['example_id', 'dataset']:
                assert len(set(new[k])) == 1, (k, new[k])
                new[k] = new[k][0]
            return new

        value = {example_id: convert(ex) for example_id, ex in value.items()}

        files[dataset_name] = value

    assert len(files), (folder, glob)

    r = {
        dataset_name: embeddings
        for dataset_name, embeddings in files.items()
    }
    assert len(r), (folder, glob)
    return r


@ex.command(unobserved=True)
def init(_config, _run):
    storage_dir = Path(_config['eg']['storage_dir'])
    print_config(_run)
    pt.io.dump_config(_config, storage_dir / 'config.yaml')


@ex.capture
def get_eg(_config) -> NNet3MeetingIVectorCalculator:
    eg = pt.Configurable.from_config(_config['eg'])
    return eg


@ex.command
def calculate_ivectors(_config, _run):
    init()
    get_eg().calculate_ivectors()


@ex.command
def write_json(_config, _run):
    init()
    get_eg().write_json()


@ex.automain
def main(_config, _run):
    init()
    get_eg()()
