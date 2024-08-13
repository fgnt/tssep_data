import os
import sys

# Fix doctest
if os.path.dirname(os.path.abspath(__file__)) == sys.path[0]:
    del sys.path[0]

import dataclasses
import functools
from pathlib import Path

import lazy_dataset
from lazy_dataset.database import JsonDatabase
import numpy as np

import paderbox as pb
import padertorch as pt

import tssep
import tssep_data

# define egs_dir and json_dir before importing data_hooks
from tssep_data.util.access import ItemAccessor
from tssep_data.data import data_hooks

from tssep_data.data.constants import json_dir, egs_dir, eg_dir


def _get_segment(num_samples, segment_num_samples,
                 minimum_segment_num_samples=None):
    """
    >>> np.random.seed(0)
    >>> [_get_segment(10, 5) for _ in range(5)]
    [(4, 9), (5, 10), (0, 5), (3, 8), (3, 8)]
    >>> np.random.seed(0)
    >>> [_get_segment(10, 9) for _ in range(5)]
    [(0, 9), (1, 10), (1, 10), (0, 9), (1, 10)]
    >>> [_get_segment(10, 10) for _ in range(5)]
    [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    >>> _get_segment(10, 12)
    Traceback (most recent call last):
    ...
    lazy_dataset.core.FilterException: (10, 12)
    >>> [_get_segment(10, 12, 8) for _ in range(5)]
    [(0, 10), (0, 10), (0, 10), (0, 10), (0, 10)]
    >>> _get_segment(10, 12, 5)
    (0, 10)
    >>> _get_segment(10, 12, 10)
    (0, 10)
    >>> _get_segment(10, 12, 11)
    Traceback (most recent call last):
    ...
    lazy_dataset.core.FilterException: (10, 12, 11)

    """
    assert isinstance(num_samples, int), num_samples
    maximum = num_samples - segment_num_samples + 1
    if maximum <= 0:
        if minimum_segment_num_samples is None:
            raise lazy_dataset.FilterException(num_samples,
                                               segment_num_samples)

        maximum = num_samples - minimum_segment_num_samples + 1
        if maximum <= 0:
            raise lazy_dataset.FilterException(num_samples,
                                               segment_num_samples,
                                               minimum_segment_num_samples)
        else:
            return 0, num_samples

    start = np.random.choice(maximum)
    stop = start + segment_num_samples
    assert isinstance(start, int) and isinstance(stop, int), (start, stop)
    return start, stop


@dataclasses.dataclass
class PBJsonDSMeta(pt.Configurable):
    json_path: str  # = '.../tssep_data/egs/libri_css/data/jsons/<...>.json'
    dataset_name: str
    num_speakers: int
    sample_rate: int = 16_000
    observation: (str, slice) = "['observation'][:1]"

    def get_dataset(self, reader: 'Reader'):
        ds = reader.get_db(self.json_path).get_dataset(self.dataset_name)

        if self.num_speakers is not None:
            orig_len = len(ds)
            ds = ds.filter(lambda ex: ex.get('num_speakers', self.num_speakers) == self.num_speakers, lazy=False)
            assert len(ds) > orig_len // 3, (ds, orig_len)
        ds = ds.map(self.add_uem)

        return ds

    def add_uem(self, ex):
        """
        Use the key "uem" to specify the scoring region of the example.
        This may differ from the region, that is loaded at training time.
        Usually the uem specifies the maximum loaded region, because
        that is the scoring region, and it is allowed to be known at test time.

        Why the name "uem"?
        NIST defines the un-partitioned evaluation map (UEM) file format.
        It contains the information for each "filename"/example, which regions
        should be scored, e.g. the regions which will be used to calculate
        the DER.
        While that file contains this information for all "filename"s/examples,
        in the examples I use them to store only the information for this
        particular example.
        """
        # un-partitioned evaluation map (UEM)
        if 'uem' not in ex:
            ex['uem'] = ex.get('start', 0), ex.get('end', None)
        return ex

    def load_audio(self, ex, load_keys=['observation']):
        start = ex.get('start', 0)
        end = ex.get('end', None)

        ex.setdefault('audio_data', {})
        for k in load_keys:
            if k not in ex['audio_data']:
                k_ = self.observation if k == 'observation' else k
                ex['audio_data'][k] = pb.io.recursive_load_audio(
                    ItemAccessor(k_)(ex['audio_path']), expected_sample_rate=self.sample_rate,
                    start=start, stop=end)
            elif k in ['vad']:
                # Might be already added by a hook.
                assert ex['audio_data'][k].shape[-1] == end - start, (ex['audio_data'][k].shape, end - start, start, end)
            else:
                raise RuntimeError(k, ex['audio_data'].keys())

        return ex


@dataclasses.dataclass
class SegmentPBJsonDSMeta(PBJsonDSMeta):
    segment_num_samples: 'int | None' = None
    minimum_segment_num_samples: 'bool | None' = None
    mixup: 'bool | float | None' = None  # 1 = 100% mixup, 0 = 0% mixup

    def get_mixup(self):
        mixup = self.mixup
        if mixup is True or mixup is False:
            pass
        elif isinstance(mixup, float):
            assert 0 < mixup < 1, mixup
            mixup = (np.random.uniform() <= mixup)
        elif isinstance(mixup, int) and mixup == 0:
            mixup = False
        elif isinstance(mixup, int) and mixup == 1:
            mixup = True
        elif mixup is None:
            mixup = False
        else:
            raise TypeError(type(mixup), mixup)
        return mixup

    def load_audio(
            self,
            ex,
            load_keys=['observation'],
    ):
        start = ex.get('start', 0)
        end = ex.get('end', None)

        ex['mixup'] = self.get_mixup()

        if self.segment_num_samples is not None:
            if end is None:
                num_samples = ex['num_samples']['observation']
                end = start + num_samples
            ex['orig_start'] = start
            ex['orig_end'] = end

        if self.segment_num_samples:
            segment_num_samples = self.segment_num_samples
            if ex['mixup']:
                segment_num_samples = segment_num_samples * 2
            start_, end_ = _get_segment(
                end - start, segment_num_samples, minimum_segment_num_samples=self.minimum_segment_num_samples,
            )
            start, end = start + start_, start + end_
        else:
            assert not ex['mixup'], (self.mixup, self.segment_num_samples)

        ex.setdefault('audio_data', {})
        for k in load_keys:
            k_ = self.observation if k == 'observation' else k
            if k not in ex['audio_data']:
                ex['audio_data'][k] = pb.io.recursive_load_audio(
                    ItemAccessor(k_)(ex['audio_path']), expected_sample_rate=self.sample_rate,
                    start=start, stop=end)
            elif k in ['vad']:

                if isinstance(ex['audio_data'][k], (tuple, list)):
                    ex['audio_data'][k] = np.array([
                        e[start:end] for e in ex['audio_data'][k]])
                elif isinstance(ex['audio_data'][k], np.ndarray):
                    ex['audio_data'][k] = ex['audio_data'][k][..., start:end]
                else:
                    raise TypeError(type(ex['audio_data'][k]), k, ex['audio_data'][k])
                assert ex['audio_data'][k].shape[-1] == end - start, (ex['audio_data'][k].shape, end - start, start, end)
            else:
                raise RuntimeError(k, ex['audio_data'].keys())

            if ex['mixup']:
                if isinstance(ex['audio_data'][k], np.ndarray):
                    tmp = ex['audio_data'][k].shape[-1] // 2
                    ex['audio_data'][k] = ex['audio_data'][k][..., :tmp] + \
                                          ex['audio_data'][k][..., tmp:2 * tmp]
                elif isinstance(ex['audio_data'][k], (tuple, list)) and k in ['vad']:
                    ex['audio_data'][k] = [
                        e[:tmp] + e[tmp:2 * tmp]
                        for e in ex['audio_data'][k]
                    ]
                else:
                    raise TypeError(type(ex['audio_data'][k]), k, ex['audio_data'][k])

        ex['start'] = start
        ex['end'] = end

        return ex


class DatasetCfgs:
    @staticmethod
    def sim_libri_css():
        """
        >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
        >>> dsmeta = reader.datasets['SimLibriCSS-train-960_000']
        >>> dsmeta  # doctest: +ELLIPSIS
        SegmentPBJsonDSMeta(json_path='.../jsons/sim_libri_css_early.json', dataset_name='SimLibriCSS-train', num_speakers=8, sample_rate=16000, observation="['observation'][:1]", segment_num_samples=960000, minimum_segment_num_samples=None, mixup=0.5)
        >>> ds = reader.__call__('SimLibriCSS-train-960_000')
        >>> ds  # doctest: +ELLIPSIS
                DictDataset(name='SimLibriCSS-train', len=8461)
              MapDataset(_pickle.loads)
            SliceDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...])
          MapDataset(<bound method PBJsonDSMeta.add_uem of SegmentPBJsonDSMeta(json_path='.../jsons/sim_libri_css_early.json', dataset_name='SimLibriCSS-train', num_speakers=8, sample_rate=16000, observation="['observation'][:1]", segment_num_samples=960000, minimum_segment_num_samples=None, mixup=0.5)>)
        MapDataset(functools.partial(<bound method Reader.prepare of Reader(datasets={...}, ..., data_hooks=...)>, load_audio_fn=<bound method SegmentPBJsonDSMeta.load_audio of SegmentPBJsonDSMeta(..., dataset_name='SimLibriCSS-train', ...)>, load_audio=True, load_keys=['observation']))
        >>> print(pb.utils.pretty.pretty(ds[0], max_seq_length=[20, 2, 1]))
        {'session_id': 'S03',
         'audio_path': {'observation': ['/scratch/hpc-prf-nt2/cbj/deploy/espnet_chime7/egs2/chime7_task1/diar_asr1/chime7_task1/chime6/audio/train/S03_U01.CH1.wav',
           ...],
          'worn': {'P09': '/scratch/hpc-prf-nt2/cbj/deploy/espnet_chime7/egs2/chime7_task1/diar_asr1/chime7_task1/chime6/audio/train/S03_P09.wav',
           ...}},
         'sampling_rate': 16000,
         'num_samples': {'original_source': [45440, ...]},
         'speaker_id': ['P12', 'P09', ...],
         'offset': [920800, 951680, ...],
         'transcription': ['[noise] What were we talking about again? [inaudible 0:00:58.96]',
          '[laughs]',
          ...],
         'kaldi_transcription': ['what were we talking about again', '', ...],
         'start': 920800,
         'end': 126140144,
         'example_id': 'S03',
         'dataset': 'train_chime6',
         'audio_data': {'observation': array(shape=(1, 480000), dtype=float64)}}

        """
        datasets = {}
        kwargs = {
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json',
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_fix8spk.json',
            'json_path': '{egs_dir}/libri_css/data/jsons/sim_libri_css_early.json',  #
            'observation': "['observation'][:1]",
            'num_speakers': 8,
        }
        datasets['SimLibriCSS-train-960_000'] = {
            'factory': SegmentPBJsonDSMeta,
            'dataset_name': 'SimLibriCSS-train',
            'segment_num_samples': 960_000,
            'minimum_segment_num_samples': None,  # Samples have to match
            'mixup': 0.5,
            **kwargs,
        }
        datasets['SimLibriCSS-dev-2_400_000'] = {
            'factory': SegmentPBJsonDSMeta,
            'dataset_name': 'SimLibriCSS-dev',
            'segment_num_samples': 2_400_000,
            'minimum_segment_num_samples': 0,  # Use all data
            **kwargs,
        }
        for name in ['SimLibriCSS-dev', 'SimLibriCSS-test']:
            datasets[name] = {
                'factory': PBJsonDSMeta, 'dataset_name': name, **kwargs,
            }

        for k in list(datasets.keys()):
            datasets[f'{k}_ch'] = {
                **datasets[k],
                # 'json_path': json_dir / 'sim_libri_css_ch_speaker_reverberation_early_fix8spk.json',
                'json_path': '{egs_dir}/libri_css/data/jsons/sim_libri_css_ch_early.json',  # single channel, i.e., split the channels to have 7 times more
                # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early_fix8spk.json',
                # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch.json',
            }
        return datasets

    @staticmethod
    def libri_css():
        """
        >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
        >>> dsmeta = reader.datasets['libri_css']
        >>> dsmeta
        PBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed.json', dataset_name=['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40'], num_speakers=8, sample_rate=16000, observation="['observation'][:1]")
        >>> ds = reader.__call__('libri_css')
        >>> ds
                DictDataset(name='0S', len=10)
              MapDataset(_pickle.loads)
                DictDataset(name='0L', len=10)
              MapDataset(_pickle.loads)
                DictDataset(name='OV10', len=10)
              MapDataset(_pickle.loads)
                DictDataset(name='OV20', len=10)
              MapDataset(_pickle.loads)
                DictDataset(name='OV30', len=10)
              MapDataset(_pickle.loads)
                DictDataset(name='OV40', len=10)
              MapDataset(_pickle.loads)
            ConcatenateDataset()
          SliceDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...])
        MapDataset(functools.partial(<bound method PBJsonDSMeta.load_audio of PBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed.json', dataset_name=['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40'], num_speakers=8, sample_rate=16000, observation="['observation'][:1]")>, load_keys=['observation']))
        >>> print(pb.utils.pretty.pretty(ds[0], max_seq_length=[20, 2, 1]))
        {'meeting_id': 'overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0',
         'audio_path': {'observation': ['/scratch/hpc-prf-nt2/cbj/deploy/libri_css/exp/data-orig/for_release/0S/overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0/record/raw_recording_ch0.wav',
           ...],
          'speaker_source': ['/scratch/hpc-prf-nt2/cbj/deploy/libri_css/exp/data-orig/for_release/0S/overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0/clean/each_spk_908.wav',
           ...]},
         'speaker_source_indices': [0, 1, ...],
         'offset': [51, 218969, ...],
         'start': 47949,
         'end': 9651549,
         'source_id': ['908-157963-0004', '672-122797-0018', ...],
         'num_samples': {'observation': 9603600, 'original_source': [217280, ...]},
         'transcription': ['THEL IS LIKE A WATRY BOW AND LIKE A PARTING CLOUD LIKE A REFLECTION IN A GLASS LIKE SHADOWS IN THE WATER LIKE DREAMS OF INFANTS LIKE A SMILE UPON AN INFANTS FACE',
          'REJOICE IN OUR PRESENCE SAID THE AIR AND THE SUNLIGHT',
          ...],
         'speaker_id': ['908', '672', ...],
         'example_id': 'overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0',
         'dataset': '0S',
         'audio_data': {'observation': array(shape=(1, 9603600), dtype=float64)}}
        """
        datasets = {}
        kwargs = {
            'factory': PBJsonDSMeta,
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed.json',
            'json_path': '{egs_dir}/libri_css/data/jsons/libriCSS_raw_chfiles.json',  # observation is split into multiple files, one per microphone
            # 'observation': '["observation"][:1]',
            'observation': '["observation"]',
            'dataset_name': ['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40'],
            'num_speakers': 8,
        }
        datasets['libri_css'] = {
            **kwargs,
        }
        datasets['libri_css_ch0'] = {
            'observation': '["observation"][:1]',
            **kwargs,
        }
        datasets['libri_css_ch'] = {
            **kwargs,
            'json_path': '{egs_dir}/libri_css/data/jsons/libriCSS_raw_chfiles_ch.json',
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed_ch.json',
        }

        return datasets

    @staticmethod
    def chime7():
        """
        >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
        >>> dsmeta = reader.datasets['c7_dev_chime6']
        >>> dsmeta
        PBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2_ch.json', dataset_name='dev_chime6', num_speakers=4, sample_rate=16000, observation="['observation'][:1]")
        >>> ds = reader.__call__('c7_dev_dipco')
        >>> ds
              DictDataset(name='dev_dipco', len=175)
            MapDataset(_pickle.loads)
          SliceDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...])
        MapDataset(functools.partial(<bound method PBJsonDSMeta.load_audio of PBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2_ch.json', dataset_name='dev_dipco', num_speakers=4, sample_rate=16000, observation="['observation'][:1]")>, load_keys=['observation']))

        >>> print(pb.utils.pretty.pretty(ds[0], max_seq_length=[20, 2, 1]))
        {'session_id': 'S02',
         'audio_path': {'observation': ['/scratch/hpc-prf-nt2/cbj/deploy/espnet_chime7/egs2/chime7_task1/diar_asr1/chime7_task1/chime6/audio/dev/S02_U01.CH1.wav']},
         'sampling_rate': 16000,
         'start': 649600,
         'end': 142507904,
         'embedding_id': 'S02',
         'example_id': 'S02_U01CH1',
         'dataset': 'dev_chime6',
         'audio_data': {'observation': array(shape=(1, 141858304), dtype=float64)}}

        >>> print(pb.utils.pretty.pretty(reader.__call__('c7_train_chime6_480_000')[0], max_seq_length=[20, 2, 1]))
        {'session_id': 'S26',
         'audio_path': {'observation': ['/scratch/hpc-prf-nt2/cbj/deploy/espnet_chime7/egs2/chime7_task1/diar_asr1/chime7_task1/dipco/audio/dev/S26_U01.CH1.wav']},
         'sampling_rate': 16000,
         'start': 0,
         'end': 28809504,
         'embedding_id': 'S26',
         'example_id': 'S26_U01CH1',
         'dataset': 'dev_dipco',
         'audio_data': {'observation': array(shape=(1, 28809000), dtype=float64)}}

        """
        datasets = {}
        kwargs = {
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7.json',  # no eval
            # 'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2.json',  # with eval
            'json_path': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2_ch.json',  # with eval and channel split
            'observation': "['observation'][:1]",
            'num_speakers': 4,
        }
        datasets['c7_train_chime6_480_000'] = {
            'factory': SegmentPBJsonDSMeta,
            'dataset_name': 'train_chime6',
            'segment_num_samples': 480_000,
            'minimum_segment_num_samples': None,  # Samples have to match
            **kwargs,
        }
        datasets['c7_dev_chime6_2_400_000'] = {
            'factory': SegmentPBJsonDSMeta,
            'dataset_name': 'dev_chime6',
            'segment_num_samples': 2_400_000,
            'minimum_segment_num_samples': None,  # Samples have to match
            **kwargs,
        }

        for name in [
            'dev_chime6', 'dev_dipco', 'dev_mixer6',
            'eval_chime6', 'eval_dipco', 'eval_mixer6',
        ]:
            datasets[f'c7_{name}'] = {
                'factory': PBJsonDSMeta,
                'dataset_name': name,
                **kwargs,
            }
        return datasets

    @staticmethod
    def librispeech():
        """
        >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
        >>> dsmeta = reader.datasets['librispeech_dev_clean']
        >>> dsmeta
        PBJsonDSMeta(json_path='{egs_dir}/libri_css/data/jsons/librispeech.json', dataset_name='dev_clean', num_speakers=1, sample_rate=16000, observation="['observation']")
        >>> ds = reader.__call__('librispeech_dev_clean')
        >>> ds
                DictDataset(name='dev_clean', len=2703)
              MapDataset(_pickle.loads)
            SliceDataset([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...])
          MapDataset(<bound method PBJsonDSMeta.add_uem of PBJsonDSMeta(...)>)
        MapDataset(functools.partial(<bound method Reader.prepare of Reader(...))
        >>> print(pb.utils.pretty.pretty(ds[0], max_seq_length=[20, 2, 1]))
        {'audio_path': {'observation': '/scratch/hpc-prf-nt2/fgnt/net/db/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'},
         'chapter_id': '128104',
         'gender': 'male',
         'num_samples': 93680,
         'speaker_id': '1272',
         'transcription': 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL',
         'example_id': '1272-128104-0000',
         'dataset': 'dev_clean',
         'audio_data': {'observation': array(shape=(93680,), dtype=float64)}}

        """
        datasets = {}
        kwargs = {
            # 'json_path': egs_dir / 'libri_css/data/jsons/librispeech.json',  # better documentation, because the true value is stored
            'json_path': '{egs_dir}/libri_css/data/jsons/librispeech.json',  # better probability, because only on path has to be env specific in the config
            'observation': "['observation']",
            'num_speakers': 1,
        }
        for name in [
            'train_960', 'dev_clean', 'eval_clean',
        ]:
            datasets[f'librispeech_{name}'] = {
                'factory': PBJsonDSMeta,
                'dataset_name': name,
                **kwargs,
            }
        return datasets


@dataclasses.dataclass
class Reader(pt.Configurable):
    """
    >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
    >>> for k in reader.datasets.keys(): print(k)  # only the used dataset will be in the config to avoid pollution
    SimLibriCSS-train-960_000_ch
    SimLibriCSS-dev-2_400_000_ch
    libri_css_ch
    SimLibriCSS-dev
    >>> len(reader.all_dataset_cfgs())  # all datasets can be vied by this method
    21
    >>> pb.utils.pretty.pprint(Reader.get_config())  # doctest: +ELLIPSIS
    {'factory': 'tssep_data.data.reader_v2.Reader',
     'datasets': {'SimLibriCSS-train-960_000_ch': {'factory': 'tssep_data.data.reader_v2.SegmentPBJsonDSMeta',
       'json_path': '.../jsons/sim_libri_css_ch_early.json',
       'dataset_name': 'SimLibriCSS-train',
       'num_speakers': 8,
       'sample_rate': 16000,
       'observation': "['observation'][:1]",
       'segment_num_samples': 960000,
       'minimum_segment_num_samples': None,
       'mixup': 0.5},
      'SimLibriCSS-dev-2_400_000_ch': {'factory': 'tssep_data.data.reader_v2.SegmentPBJsonDSMeta',
       'json_path': '.../jsons/sim_libri_css_ch_early.json',
       'dataset_name': 'SimLibriCSS-dev',
       'num_speakers': 8,
       'sample_rate': 16000,
       'observation': "['observation'][:1]",
       'segment_num_samples': 2400000,
       'minimum_segment_num_samples': 0,
       'mixup': None},
      'libri_css_ch': {'factory': 'tssep_data.data.reader_v2.PBJsonDSMeta',
       'json_path': '.../jsons/libriCSS_raw_compressed_ch.json',
       'dataset_name': ['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40'],
       'num_speakers': 8,
       'sample_rate': 16000,
       'observation': '["observation"][:1]'},
      'SimLibriCSS-dev': {'factory': 'tssep_data.data.reader_v2.PBJsonDSMeta',
       'json_path': '.../jsons/sim_libri_css_early.json',
       'dataset_name': 'SimLibriCSS-dev',
       'num_speakers': 8,
       'sample_rate': 16000,
       'observation': "['observation'][:1]"}},
     'train_dataset_name': 'SimLibriCSS-train-960_000_ch',
     'validate_dataset_name': 'SimLibriCSS-dev-2_400_000_ch',
     'domain_adaptation_src_dataset_name': 'SimLibriCSS-dev',
     'eval_dataset_name': ['libri_css_ch'],
     'data_hooks': {'factory': 'tssep_data.data.data_hooks.Sequential',
      'tasks': {'factory': 'dict',
       'auxInput': {'factory': 'tssep_data.data.data_hooks.SpeakerEmbeddings',
        'json': ['/home/cbj/python/tssep_data/egs'],
        'output_size': 100,
        'estimate': {'factory': 'dict',
         '0L': True,
         '0S': True,
         'OV10': True,
         'OV20': True,
         'OV30': True,
         'OV40': True}}}}}

    """
    datasets: dict

    train_dataset_name: 'str | list[str]' = 'SimLibriCSS-train-960_000_ch'
    validate_dataset_name: 'str | list[str]' = 'SimLibriCSS-dev-2_400_000_ch'
    domain_adaptation_src_dataset_name: 'str | list[str]' = 'SimLibriCSS-dev'
    # eval_dataset_name: 'str | list[str]' = ('libri_css_ch', 'SimLibriCSS-dev', 'SimLibriCSS-test')
    eval_dataset_name: 'str | list[str]' = ('libri_css',)

    data_hooks: 'data_hooks.Sequential' = dataclasses.field(
        default_factory=data_hooks.Sequential
    )

    # ToDo: Remove this, see /scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/150/slurm-train150-4500773.out
    sample_rate = 16000

    path_format_mapping: dict = dataclasses.field(
        default_factory=functools.partial(
            dict,
            egs_dir=egs_dir,
        )
    )

    # update_ex: dict = dataclasses.field(
    #     default_factory=functools.partial(
    #         dict,
    #         auxInput={
    #             'factory': 'tssep.data.data_hooks.SpeakerEmbeddings',
    #             'json': [
    #                 '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/simLibriCSS_oracle_ivectors.json',
    #                 '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/libriCSS_espnet_ivectors.json',
    #             ]
    #         },
    #         framewise_embeddings={
    #             'factory': 'tssep.data.data_hooks.FramewiseEmbeddings',
    #             'json': [
    #                 '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_3_simlibricss/framewise_embeddings.json',
    #                 '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/libriCSS_espnet_ivectors.json',
    #             ]
    #         },
    #     )
    # )

    db_cache: dict = dataclasses.field(
        init=False, repr=False, default_factory=dict)

    def _format_json_path(self, json_path):
        """
        >>> reader = Reader.new(updates=dict(datasets=Reader.all_dataset_cfgs()))
        >>> reader._format_json_path('{egs_dir}/libri_css/data/jsons/sim_libri_css_early.json')  # doctest: +ELLIPSIS
        '.../tssep_data/egs/libri_css/data/jsons/sim_libri_css_early.json'
        >>> reader._format_json_path(Path('{egs_dir}/libri_css/data/jsons/sim_libri_css_early.json'))  # doctest: +ELLIPSIS
        PosixPath('.../tssep_data/egs/libri_css/data/jsons/sim_libri_css_early.json')
        """
        if isinstance(json_path, (tuple, list)):
            return tuple(map(self._format_json_path, json_path))
        elif isinstance(json_path, str):
            return json_path.format(**self.path_format_mapping)
        elif isinstance(json_path, Path):
            return Path(os.fspath(json_path).format(**self.path_format_mapping))
        else:
            raise TypeError(type(json_path), json_path)

    def get_db(self, json_path) -> 'JsonDatabase':
        json_path = self._format_json_path(json_path)
        try:
            return self.db_cache[json_path]
        except KeyError:
            self.db_cache[json_path] = JsonDatabase(json_path)
        return self.db_cache[json_path]

    def __post_init__(self):
        self.datasets = pb.utils.mapping.Dispatcher(self.datasets)

        for k, v in self.datasets.items():
            assert isinstance(v, pt.Configurable), (type(v), k, v)

        if isinstance(self.data_hooks, dict):
            self.data_hooks = data_hooks.Sequential(self.data_hooks)

    @staticmethod
    def all_dataset_cfgs():
        return pb.utils.mapping.Dispatcher({
            **DatasetCfgs.sim_libri_css(),
            **DatasetCfgs.libri_css(),
            **DatasetCfgs.chime7(),
            **DatasetCfgs.librispeech(),
        })

    @classmethod
    def finalize_dogmatic_config(cls, config):
        def as_list(x):
            return [x] if isinstance(x, str) else x

        # if 'datasets' not in config or len(config['datasets'].keys()) <= 1:
        #     all_datasets = cls.all_datasets()
        #     config['datasets'] = {
        #         dataset_name: all_datasets[dataset_name]
        #         for name in ['train_dataset_name', 'validate_dataset_name', 'eval_dataset_name', 'domain_adaptation_src_dataset_name']
        #         for dataset_name in as_list(config[name])
        #     }

        all_datasets = cls.all_dataset_cfgs()
        if 'datasets' not in config:
            config['datasets'] = {}

        for name in [
                'train_dataset_name', 'validate_dataset_name',
                'eval_dataset_name', 'domain_adaptation_src_dataset_name',
        ]:
            if name in config:
                for dataset_name in as_list(config[name]):
                    if (
                            dataset_name not in config['datasets']  # missing
                            or not config['datasets'][dataset_name]  # or empty
                    ):
                        config['datasets'][dataset_name] = all_datasets[dataset_name]

        # /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed.json
        # /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json
        return config

    def prepare(
            self,
            ex,
            load_audio_fn,
            load_audio,
            load_keys,
    ):
        ex = self.data_hooks.pre_load_audio(ex, reader=self, load=load_audio, load_keys=load_keys)
        if load_audio:
            ex = load_audio_fn(ex, load_keys=load_keys)
        ex = self.data_hooks(ex, reader=self, load=load_audio)
        return ex

    class _AddVariantName:
        def __init__(self, dataset_variant):
            self.dataset_variant = dataset_variant

        def __repr__(self):
            return f"{self.__class__.__qualname__}({self.dataset_variant})"

        def __call__(self, ex):
            ex['dataset_variant'] = self.dataset_variant
            return ex

    def __call__(
            self,
            dataset_name="SimLibriCSS-test",
            load_audio=True,
            pre_load_apply=None,
            load_keys=['observation'],
    ):
        def get_ds(dataset_name):
            dsmeta: PBJsonDSMeta = self.datasets[dataset_name]
            ds = dsmeta.get_dataset(self).map(self._AddVariantName(dataset_name))
            ds = ds.apply(pre_load_apply)
            ds = ds.map(functools.partial(
                self.prepare, load_audio_fn=dsmeta.load_audio,
                load_audio=load_audio, load_keys=load_keys))
            return ds

        if isinstance(dataset_name, str):
            ds = get_ds(dataset_name)
        else:
            assert len(dataset_name) >= 1, dataset_name
            ds = lazy_dataset.concatenate([
                get_ds(name) for name in dataset_name])
        return ds


# @dataclasses.dataclass
# class Reader8spk(pt.Configurable):
#
#     @classmethod
#     def finalize_dogmatic_config(cls, config):
#         if 'datasets' not in config or len(config['datasets'].keys()) <= 1:
#             config['datasets'] = {
#                 **DatasetCfgs.sim_libri_css(),
#                 **DatasetCfgs.libri_css(),
#                 **DatasetCfgs.librispeech(),
#             }
#         return config
