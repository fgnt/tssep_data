"""

Licence MIT
Origin: Communications Department, Paderborn University, Germany

"""

# Fix doctest
import itertools
import os
import sys
import typing

if os.path.dirname(os.path.abspath(__file__)) == sys.path[0]:
    del sys.path[0]

import collections
import dataclasses
import functools
import re

import lazy_dataset
from lazy_dataset.database import JsonDatabase
import numpy as np
from pathlib import Path
import cached_property
import torch
import paderbox as pb
import padertorch as pt

# import css
# from css.utils import zip_strict
# from css.io.cache_loader import DiskCacheLoader

# from css.egs.extract.data.sim_libri_css import SimLibriCSS
# from css.egs.extract.data.util import _NO_VALUE
#
# from cbj.lib.access import ItemAccessor
# from css.egs.extract.data_ivector import _IvectorLoader
from paderbox.utils.iterable import zip
from tssep_data.data.kaldi import Loader as KaldiLoader

if typing.TYPE_CHECKING:
    # Cyclic import
    from tssep_data.data.reader_v2 import Reader

from tssep_data.data.reader_v2 import egs_dir

@dataclasses.dataclass
class ABC:
    """
    When are the functions called:

        Input pipeline:
             - pre_load_audio  # <--
             - load_audio
             - __call__  # <--
        Model:
             - feature transformation
             - pre_net  # <--
             - net
             - loss/review

    """
    def pre_load_audio(self, ex, reader: 'Reader', load=True, load_keys=['observation']):
        return ex

    def __call__(self, ex, reader: 'Reader', load=True):
        """
        Executed in the input pipeline in the prepare function.
        This is executed in parallel and should do the data loading.
        """
        return ex

    def pre_net(self, ex):
        """
        Executed just before the net. This is executed in the main thread.
        Usecase:
         - Fast operations, that are better executed on the GPU.
           e.g. repeat_interleave is fast and the transfer from more CPU memory
           to GPU memory is slower than transferring the input plus the
           repeat_interleave operation.
        """
        return ex


@dataclasses.dataclass
class _Template(ABC, pt.configurable.Configurable):
    json: str

    cache: dict = dataclasses.field(default_factory=dict, init=False)

    def get_ds(self, reader, dataset_name):
        # Takes care of the caching to avoid overhead
        # The reader caches the json data, that might e used between different
        # instances.
        # The instance itself caches the dataset.
        if isinstance(dataset_name, (tuple, list)):
            return lazy_dataset.concatenate(
                [self.get_ds(reader, name) for name in dataset_name])

        try:
            # Use try except, to make second call as fast as possible.
            ds = self.cache[dataset_name]
        except KeyError:
            db = reader.get_db(self.json)
            ds = self.cache[dataset_name] = db.get_dataset(dataset_name)
        return ds


@dataclasses.dataclass
class FramewiseEmbeddings(_Template):
    json: 'str | list[str] | tuple[str] ' = (
        '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_3_simlibricss/framewise_embeddings.json',
        '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_1_libricss/framewise_embeddings.json',
    )
    output_size: int = 256

    stft: pb.transform.STFT = dataclasses.field(
        default_factory=functools.partial(
            pb.transform.STFT,
            shift=160, window_length=400, size=512, fading=None,
            window='hamming'
        ))

    def __call__(self, ex, reader: 'Reader', load=True):
        """
        >>> np.random.seed(0)
        >>> from tssep_data.data.reader_v2 import Reader
        >>> reader = Reader.new()
        >>> reader.datasets['SimLibriCSS-train-960_000_ch']
        SegmentPBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch.json', dataset_name='SimLibriCSS-train', num_speakers=8, sample_rate=16000, observation="['observation'][:1]", segment_num_samples=960000, minimum_segment_num_samples=None, mixup=None)
        >>> ds = reader.__call__(dataset_name='SimLibriCSS-train-960_000_ch', load_keys=['observation'], load_audio=True)
        >>> ex = ds[0]
        >>> ex['start'], ex['end']
        (1663110, 2623110)
        >>> _ = FramewiseEmbeddings().__call__(ex, reader)
        >>> ex['framewise_embeddings'].shape
        (1500, 256)
        >>> ex['audio_data']['observation'].shape
        (1, 960000)
        >>> FramewiseEmbeddings().stft(ex['audio_data']['observation']).shape
        (1, 5999, 257)
        >>> ex['framewise_embeddings'].shape
        (1500, 256)
        >>> ex['framewise_embeddings_stride']
        4
        >>> ex['start'], ex['end'] = 0, 960000
        >>> _ = FramewiseEmbeddings().__call__(ex, reader)
        >>> ex['framewise_embeddings'].shape
        >>> ex['num_samples']['observation']
        4213932
        >>> pb.utils.pretty.pprint(ex['audio_data']['observation'])
        array(shape=(1, 960000), dtype=float64)
        >>> ex['start'], ex['end'] = ex['num_samples']['observation'] - 960000, ex['num_samples']['observation']
        >>> _ = FramewiseEmbeddings().__call__(ex, reader)
        >>> ex['framewise_embeddings'].shape
        ex['framewise_embeddings'] (6574, 256)
        >>> pb.utils.pretty.pprint(ex['audio_data']['observation'])
        array(shape=(1, 960000), dtype=float64)
        >>> ex['framewise_embeddings'].shape
        (1500, 256)
        >>> FramewiseEmbeddings().stft(pb.io.load(ex['audio_path']['observation'])).shape
        (1, 26336, 257)

        """
        # ToDO: Consider start_orig and end_orig
        
        ds = self.get_ds(reader, ex['dataset'])

        ex_aux = ds[ex['example_id']]

        if 'start' not in ex:
            ex['start'] = 0

        if 'end' not in ex:
            try:
                ex['end'] = ex['start'] + ex['num_samples']['observation']
            except KeyError:
                ex['end'] = ex['start'] + ex['num_samples']

        frames = self.stft.samples_to_frames(ex['end'] - ex['start'])

        start = ex['start']
        end = ex['end']

        orig_start, orig_end = ex['uem']
        if orig_end is not None:
            assert end <= orig_end, (ex['start'], ex['end'], ex['uem'])
        if orig_start != 0:
            start = start - orig_start
            end = end - orig_start
            assert start >= 0, (ex['start'], ex['end'], ex['uem'])

        start_frame = self.stft.sample_index_to_frame_index(start)
        end_frame = self.stft.sample_index_to_frame_index(end)

        if end_frame - start_frame - 1 == frames:
            pass
        elif end_frame - start_frame - 1 == frames + 1:
            # sample_index_to_frame_index is not the best function to calculate this.
            end_frame = start_frame + frames
        elif start_frame == 0:  # and end_frame - start_frame - 1 in [frames - 1, frames - 2]:
            end_frame = start_frame + frames
            # raise lazy_dataset.FilterException('NotImplemented', ex['start'], ex['end'], start_frame, end_frame, frames)
        else:
            raise AssertionError(end_frame - start_frame - 1, '!=', frames, start_frame, end_frame)


        # 5 frames in the beginning and end are not estimated
        # start_frame = max(0, start_frame-5)
        # end_frame = max(0, end_frame-5)

        # Stride of the model is 4
        start_frame = start_frame // 4

        # 5 frames in the beginning and end are not estimated:
        # These are the frames after striding
        start_frame = start_frame - 5

        #  Use one more frame than necessary
        end_frame = start_frame + -(-frames // 4)


        # ToDo: Handle mixup

        if load:
            ex['framewise_embeddings'] = pb.io.load(ex_aux['path'], unsafe=True)
            # print("ex['framewise_embeddings']", ex['framewise_embeddings'].shape)

            ex['framewise_embeddings'] = self._extract_piece(ex['framewise_embeddings'], start_frame, end_frame)

            if self._transform is not None:
                ex['framewise_embeddings'] = self._transform(
                    ex['framewise_embeddings'])

            # if start_frame >= 0:
            #     if end_frame <= ex['framewise_embeddings'].shape[0]:
            #         ex['framewise_embeddings'] = ex['framewise_embeddings'][start_frame:end_frame].copy()
            #     else:
            #         # ToDo
            #         ex['framewise_embeddings'] = pb.array.pad_axis(
            #             ex['framewise_embeddings'],
            #             pad_width=(0, end_frame - ex['framewise_embeddings'].shape[0]),
            #             axis=0, mode='edge')
            #         # raise lazy_dataset.FilterException('NotImplemented', start_frame, end_frame, ex['framewise_embeddings'].shape[0])
            # else:
            #     if end_frame <= ex['framewise_embeddings'].shape[0]:
            #
            #         raise lazy_dataset.FilterException('NotImplemented', start_frame, end_frame, ex['framewise_embeddings'].shape[0])
            #     else:
            #         # ToDo
            #         raise lazy_dataset.FilterException('NotImplemented', start_frame, end_frame, ex['framewise_embeddings'].shape[0])

        else:
            ex['framewise_embeddings'] = {
                'path': ex_aux['path'],
                'slice': slice(start_frame, end_frame),
            }
        ex['framewise_embeddings_stride'] = 4

        # breakpoint()

        return ex

    def _test_libri_css(self):
        """

        >>> np.random.seed(0)
        >>> from tssep_data.data.reader_v2 import Reader
        >>> from pprint import pprint

        >>> cfg = pb.io.load('/scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/150/eval/21000/1/config.yaml')
        >>> cfg = cfg['eg']['trainer']['model']['reader']
        >>> # cfg = Reader.get_config()
        >>> # del cfg['data_hooks']['tasks']['auxInput']  # ['factory'] = ABC
        >>> reader = Reader.from_config(cfg)
        >>> reader.datasets['libri_css_ch']
        PBJsonDSMeta(json_path='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/libriCSS_raw_compressed_ch.json', dataset_name=['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40'], num_speakers=8, sample_rate=16000, observation="['observation'][:1]")
        >>> pprint(reader.data_hooks)
        Sequential(tasks={'auxInput': SpeakerEmbeddings(json=['/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/3/sim_libri_css_ch_8spk_oracle_dvectors.json',
                                                              '/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/5/libri_css_ch_8spk_ch_oracle_dvectors.json'],
                                                        cache={},
                                                        output_size=100,
                                                        _ivector_loader=css.egs.extract.data_ivector._IvectorLoader()),
                          'framewise_embeddings': FramewiseEmbeddings(json=['/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_3_simlibricss/framewise_embeddings.json',
                                                                            '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_1_libricss/framewise_embeddings.json'],
                                                                      cache={},
                                                                      output_size=256,
                                                                      stft=STFT(shift=160,
                                                                                size=512,
                                                                                window_length=400,
                                                                                window='hamming',
                                                                                symmetric_window=False,
                                                                                pad=True,
                                                                                fading=None))})
        >>> ds = reader.__call__(dataset_name='libri_css_ch', load_keys=['observation'], load_audio=True)
        >>> ex = ds['overlap_ratio_30.0_sil0.1_1.0_session2_actual29.6_ch6']
        >>> ex['start'], ex['end']
        (47949, 9899818)
        >>> hook = FramewiseEmbeddings()
        >>> _ = hook.__call__(ex, reader)
        >>> ex['framewise_embeddings'].shape
        (15394, 256)

        >>> # ex['Observation'] = torch.from_numpy(hook.stft(ex['audio_data']['observation']))
        >>> ex['Input'] = torch.from_numpy(hook.stft(ex['audio_data']['observation']))
        >>> ex['Input'].shape
        torch.Size([1, 61573, 257])
        >>> ex['framewise_embeddings'] = torch.from_numpy(ex['framewise_embeddings'])
        >>> ex = hook.pre_net(ex)
        >>> ex['Input'].shape
        torch.Size([1, 61573, 513])
        >>> ex['auxInput'].shape
        (8, 256)

        >>> ex['frame_slice'] = {'slice': slice(0, 4000), 'orig_num_frames': ex['Input'].shape[-2]}
        >>> ex['Input'] = ex['Input'][..., ex['frame_slice']['slice'], :]
        >>> # del ex['Input']
        >>> ex = hook.pre_net(ex)
        >>> ex['Input'].shape
        torch.Size([1, 4000, 769])
        """

    def _extract_piece(self, framewise_embeddings, start_frame, end_frame):
        """
        >>> femb = FramewiseEmbeddings()
        >>> framewise_embeddings = np.arange(10) + 1
        >>> femb._extract_piece(framewise_embeddings, 3, 7)
        array([4, 5, 6, 7])
        >>> femb._extract_piece(framewise_embeddings, 0, 10)
        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
        >>> femb._extract_piece(framewise_embeddings, -5, 5)
        array([1, 1, 1, 1, 1, 1, 2, 3, 4, 5])
        >>> femb._extract_piece(framewise_embeddings, 5, 15)
        array([ 6,  7,  8,  9, 10, 10, 10, 10, 10, 10])
        >>> femb._extract_piece(framewise_embeddings, -5, 15)
        array([ 1,  1,  1,  1,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 10, 10,
               10, 10, 10])

        """
        pad_width = (max(0, -start_frame),
                     max(0, end_frame - framewise_embeddings.shape[0]))
        s = slice(max(0, start_frame),
                  min(end_frame, framewise_embeddings.shape[0]))
        assert pad_width[0] <= 5 and pad_width[1] <= 5, (pad_width, s, start_frame, end_frame, framewise_embeddings.shape)
        framewise_embeddings = framewise_embeddings[s]
        if pad_width != (0, 0):
            return pb.array.pad_axis(
                framewise_embeddings,
                pad_width=pad_width,
                axis=0, mode='edge')
        else:
            return framewise_embeddings.copy()

    def pre_net(self, ex):
        """

        >>> from tssep_data.data.reader_v2 import Reader
        >>> reader = Reader.new()
        >>> ds = reader.__call__(dataset_name='SimLibriCSS-train-960_000_ch', load_keys=['observation'], load_audio=True)
        >>> ex = ds[0]
        >>> # _ = FramewiseEmbeddings().update_ex(ex, reader)
        >>> hook = FramewiseEmbeddings()
        >>> ex = hook.__call__(ex, reader)
        >>> ex['Input'] = torch.from_numpy(hook.stft(ex['audio_data']['observation']))
        >>> ex['Input'].shape
        torch.Size([1, 11999, 257])
        >>> ex['framewise_embeddings'] = torch.from_numpy(ex['framewise_embeddings'])
        >>> ex = hook.pre_net(ex)
        >>> ex['Input'].shape
        torch.Size([1, 11999, 513])
        """
        Input = ex['Input']
        framewise_embeddings = ex['framewise_embeddings']
        framewise_embeddings_stride = ex['framewise_embeddings_stride']
        if not isinstance(framewise_embeddings_stride, int):
            assert len(set(framewise_embeddings_stride)) == 1, framewise_embeddings_stride
            framewise_embeddings_stride = framewise_embeddings_stride[0]

        frames = Input.shape[-2]

        if 'frame_slice' in ex:
            frame_slice = ex['frame_slice']['slice']
            frames = ex['frame_slice']['orig_num_frames']
        else:
            frame_slice = slice(None)

        assert (framewise_embeddings.shape[-2] - 1) * framewise_embeddings_stride < frames <= framewise_embeddings.shape[-2] * framewise_embeddings_stride, (framewise_embeddings.shape, Input.shape, framewise_embeddings_stride, ex['example_id'])

        # try:
        #     raise Exception(Input.shape, ex['example_id'])
        # except Exception:
        #     import traceback
        #     print(traceback.format_exc())
        #     # # or
        #     # print(sys.exc_info()[2])
        # traceback.print_stack()

        # framewise_embeddings = torch.repeat_interleave(framewise_embeddings,framewise_embeddings_stride, dim=-2)

        framewise_embeddings = torch.repeat_interleave(
            framewise_embeddings,
            framewise_embeddings_stride, dim=-2
        )
        try:
            assert frames <= framewise_embeddings.shape[-2] <= frames + framewise_embeddings_stride, (framewise_embeddings.shape, Input.shape, framewise_embeddings_stride)
        except AttributeError as e:
            raise Exception(pb.utils.pretty.pretty([framewise_embeddings, Input, framewise_embeddings_stride])) from e

        framewise_embeddings = framewise_embeddings[..., :frames, :]
        framewise_embeddings = framewise_embeddings[..., frame_slice, :]


        # Add a singleton dimension for the channel
        # ToDo: How to support multiple channels?
        #       Ideas:
        #        - One embedding per channel
        #        - Same embedding for all channels

        if len(Input.shape) - 1 == len(framewise_embeddings.shape) and Input.shape[-3] == 1:
            framewise_embeddings = framewise_embeddings[..., None, :, :]
        if len(framewise_embeddings.shape) == 3 and framewise_embeddings.shape[-3] == 1 and len(Input.shape) == 2:
            framewise_embeddings = torch.squeeze(framewise_embeddings, -3)

        # assert Input.shape[-1] == 337, Input.shape

        try:
            ex['Input'] = torch.cat((Input, framewise_embeddings), -1)
        except RuntimeError as e:
            raise Exception(Input.shape, framewise_embeddings.shape)
        return ex

    _transform = None

    def register_transform(self, transform):
        """
        This function can be used to transform the Emb-Vectors.

        Usecase:
            Mean and variance normalization for domain adaptation.

        >>> from paderbox.utils.pretty import pprint
        >>> from tssep_data.data.reader_v2 import Reader
        >>> # ex = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json').get_dataset('0L')[0]
        >>> reader = Reader.new()
        >>> ex = reader('libri_css_ch')[0]
        >>> pprint(ex['framewise_embeddings'])
        array(shape=(15006, 256), dtype=float32)
        >>> reader.data_hooks.tasks['framewise_embeddings'].register_transform(lambda iv: iv[..., :50])
        >>> ex = reader('libri_css_ch')[0]
        >>> pprint(ex['framewise_embeddings'])
        array(shape=(15006, 50), dtype=float32)
        """
        assert self._transform is None, (self._transform, transform)
        self._transform = transform

# def concaternate(tensors, dim):
#     """
#     >>> import torch
#     >>> a = torch.ones((2, 3))
#     >>> b = torch.ones((1, 3))
#     >>> c = torch.ones((3,))
#     >>> concaternate([a, b, c], dim=-1).shape
#     torch.Size([6, 3])
#     >>> concaternate([a, b, c], dim=-1).shape
#     torch.Size([2, 9])
#     >>> concaternate([a, b, c], dim=-1).shape
#     torch.Size([2, 3, 9])
#     """
#     shapes = [
#         tensor.shape
#         for tensor in tensors
#     ]
#     ndims = list(map(len, shapes))
#     if min(ndims) =! max(ndims):
#
#
#     return torch.cat(tensors, dim=dim)


@dataclasses.dataclass
class SpeakerEmbeddings(_Template):
    # json: 'str | list[str]' = '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/simLibriCSS_oracle_ivectors.json'
    json: 'str | list[str] | tuple[str]' = (
        # '/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/3/sim_libri_css_ch_8spk_oracle_dvectors.json',
    )
    output_size: int = 100
    estimate: 'dict[str, bool]' = dataclasses.field(default_factory=functools.partial(
        dict,
        **{
            '0L': True,
            '0S': True,
            'OV10': True,
            'OV20': True,
            'OV30': True,
            'OV40': True,
        }
    ))

    _ivector_loader: KaldiLoader = dataclasses.field(
        default_factory=KaldiLoader, init=False)

    def __call__(self, ex, reader: 'Reader', load=True):
        """
        >>> from tssep_data.data.reader_v2 import Reader
        >>> reader = Reader.new()
        >>> # ds = reader.__call__(dataset_name='SimLibriCSS-train-960_000_ch', load_keys=['observation'], load_audio=True)
        >>> ds = reader.__call__(dataset_name='SimLibriCSS-train-960_000_ch', load_keys=['observation'], load_audio=True)
        >>> ex = ds[0]
        >>> _ = SpeakerEmbeddings().__call__(ex, reader)
        >>> pb.utils.pretty.pprint(ex['auxInput'])
        {'1752': array(shape=(100,), dtype=float32),
         '289': array(shape=(100,), dtype=float32),
         '3109': array(shape=(100,), dtype=float32),
         '4766': array(shape=(100,), dtype=float32),
         '6121': array(shape=(100,), dtype=float32),
         '6549': array(shape=(100,), dtype=float32),
         '7912': array(shape=(100,), dtype=float32),
         '7967': array(shape=(100,), dtype=float32)}
        """
        ds = self.get_ds(reader, ex['dataset'])

        example_id = ex['example_id']
        embedding_id = ex.get('embedding_id', example_id)

        if example_id in ex:
            aux_ex = ds[example_id]
        else:
            aux_ex = ds[embedding_id]

        # ex['auxInput'] = {
        #     speaker_id: self._ivector_loader(speaker_embedding_path) if load else speaker_embedding_path
        #     for speaker_id, speaker_embedding_path in zip_strict(
        #             ex['speaker_id'], aux_ex['speaker_embedding_path'])
        # }

        aux_speaker_ids = list(dict.fromkeys(aux_ex['speaker_id']))

        if ex['dataset'] in self.estimate and self.estimate[ex['dataset']]:
            speaker_ids = aux_speaker_ids
        else:
            speaker_ids = list(dict.fromkeys(ex['speaker_id']))

            if set(speaker_ids) != set(aux_speaker_ids):
                # I-Vectors json skips missing speakers
                raise lazy_dataset.FilterException(speaker_ids, aux_speaker_ids)

            if not all(aux_ex['speaker_embedding_path']):
                # D-Vectors json have an empty string for missing speakers
                raise lazy_dataset.FilterException(aux_ex['speaker_embedding_path'])

        zip_args = aux_speaker_ids, aux_ex['speaker_embedding_path']
        if not len(set(map(len, zip_args))) == 1:
            raise Exception(
                'The number of speaker ids and speaker embedding paths must be the same.\n'
                f'embedding_id: {embedding_id}\n'
                f'json: {self.json}\n'
                f'{len(speaker_ids)} speaker_ids: {speaker_ids}\n'
                f'{len(aux_ex["speaker_embedding_path"])} speaker_embedding_paths: {aux_ex["speaker_embedding_path"]}\n'

            )

        speaker_embedding_paths = dict(zip(*zip_args, strict=True))

        ex['auxInput'] = [
            self._ivector_loader(speaker_embedding_paths[speaker_id]) if load else speaker_embedding_paths[speaker_id]
            for speaker_id in speaker_ids
        ]
        if load:
            ex['auxInput'] = np.array(ex['auxInput'])

            if self._transform is not None:
                ex['auxInput'] = self._transform(ex['auxInput'])

        ex['auxInput_keys'] = ex['speaker_id']

        return ex

    def mean_std(self, datasets: 'list[str]', reader: 'Reader'):
        """
        >>> import paderbox as pb
        >>> from tssep_data.data.reader_v2 import Reader
        >>> reader = Reader.new()
        >>> # ds = reader.__call__(dataset_name='SimLibriCSS-train-960_000_ch', load_keys=['observation'], load_audio=True)
        >>> auxInput = reader.data_hooks.tasks['auxInput']
        >>> # auxInput.json
        >>> pb.utils.pretty.pprint(auxInput.mean_std('OV40', reader))
        (array(shape=(256,), dtype=float32), array(shape=(256,), dtype=float32))
        """
        ds = self.get_ds(reader, datasets)

        paths = []
        for ex in ds:
            for path in ex['speaker_embedding_path']:
                if not path:
                    continue
                paths.append(path)

        paths = list(dict.fromkeys(paths))
        embs = [self._ivector_loader(p) for p in paths]

        embs = np.array(embs)

        assert self._transform is None, self._transform

        mean = np.mean(embs, axis=0, keepdims=False)
        std = np.std(embs, axis=0, keepdims=False)
        return mean, std

    _transform = None

    def domain_adaptation(self, reader, kind, consider_mpi=True):
        import dlp_mpi

        if dlp_mpi.IS_MASTER:
            print('Domain Adaptation', kind,
                  f'{reader.domain_adaptation_src_dataset_name} -> {reader.eval_dataset_name}')
            # return dlp_mpi.bcast(
            #     self.domain_adaptation(reader, kind, consider_mpi=False)
            #     if dlp_mpi.IS_MASTER else
            #     None
            # )
        
            datasets = sorted({
                ex['dataset']
                for ex in
                reader(reader.eval_dataset_name, load_audio=False)
            })
            # reader.data_hooks.tasks['auxInput']
            aux_eval_mean, aux_eval_std = self.mean_std(datasets, reader)

            datasets = sorted({
                ex['dataset']
                for ex in reader(reader.domain_adaptation_src_dataset_name, load_audio=False)
            })
            aux_val_mean, aux_val_std = self.mean_std(datasets, reader)
        else:
            aux_val_mean, aux_val_std = None, None
            aux_eval_mean, aux_eval_std = None, None

        aux_val_mean, aux_val_std, aux_eval_mean, aux_eval_std = \
            dlp_mpi.bcast([aux_val_mean, aux_val_std, aux_eval_mean, aux_eval_std])

        if kind == 'mean_std':
            aux_feature_transform = lambda x: (x - aux_eval_mean) * (
                        aux_val_std / aux_eval_std) + aux_val_mean
        elif kind == 'mean_scale':
            aux_val_std = np.mean(aux_val_std)
            aux_eval_std = np.mean(aux_eval_std)
            aux_feature_transform = lambda x: (x - aux_eval_mean) * (
                        aux_val_std / aux_eval_std) + aux_val_mean
        elif kind == 'mean':
            aux_feature_transform = lambda x: (x - aux_eval_mean) + aux_val_mean
        else:
            raise ValueError(kind)

        self.register_transform(aux_feature_transform)
        # reader.data_hooks.tasks['auxInput'].register_transform(aux_feature_transform)
        self.register_framewise_transform(aux_feature_transform, reader)

    def register_framewise_transform(self, transform, reader):
        try:
            reader.data_hooks.tasks['framewise_embeddings'].register_transform(transform)
        except AttributeError:
            t = type(reader.data_hooks.tasks['framewise_embeddings'])
            assert t == ABC, (t, reader.data_hooks.tasks['framewise_embeddings'])
        except KeyError:
            pass

    def register_transform(self, transform):
        """
        This function can be used to transform the Emb-Vectors.

        Usecase:
            Mean and variance normalization for domain adaptation.

        >>> from paderbox.utils.pretty import pprint
        >>> from tssep_data.data.reader_v2 import Reader
        >>> # ex = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json').get_dataset('0L')[0]
        >>> reader = Reader.new()
        >>> ex = reader('libri_css_ch')[0]
        >>> pprint(ex['auxInput'])
        array(shape=(8, 256), dtype=float32)
        >>> reader.data_hooks.tasks['auxInput'].register_transform(lambda iv: iv[..., :50])
        >>> ex = reader('libri_css_ch')[0]
        >>> pprint(ex['auxInput'])
        array(shape=(8, 50), dtype=float32)
        """
        assert self._transform is None, (self._transform, transform)
        self._transform = transform


@dataclasses.dataclass
class SecondSpeakerEmbeddings(SpeakerEmbeddings):
    """
    Stack two embeddings together.
    Assume framewise belongs to the first.
    """

    def register_framewise_transform(self, transform, reader):
        """
        Disable framewise_transform, assume first embedding belongs to
        framewise.
        """
    def domain_adaptation(self, reader, kind, consider_mpi=True):
        if self.json:
            super().domain_adaptation(reader, kind, consider_mpi)
        else:
            pass

    def __call__(self, ex, reader: 'Reader', load=True):
        if not self.json:
            return ex
        otherAuxInput = ex['auxInput']
        otherAuxInput_keys = ex['auxInput_keys']
        ex = super().__call__(ex, reader, load)
        assert otherAuxInput is not ex['auxInput']
        assert otherAuxInput_keys is ex['auxInput_keys']
        # if otherAuxInput_keys != ex['auxInput_keys']:
        #     raise lazy_dataset.FilterException(otherAuxInput_keys, ex['auxInput_keys'])
        ex['auxInput'] = np.concatenate([otherAuxInput, ex['auxInput']], axis=-1)
        return ex


@dataclasses.dataclass
class VAD(ABC):
    files: 'dict[str, str]' = dataclasses.field(default_factory=functools.partial(
        dict,
        **{
            'SimLibriCSS-train': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css/target_vad/1/SimLibriCSS-train.pkl',
            'SimLibriCSS-dev': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css/target_vad/1/SimLibriCSS-dev.pkl',
            'SimLibriCSS-test': '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css/target_vad/1/SimLibriCSS-test.pkl',
        },
    ))

    cache: dict = dataclasses.field(default_factory=dict, init=False)

    def get_ds(self, reader, dataset_name):
        try:
            ds = self.cache[dataset_name]
        except KeyError:
            ds = self.cache[dataset_name] = lazy_dataset.new(
                pb.io.load(self.files[dataset_name], unsafe=True),
                # immutable_warranty=None,  # Add again, when the new version is released.
            )
        return ds

    def pre_load_audio(self, ex, reader: 'Reader', load=True, load_keys=['observation']):

        # SegmentPBJsonDSMeta takes care of cutting the vad and doing mixup.

        if load and 'vad' in load_keys:
        # start, end = ex['start'], ex['end']
            ex.setdefault('audio_data', {})['vad'] = self.get_ds(
                reader, ex['dataset']
            )[ex['example_id']]
        # if ex['mixup']:
        #     tmp = ex['audio_data']['vad'].shape[-1] // 2
        #     ex['audio_data']['vad'] = (
        #             ex['audio_data']['vad'].slice[..., :tmp]
        #             + ex['audio_data']['vad'].slice[..., tmp:2 * tmp]
        #     )
        return ex


@dataclasses.dataclass
class Sequential(ABC):

    tasks: dict = dataclasses.field(
        default_factory=functools.partial(
            dict,
            auxInput={
                'factory': SpeakerEmbeddings,
                'json': [
                    egs_dir / '',
                    # '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/simLibriCSS_oracle_ivectors.json',
                    # '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/libriCSS_espnet_ivectors.json',
                    # '/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/3/sim_libri_css_ch_8spk_oracle_dvectors.json',
                    # '/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/4/libri_css_ch_8spk_ch0_oracle_dvectors.json',
                    # '/scratch/hpc-prf-nt2/cbj/deploy/cbj/egs/2023/dvector/5/libri_css_ch_8spk_ch_oracle_dvectors.json',
                ]
            },
            # auxInput2={
            #     'factory': 'tssep.data.data_hooks.SecondSpeakerEmbeddings',
            # },
            # framewise_embeddings={
            #     'factory': 'tssep.data.data_hooks.FramewiseEmbeddings',
            #     'json': [
            #         '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_3_simlibricss/framewise_embeddings.json',
            #         '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/dvector/framewise/tcl_1_libricss/framewise_embeddings.json',
            #         # '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/libriCSS_espnet_ivectors.json',
            #     ]
            # },
        )
    )

    def pre_load_audio(self, ex, reader: 'Reader', load=True, load_keys=['observation']):
        for k, v in self.tasks.items():
            ex = v.pre_load_audio(ex, reader=reader, load=load, load_keys=load_keys)
        return ex

    def __call__(self, ex, reader: 'Reader', load=True):
        for k, v in self.tasks.items():
            ex = v.__call__(ex, reader=reader, load=load)
        return ex

    def pre_net(self, ex):
        for k, v in self.tasks.items():
            ex = v.pre_net(ex)
        return ex
