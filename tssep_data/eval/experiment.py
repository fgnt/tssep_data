from __future__ import annotations  # Ignore cyclic imports for annotations, i.e. don't execute annotations

import dataclasses
import itertools
import subprocess
import functools
import threading
import traceback
from pathlib import Path
import operator

import torch.nn.functional
import numpy as np
from einops import rearrange, repeat
import lazy_dataset
import tqdm

import paderbox as pb
import padertorch as pt

import tssep_data.io.compess
import tssep.train.enhancer
from tssep_data.eval.probability_to_segments import DiscretizeVAD
# import css.io.compess
# import css.egs.extract
import tssep_data.data.data_hooks
import tssep.train.model

# from css.egs.extract.find_optimal_der import DiscretizeVAD


def _field(default_factory, /, **kwargs):
    if kwargs:
        return dataclasses.field(
            default_factory=functools.partial(default_factory, **kwargs))
    else:
        return dataclasses.field(
            default_factory=default_factory)


@dataclasses.dataclass
class Segmenter:
    length: int
    shift: int  # Bug: Implementation is shift not overlap. But nothing is wrongly computed.

    # Segmenter.Segment(s=10, e=20, s_ins=12, e_ins=20, s_cut=2, e_cut=10, start=0, end=20)
    @dataclasses.dataclass
    class Segment:
        s: int  # Used to slice the segment from the full data.
        e: int
        s_ins: int  # Used to insert the processed result.
        e_ins: int
        s_cut: int  # Used to cut the processed result.
        e_cut: int

        start: int
        end: int

        # Names:
        #  x.shape == (end - start)  # Is start != 0 supported?
        #  s = x[s:e]
        #  y[s_ins:e_ins] = foo(s)[s_cut:e_cut]

        def __post_init__(self):
            self.slice = slice(self.s, self.e)
            self.slice_ins = slice(self.s_ins, self.e_ins)
            self.slice_cut = slice(self.s_cut, self.e_cut)

        def _slice_axis(self, s, axis):
            if axis == -1:
                return (..., s)
            elif axis == 0:
                return (s,)
            elif axis == -2:
                return (..., s, slice(None))
            else:
                raise NotImplementedError(axis, s)

        def _assign(self, destination, value, axis, pad_fn):
            value = value[self._slice_axis(self.slice_cut, axis)]
            if destination is None:
                destination = pad_fn(value, (self.s_ins, self.end-self.e_ins), axis=axis)
            else:
                destination[self._slice_axis(self.slice_ins, axis)] = value
            return destination

        def np_assign(self, destination, value, axis):
            def pad_fn(a, pad, axis):
                return pb.array.pad_axis(a, axis=axis, pad_width=pad)

            return self._assign(destination, value, axis, pad_fn)

        def torch_assign(self, destination, value, axis):
            """
            Assigns the valid values of value to the correct position in destination.
            If destination does not exist (i.e. is None), create that array/tensor with zero padding.

            """
            def pad_fn(a, pad, axis):
                if axis == -1:
                    pass
                elif axis == -2:
                    pad = (0, 0) + pad
                else:
                    raise NotImplementedError(axis, pad, a)

                return torch.nn.functional.pad(a, pad)

            return self._assign(destination, value, axis, pad_fn)

    def boundaries(
            self, start, end
    ):
        """
        >>> from paderbox.utils.pretty import pprint

        >>> segmenter = Segmenter(length=10, shift=2)
        >>> pprint(segmenter.boundaries(0, 20))
        (Segmenter.Segment(s=0, e=10, s_ins=0, e_ins=9, s_cut=0, e_cut=9, start=0, end=20),
         Segmenter.Segment(s=2, e=12, s_ins=9, e_ins=11, s_cut=7, e_cut=9, start=0, end=20),
         Segmenter.Segment(s=4, e=14, s_ins=11, e_ins=13, s_cut=7, e_cut=9, start=0, end=20),
         Segmenter.Segment(s=6, e=16, s_ins=13, e_ins=15, s_cut=7, e_cut=9, start=0, end=20),
         Segmenter.Segment(s=8, e=18, s_ins=15, e_ins=17, s_cut=7, e_cut=9, start=0, end=20),
         Segmenter.Segment(s=10, e=20, s_ins=17, e_ins=20, s_cut=7, e_cut=10, start=0, end=20))

        >>> segmenter = Segmenter(10, 5)
        >>> pprint(segmenter.boundaries(0, 20))
        (Segmenter.Segment(s=0, e=10, s_ins=0, e_ins=7, s_cut=0, e_cut=7, start=0, end=20),
         Segmenter.Segment(s=5, e=15, s_ins=7, e_ins=12, s_cut=2, e_cut=7, start=0, end=20),
         Segmenter.Segment(s=10, e=20, s_ins=12, e_ins=20, s_cut=2, e_cut=10, start=0, end=20))
        >>> pprint(segmenter.boundaries(0, 19))
        (Segmenter.Segment(s=0, e=10, s_ins=0, e_ins=7, s_cut=0, e_cut=7, start=0, end=19),
         Segmenter.Segment(s=5, e=15, s_ins=7, e_ins=12, s_cut=2, e_cut=7, start=0, end=19),
         Segmenter.Segment(s=9, e=19, s_ins=12, e_ins=19, s_cut=3, e_cut=10, start=0, end=19))
        >>> pprint(segmenter.boundaries(0, 21))
        (Segmenter.Segment(s=0, e=10, s_ins=0, e_ins=7, s_cut=0, e_cut=7, start=0, end=21),
         Segmenter.Segment(s=5, e=15, s_ins=7, e_ins=12, s_cut=2, e_cut=7, start=0, end=21),
         Segmenter.Segment(s=10, e=20, s_ins=12, e_ins=17, s_cut=2, e_cut=7, start=0, end=21),
         Segmenter.Segment(s=11, e=21, s_ins=17, e_ins=21, s_cut=6, e_cut=10, start=0, end=21))
        >>> pprint(segmenter.boundaries(0, 10))
        (Segmenter.Segment(s=0, e=10, s_ins=0, e_ins=10, s_cut=0, e_cut=10, start=0, end=10),)
        >>> pprint(segmenter.boundaries(0, 9))
        (Segmenter.Segment(s=0, e=9, s_ins=0, e_ins=9, s_cut=0, e_cut=9, start=0, end=9),)
        >>> pprint(segmenter.boundaries(0, 8))
        (Segmenter.Segment(s=0, e=8, s_ins=0, e_ins=8, s_cut=0, e_cut=8, start=0, end=8),)

        >>> a = np.arange(21)
        >>> b = None
        >>> c = None
        >>> for seg in segmenter.boundaries(0, len(a)):
        ...     a_seg = a[seg.slice]
        ...     b_seg = 10 + a_seg  # Here you would do something
        ...     c_seg = 10 + torch.tensor(a_seg)  # Here you would do something
        ...     b = seg.np_assign(b, b_seg, axis=-1)
        ...     c = seg.torch_assign(c, c_seg, axis=-1)
        ...     print(b)
        ...     print(c.numpy(), 'torch')
        [10 11 12 13 14 15 16  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        [10 11 12 13 14 15 16  0  0  0  0  0  0  0  0  0  0  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21  0  0  0  0  0  0  0  0  0]
        [10 11 12 13 14 15 16 17 18 19 20 21  0  0  0  0  0  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  0  0  0  0]
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30] torch

        >>> a = np.arange(21)
        >>> b = None
        >>> c = None
        >>> for seg in segmenter.boundaries(0, len(a)):
        ...     a_seg = a[seg.slice]
        ...     b_seg = 10 + a_seg[:, None]  # Here you would do something
        ...     c_seg = 10 + torch.tensor(a_seg)[:, None]  # Here you would do something
        ...     b = seg.np_assign(b, b_seg, axis=-2)
        ...     c = seg.torch_assign(c, c_seg, axis=-2)
        ...     print(np.squeeze(b, axis=-1))
        ...     print(np.squeeze(c.numpy(), axis=-1), 'torch')
        [10 11 12 13 14 15 16  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
        [10 11 12 13 14 15 16  0  0  0  0  0  0  0  0  0  0  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21  0  0  0  0  0  0  0  0  0]
        [10 11 12 13 14 15 16 17 18 19 20 21  0  0  0  0  0  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  0  0  0  0]
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26  0  0  0  0] torch
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30] torch

        >>> a = np.arange(21)
        >>> b = None
        >>> c = None
        >>> segments = list(segmenter.boundaries(0, len(a)))
        >>> np.random.seed(0)
        >>> np.random.shuffle(segments)
        >>> for seg in segments:
        ...     a_seg = a[seg.slice]
        ...     b_seg = 10 + a_seg  # Here you would do something
        ...     c_seg = 10 + torch.tensor(a_seg)  # Here you would do something
        ...     b = seg.np_assign(b, b_seg, axis=-1)
        ...     c = seg.torch_assign(c, c_seg, axis=-1)
        ...     print(b)
        ...     print(c.numpy(), 'torch')
        [ 0  0  0  0  0  0  0  0  0  0  0  0 22 23 24 25 26  0  0  0  0]
        [ 0  0  0  0  0  0  0  0  0  0  0  0 22 23 24 25 26  0  0  0  0] torch
        [ 0  0  0  0  0  0  0  0  0  0  0  0 22 23 24 25 26 27 28 29 30]
        [ 0  0  0  0  0  0  0  0  0  0  0  0 22 23 24 25 26 27 28 29 30] torch
        [ 0  0  0  0  0  0  0 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
        [ 0  0  0  0  0  0  0 17 18 19 20 21 22 23 24 25 26 27 28 29 30] torch
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30]
        [10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30] torch

        """
        return tuple(self._boundaries(start, end))

    def _boundaries(
            self, start, end
    ):
        assert self.shift > 0, self.shift

        not_done = True

        s = start
        s_paste = s
        while not_done:
            e = s + self.length
            if e < end:
                e_paste = e - (self.shift+1) // 2

            else:
                not_done = False
                e = end
                e_paste = end
                s = max(e - self.length, start)
                # yield s, e, s_paste, e_paste
                # return

            s_cut = s_paste - s
            e_cut = e_paste - s

            yield self.Segment(s, e, s_paste, e_paste, s_cut, e_cut, start, end)

            # yield s, e, s_paste, e_paste, s_cut, e_cut
            s += self.shift
            s_paste = e_paste
    #
    # def __call__(self, array, axis=-1):
    #     assert axis == -1, axis
    #     length = shape[axis]
    #
    #     length

# import css.egs.extract.find_optimal_der

import contextlib

@contextlib.contextmanager
def dumper():
    """

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     with dumper() as dump:
    ...         dump(['abc'], f'{tmpdir}/test1.json')
    ...         dump(['def'], f'{tmpdir}/test2.json')

    """
    import threading
    import queue

    q = queue.Queue(maxsize=1)

    def _worker_fn():
        while True:
            tmp = q.get()
            if tmp is None:
                break
            args, kwargs = tmp
            pb.io.dump(*args, **kwargs)
            q.task_done()

    def inner():
        t = threading.Thread(target=_worker_fn, daemon=False)
        t.start()

        while True:
            tmp = yield None
            args, kwargs = tmp
            q.put((args, kwargs))

    gen = inner()
    next(gen)
    dump = lambda *args, **kwargs: gen.send((args, kwargs))
    try:
        yield dump
    finally:
        q.put(None)


class NestedDict(dict):
    """
    Can be used as replacement for dlp_mpi.collection.NestedDict, when
    no MPI is used.

    >>> d = NestedDict()
    >>> d['a']['b'] = 1
    >>> d['a']['c']['d'] = 2
    >>> d['b'] = 3
    >>> print(d.gather())
    {'a': {'b': 1, 'c': {'d': 2}}, 'b': 3}
    """
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    def gather(self):
        return {
            k: v.gather() if isinstance(v, NestedDict) else v
            for k, v in self.items()
        }


@dataclasses.dataclass
class EvalExperiment(pt.Configurable):
    ckpt: str
    eval_dir: str
    save_embedding: bool = True
    save_sad: bool = True
    save_vad_output: bool = True
    save_mask_output: bool = True
    save_audio: bool = True
    save_ex_and_output: bool = False
    reader: 'css.egs.extract.data.LibriCSSRaw' = None
    wpe: 'css.egs.extract.enhancer.WPE' = None
    # aux_data: 'css.egs.extract.data_ivector.MeetingIVectorReader' = None
    probability_to_segments: 'DiscretizeVAD' = _field(
        DiscretizeVAD,
        min_kernel=41,
        max_kernel=81,
        thresh=0.6,
        # css.egs.extract.discretize_sad.HMMSmooth
    )

    # 4000 frames should fit to the memory
    # Rejecting 250 frames on each side from the boarder should be more than enough.
    # Note: overlap = 2 times the number of rejected frames
    # nn_segmenter: 'Segmenter' = _field(Segmenter, length=4000, shift=500)
    nn_segmenter: 'Segmenter' = None

    channel_reduction: str = 'reference'

    # enhancer: css.egs.extract.enhancer.ClassicBF_np = dataclasses.field(default_factory=css.egs.extract.enhancer.ClassicBF_np)
    enhancer: 'css.egs.extract.enhancer.ClassicBF_np' = None

    feature_statistics_domain_adaptation: bool = 'mean_std'
    aux_feature_statistics_domain_adaptation: bool = None
    feature_statistics_domain_adaptation_channel_slice: 'None | str' = None
    feature_statistics_domain_adaptation_channel_wise: 'None | str' = False
    # feature_statistics_domain_adaptation: bool = 'mean'
    # feature_statistics_domain_adaptation: bool = False

    def work(
            self,
            *,
            ex,
            # ds,
            details,
            audio,
            c7json,
            # i,
            t,
            model: 'tssep.train.model.Model',
            feature_transform,
            eg,
            reader,
    ):
        num_samples = ex['observation'].shape[-1]
        ex['Observation'] = model.fe.stft(ex['observation'])
        if not self.save_ex_and_output:
            del ex['observation']
        if self.wpe is not None:
            with t['wpe']:
                ex['Observation'] = self.wpe(ex['Observation'])

        if self.enhancer is not None:
            model.enhancer = tssep.train.enhancer.Dummy()

        def segmented_model_forward(ex, feature_transform):
            Observation = ex['Observation']
            output = model.ForwardOutput(
                mask=None, logit=None, embedding=None, vad_mask=None)

            for i, seg in enumerate(self.nn_segmenter.boundaries(0, Observation.shape[-2])):
                ex['frame_slice'] = {
                    'slice': seg.slice,
                    'orig_num_frames': Observation.shape[-2],
                }
                ex['Observation'] = Observation[..., seg.slice, :]
                ex.pop('Input', None)

                output_: model.ForwardOutput = model.forward(ex, feature_transform=feature_transform)

                output.mask = seg.torch_assign(output.mask, output_.mask, axis=-2)
                if self.save_vad_output:
                    if output_.vad_mask is None:
                        output.vad_mask = None
                    else:
                        output.vad_mask = seg.torch_assign(output.vad_mask, output_.vad_mask, axis=-1)

                if i == 0:
                    output.embedding = output_.embedding
            del ex['frame_slice']
            ex['Observation'] = Observation
            return output

        with t[f'forward {self.channel_reduction}']:
            if self.channel_reduction == 'reference':
                if self.nn_segmenter is None:
                    output: model.ForwardOutput = model.forward(ex, feature_transform=feature_transform)
                    assert output.mask.numel() > 0, output.mask.shape
                else:
                    output: model.ForwardOutput = segmented_model_forward(ex, feature_transform=feature_transform)
                    assert output.mask.numel() > 0, output.mask.shape

            elif self.channel_reduction in ['median', 'none']:
                # Use loop over microphone channel, otherwise some intermediate tensors have a size of 4.92 GiB.
                Observation = ex['Observation']
                output = model.ForwardOutput(mask=None, logit=None, embedding=[], vad_mask=[])

                for i, O in enumerate(Observation):
                    ex['Observation'] = O[None, ...]

                    if self.nn_segmenter is None:
                        output_: model.ForwardOutput = model.forward(ex, feature_transform=feature_transform[i])
                    else:
                        output_: model.ForwardOutput = segmented_model_forward(ex, feature_transform=feature_transform[i])

                    assert output_.mask.numel() > 0, output_.mask.shape
                    if i == 0:
                        # The make takes lots of the memory, hence store it
                        # directly in the output object and avoid the list.
                        # Example:
                        #   Observation: Shape(7, 76971, 257), Bytes(2_215_533_264)
                        #   output_.mask: Shape(8, 1, 76971, 257), Bytes(633_009_504)
                        #   output.mask: Shape(7, 8, 1, 76971, 257), Bytes(4_431_066_528)
                        # Storing it in a list, will cause a peak memory usage
                        # (output.mask exists two times), when torch.stack is
                        # used to create the tensor from the list of tensors.
                        # Preallocating reduces the peak by
                        # 4_431_066_528 - 633_009_504 bytes = 3_798_057_024 bytes.
                        output.mask = output_.mask.new_empty([Observation.shape[0], *output_.mask.shape])
                    assert output.mask.ndim > 1, output.mask.shape
                    output.mask[i, ...] = output_.mask
                    # output.mask.append(output_.mask)
                    output.embedding.append(output_.embedding)

                    if self.save_vad_output:
                        output.vad_mask.append(output_.vad_mask)

                    del output_

                if self.channel_reduction in ['median']:
                    tmp_func = lambda x: torch.median(x, dim=0).values
                elif self.channel_reduction in ['none']:
                    tmp_func = lambda x: rearrange(x, 'channel spk mask time ... -> spk (channel mask) time ...', mask=1)
                else:
                    raise ValueError(self.channel_reduction)

                # output.mask = tmp_func(torch.stack(output.mask, dim=0))
                output.mask = tmp_func(output.mask)
                assert output.mask.numel() > 0, output.mask.shape
                if self.save_vad_output:
                    if output.vad_mask is None or output.vad_mask[0] is None:
                        output.vad_mask = None
                    else:
                        output.vad_mask = tmp_func(torch.stack(output.vad_mask, dim=0))

                if self.enhancer is None:
                    output.stft_estimate = model.enhancer(output.mask, ex, self)

                ex['Observation'] = Observation
                del Observation

                output.embedding = output.embedding[0]  # torch.median(output.embedding, dim=0)

            else:
                raise ValueError(self.channel_reduction)

        if self.save_ex_and_output:
            with t['save_ex_and_output']:
                pb.io.dump(
                    {
                        'ex': ex,
                        'output': output,
                    },
                    Path(self.eval_dir) / 'ex_and_output' / ex['dataset'] / f'{ex["example_id"]}.pth',
                    unsafe=True,
                    mkdir=True,
                    mkdir_parents=True,
                    mkdir_exist_ok=True,
                )

        del output.logit

        if self.enhancer is None:
            time_estimate = pt.utils.to_numpy(
                model.fe.istft(output.stft_estimate, num_samples=num_samples))
        else:
            time_estimate = None

        del output.stft_estimate

        if self.probability_to_segments is not None:
            assert self.probability_to_segments.shift == eg.trainer.model.fe.shift, (
            self.probability_to_segments.shift, eg.trainer.model.fe.shift)

            if self.enhancer is None:
                with self.probability_to_segments as probability_to_segments:
                    segments = probability_to_segments(
                        np.squeeze(  # squeeze mask dimension, it is not used for vad
                            np.mean(output.mask.cpu().numpy(), axis=-1),
                            axis=-2
                        )
                    )
                s: pb.array.interval.ArrayInterval
                time_estimate_segments = [
                    {
                        (s, e): time_estimate[i, s:e]
                        for s, e in s.normalized_intervals
                    }
                    for i, s in enumerate(segments)
                ]
            else:
                with self.probability_to_segments as probability_to_segments:
                    if self.channel_reduction == 'none':
                        # channel_reduction is used to save all masks.
                        # Hence the masks and vad aren't reduced to a single
                        # channel.
                        # The mask type dimension is used to store the channels
                        # Hence in this case, this dim is has more than one entry
                        # Assuming the training would use the mask dim,
                        # the first one should be the correct one.
                        def tmp_squeeze(x, axis):
                            assert axis == -2, axis
                            return x[..., 0, :]
                    else:
                        tmp_squeeze = np.squeeze

                    segments = probability_to_segments(
                        tmp_squeeze(  # squeeze mask dimension, it is not used for vad
                            np.mean(output.mask.cpu().numpy(), axis=-1),
                            axis=-2
                        ),
                        unit='frames',
                    )

                time_estimate_segments = [
                    {
                        start_end_frame: model.fe.istft(v)
                        for start_end_frame, v in estimate_for_spk_idx.items()
                    }
                    for estimate_for_spk_idx in self.enhancer(
                        output.mask,
                        ex['Observation'],
                        dia=segments,
                        segment_bf=True,
                        numpy_out=False,
                    )
                ]
        else:
            time_estimate_segments = None
            if self.enhancer is None:
                pass
            else:
                time_estimate = model.fe.istft(self.enhancer(
                    output.mask,
                    ex['Observation'],
                    dia=None,
                    segment_bf=False,
                    numpy_out=True,
                ), num_samples=num_samples)

        eval_dir = Path(self.eval_dir)

        if self.save_embedding:
            with t['save_embedding']:
                pb.io.dump(
                    output.embedding.cpu().numpy(),
                    eval_dir / 'embedding' / ex['dataset'] / f'{ex["example_id"]}.npy',
                    mkdir=True,
                    mkdir_parents=True,
                    mkdir_exist_ok=True,
                )
        if self.save_sad:
            with t['save_sad']:
                pb.io.dump(
                    np.mean(output.mask.cpu().numpy(), axis=-1),
                    eval_dir / 'sad' / ex['dataset'] / f'{ex["example_id"]}.npy',
                    mkdir=True,
                    mkdir_parents=True,
                    mkdir_exist_ok=True,
                )
        if self.save_vad_output and output.vad_mask is not None:
            with t['save_vad_output']:
                pb.io.dump(
                    output.vad_mask.cpu().numpy(),
                    eval_dir / 'sad' / ex['dataset'] / f'{ex["example_id"]}_vad.npy',
                    mkdir=True,
                    mkdir_parents=True,
                    mkdir_exist_ok=True,
                )
        if self.save_mask_output:
            # max_std_over_f = np.amax(np.std(output.mask.cpu().numpy(), axis=-1))
            # assert max_std_over_f > 1e-5, max_std_over_f
            with t['save_mask_output']:
                pb.io.dump(
                    tssep_data.io.compess.ReduceMaskAsUint8(output.mask.cpu().numpy()),
                    eval_dir / 'mask' / ex['dataset'] / f'{ex["example_id"]}.pkl',
                    mkdir=True,
                    mkdir_parents=True,
                    mkdir_exist_ok=True,
                    unsafe=True,
                )
        if self.save_audio:
            with t['save_audio']:
                if time_estimate_segments is None:
                    file_names = []
                    for i, e in enumerate(time_estimate):
                        file_name = eval_dir / 'audio' / ex['dataset'] / f'{ex["example_id"]}_{i}.wav'
                        file_names.append(file_name)
                        pb.io.dump(
                            e,
                            file_name,
                            mkdir=True,
                            mkdir_parents=True,
                            mkdir_exist_ok=True,
                        )

                    audio[ex['dataset']][ex["example_id"]] = file_names
                else:
                    file_names = [[] for _ in range(len(time_estimate_segments))]
                    num_samples = [[] for _ in range(len(time_estimate_segments))]
                    approx_offset = [[] for _ in range(len(time_estimate_segments))]
                    for spk_idx, estimates in enumerate(time_estimate_segments):
                        for (s, e), estimate in estimates.items():
                            file_name = eval_dir / 'audio' / ex['dataset'] / f'{ex["example_id"]}_{spk_idx}_{s}_{e}.wav'
                            file_names[spk_idx].append(file_name)
                            num_samples[spk_idx].append(estimate.shape[-1])
                            approx_offset[spk_idx].append(pb.transform.module_stft.stft_frame_index_to_sample_index(
                                s, window_length=model.fe.window_length, shift=model.fe.shift,
                                pad=model.fe.pad, fading=model.fe.fading, mode='first', num_samples=None
                            ))
                            pb.io.dump(
                                estimate,
                                file_name,
                                mkdir=True,
                                mkdir_parents=True,
                                mkdir_exist_ok=True,
                            )
                            c7json.append({
                                'start_frame': s,
                                'stop_frame': e,
                                'dataset': ex['dataset'],
                                'session_id': ex['example_id'],
                                'audio_path': str(file_name),
                                'speaker': str(spk_idx),
                                'num_samples': len(estimate),
                                # 'start_sample': ,
                                # 'stop_sample': ,
                                # "start_time": ,
                                # "end_time": ,
                            })

                    audio[ex['dataset']][ex["example_id"]] = {
                        'audio_path': {'estimate': file_names},
                        'num_samples': {'estimate': num_samples},
                        **{
                            k: ex[k]
                            for k in ['transcription', 'kaldi_transcription']
                            if k in ex
                        },
                    }

    def eval(
            self,
            eg: 'css.egs.extract.experiment.Experiment',
    ):
        eval_dir = Path(self.eval_dir)

        print(f'Run eval: {eval_dir}')
        print(f'device: {eg.device}')

        eg.load_model_state_dict(self.ckpt, True)

        model: 'css.egs.extract.model.Model' = eg.trainer.model

        if self.feature_statistics_domain_adaptation:
            # assert self.reader is not None, (self.reader, 'Disable Domain adaptation, when data is the same.')
            from tssep_data.eval.estimate_feature_mean import estimate_mean_std

            allow_thread = True

            # Use pickle to keep the exact values.
            # Json and yaml may change float values.
            feature_statistics_cache = eval_dir / 'cache' / 'feature_statistics.pkl'

            if not feature_statistics_cache.exists():
                eval_mean, eval_std = estimate_mean_std(
                    self.reader or model.reader,
                    self.reader.eval_dataset_name if self.reader else model.reader.eval_dataset_name,
                    model,
                    dtype=torch.float32, device=eg.device, allow_thread=allow_thread,
                    channel_slice=self.feature_statistics_domain_adaptation_channel_slice,
                    channel_wise=self.feature_statistics_domain_adaptation_channel_wise,
                )
                val_mean, val_std = estimate_mean_std(
                    model.reader, model.reader.domain_adaptation_src_dataset_name, model,
                    dtype=torch.float32, device=eg.device,
                    channel_slice=self.feature_statistics_domain_adaptation_channel_slice,
                    # channel_wise=self.feature_statistics_domain_adaptation_channel_wise,
                )
                feature_statistics = {
                    'eval_mean': pt.utils.to_numpy(eval_mean),
                    'eval_std': pt.utils.to_numpy(eval_std),
                    'val_mean': pt.utils.to_numpy(val_mean),
                    'val_std': pt.utils.to_numpy(val_std),
                }
                pb.io.dump(feature_statistics,
                           feature_statistics_cache,
                           mkdir=True, mkdir_exist_ok=True, mkdir_parents=True,
                           unsafe=True)
                pb.io.dump(feature_statistics,
                           feature_statistics_cache.with_suffix('.json'),  # just for logging, i.e., human readable
                           mkdir=True, mkdir_exist_ok=True, mkdir_parents=True)
            else:
                print(f'Load feature statistics from cache: {feature_statistics_cache}')
                feature_statistics = pb.io.load(feature_statistics_cache, unsafe=True)
                feature_statistics = {
                    k: torch.tensor(v, device=eg.device) if isinstance(v, np.ndarray) else v.to(eg.device)
                    for k, v in feature_statistics.items()
                }
                eval_mean, eval_std = feature_statistics['eval_mean'], feature_statistics['eval_std']
                val_mean, val_std = feature_statistics['val_mean'], feature_statistics['val_std']

            def get_feature_transform(val_mean, val_std, eval_mean, eval_std):
                if self.feature_statistics_domain_adaptation == 'mean_std':
                    return lambda x: (x - eval_mean) * (val_std / eval_std) + val_mean
                elif self.feature_statistics_domain_adaptation == 'mean':
                    return lambda x: (x - eval_mean) + val_mean
                else:
                    raise ValueError(self.feature_statistics_domain_adaptation)

            if self.feature_statistics_domain_adaptation_channel_wise is False:
                feature_transform = get_feature_transform(val_mean, val_std, eval_mean, eval_std)
                if self.channel_reduction == 'median':
                    class FeatureTransform:
                        def __init__(self, feature_transform):
                            self.feature_transform = feature_transform
                        def __getitem__(self, item):
                            assert item < 30, item
                            return self.feature_transform

                    feature_transform = FeatureTransform(feature_transform)

            else:
                assert self.channel_reduction in ['median', 'none'], self.channel_reduction

                assert len(eval_mean.shape) == 2, eval_mean.shape
                assert len(val_mean.shape) == 1, val_mean.shape
                assert eval_mean.shape == eval_std.shape, (
                eval_mean.shape, eval_std.shape)
                assert val_mean.shape == val_std.shape, (
                val_mean.shape, val_std.shape)
                assert eval_mean.shape[0] < 30, eval_mean.shape

                feature_transform = [
                    get_feature_transform(val_mean, val_std, eval_mean[i],
                                          eval_std[i])
                    for i in range(eval_mean.shape[0])
                ]

        else:
            feature_transform = None

        if self.aux_feature_statistics_domain_adaptation is not None:
            model.reader.data_hooks.tasks['auxInput'].domain_adaptation(
                model.reader, self.aux_feature_statistics_domain_adaptation,
                consider_mpi=True,
            )
            if 'auxInput2' in model.reader.data_hooks.tasks:
                model.reader.data_hooks.tasks['auxInput2'].domain_adaptation(
                    model.reader, self.aux_feature_statistics_domain_adaptation,
                    consider_mpi=True,
                )

        reader: [
            'tssep_data.data.LibriCSSRaw',
            'tssep_data.data.SimLibriCSS',
        ]
        if self.reader is None:
            reader = model.reader
        elif isinstance(self.reader, dict):
            assert self.reader.keys() == {'channel_slice'}, self.reader
            reader = dataclasses.replace(model.reader, **self.reader)
        else:
            reader = self.reader

        ds = model.prepare_eval_dataset(
            eg.device, prefetch=False, sort=True, reader=reader,
            load_keys=['observation']  # ToDO load more data if nessesary
        )

        t = pb.utils.timer.TimerDict('float')

        model.eval()
        model.to(eg.device)
        with torch.no_grad(), t['total']:
            details = NestedDict()
            audio = NestedDict()
            c7json = []

            print('Use prefetch with threads for dataloading')
            for ex in tqdm.tqdm(t['load_data'](ds.prefetch(2, 3, catch_filter_exception=True)), total=len(ds)):
                self.work(
                    ex=ex,
                    details=details, audio=audio,
                    c7json=c7json,
                    t=t, model=model,
                    feature_transform=feature_transform, eg=eg, reader=reader,
                )
                ex = None
                print(t.as_yaml)

            print('#' * 79)
            print('Finished evaluation.')
            print('#' * 79)

        audio = audio.gather()
        details = details.gather()
        # c7json = c7json.gather()

        print(f'Write summary: {eval_dir}')

        eval_dir = Path(eval_dir)
        pb.io.dump(details, eval_dir / 'details.json')

        pb.io.dump(c7json, eval_dir / 'c7.json')

        if audio:
            for k, v in audio.items():
                pb.io.dump(v, eval_dir / 'audio' / f'{k}.json')

        summary = pb.utils.nested.deflatten({
            k: np.mean([e for _, e in v])
            for k, v in pb.utils.iterable.groupby(
                pb.utils.nested.flatten(details, sep=None).items(),
                lambda x: (x[0][0], *x[0][2:])
            ).items()
        }, sep=None)
        pb.io.dump(summary, eval_dir / 'summary.yaml')
        for k, v in details.items():
            pb.io.dump(v, eval_dir / f'summary_{k}.yaml')

        cmd = f"cat {eval_dir / 'summary.yaml'}"
        print(f'$ {cmd}')
        subprocess.run(cmd, shell=True)


def approx_der(
        sad_estimate,
        sad_target,
        shift_in_seconds,
        threshold=0.45,
        collar=0.25,
):
    """

    Args:
        sad_estimate:
        sad_target:
        shift_in_seconds:
        threshold:
            threshold = np.linspace(0.2, 0.8, 1000, endpoint=False)
            plot.line(threshold, np.sum(((sad_estimate > np.array(threshold)[..., None, None]) ^ sad_target), axis=(1, 2)))
    Returns:

    """

    from pyannote.core import Segment, Annotation
    from pyannote.metrics.diarization import DiarizationErrorRate
    import string

    reference = Annotation()

    for i, t in enumerate(sad_target):
        for s, e in pb.array.interval.ArrayInterval(t).normalized_intervals:
            reference[Segment(s * shift_in_seconds, e * shift_in_seconds)] = string.ascii_uppercase[i]

    hypothesis = Annotation()

    for i, t in enumerate(sad_estimate > threshold):
        for s, e in pb.array.interval.ArrayInterval(t).normalized_intervals:
            hypothesis[Segment(s * shift_in_seconds, e * shift_in_seconds)] = string.ascii_lowercase[i]

    metric = DiarizationErrorRate(collar=collar)
    return metric(reference, hypothesis, detailed=True)

if False:
    import fire

    fire.Fire()

    def cli(*src_dst):
        *src, dst = src_dst
