import functools

import numpy as np
import dataclasses
import operator

import paderbox as pb
from paderbox.transform.module_stft import stft_frame_index_to_sample_index, sample_index_to_stft_frame_index

# from css.utils import zip_strict
# from css.kernel import Kernel1D

from paderbox.utils.iterable import zip
from paderbox.array.kernel import min_kernel1d, max_kernel1d, median_kernel1d


__all__ = [
    'MaxlenSubsegment',
    'ABCDiscretizeVAD',
    'DiscretizeVAD',
    'MedianDiscretizeVAD',
]


@dataclasses.dataclass
class MaxlenSubsegment:
    max_len: int = 750
    border_margin: int = 20

    # Drop some frame, otherwise ArrayIntervall would merge the segments
    # From the found split position subsegment_margin to the left and right are removed.
    subsegment_margin: int = 5

    max_kernel: int = 21
    threshold: float = 0.2

    algorithm: ['optimal', 'greedy', 'partial_gready'] = 'optimal'

    def __post_init__(self):
        # self.max_kernel_fn = Kernel1D(self.max_kernel, kernel=np.amax)
        self.max_kernel_fn = functools.partial(max_kernel1d, kernel_size=self.max_kernel)
        assert self.border_margin >= 1, self.border_margin
        assert self.max_len >= self.border_margin, (self.max_len, self.border_margin)
        assert self.subsegment_margin >= 0, self.subsegment_margin
        assert self.max_kernel > self.subsegment_margin * 2, (self.max_kernel, self.subsegment_margin)

    def find_split_pos_partial_gready(self, subsegment_candidates, segment_start, segment_end, debug=False):
        """
        First find a gready solution and the remove the last to split positions and find the optimal ones.
        Reason:
            Assuming, there are enough positions to split, the gready algorithm will find ok positions to
            split the data.
            At the end, the last segment might be relative small (its just the rest).
            Hence, drop the estimates at the end and find there the optimal split positions.

        >>> MaxlenSubsegment(max_len=225).find_split_pos_partial_gready([], 0, 500)
        []
        >>> MaxlenSubsegment(max_len=225).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [200, 320]
        >>> MaxlenSubsegment(max_len=250).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [250]
        >>> MaxlenSubsegment(max_len=499).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [250]
        >>> MaxlenSubsegment(max_len=500).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        []
        >>> MaxlenSubsegment(max_len=100).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [100, 200, 300, 400]
        >>> MaxlenSubsegment(max_len=99).find_split_pos_partial_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500, debug=True)
        Trials: 10
        [10, 100, 200, 250, 320, 400]
        >>> MaxlenSubsegment(max_len=100).find_split_pos_partial_gready([i * 100 for i in range(20)], 0, 70*100, debug=True)
        Trials: 3
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        """

        # if (segment_end > segment_start) <= 2 * self.max_len:
        #     return self.find_split_pos(subsegment_candidates, segment_start, segment_end, debug)
        # else:
        out = self.find_split_pos_gready(subsegment_candidates, segment_start, segment_end, debug)
        out = out[:-2]

        if out:
            return out + self.find_split_pos([c for c in subsegment_candidates if c > out[-1]], out[-1], segment_end, debug)
        else:
            return out + self.find_split_pos(subsegment_candidates, segment_start, segment_end, debug)

    def find_split_pos_gready(self, subsegment_candidates, segment_start, segment_end, debug=False):
        """
        >>> MaxlenSubsegment(max_len=225).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [200, 400]
        >>> MaxlenSubsegment(max_len=250).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [400]
        >>> MaxlenSubsegment(max_len=499).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [250]
        >>> MaxlenSubsegment(max_len=500).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        []
        >>> MaxlenSubsegment(max_len=100).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [100, 200, 300, 400]
        >>> MaxlenSubsegment(max_len=99).find_split_pos_gready([10, 100, 200, 250, 300, 310, 320, 400], 0, 500, debug=True)
        [10, 100, 200, 250, 320, 400]
        >>> MaxlenSubsegment(max_len=100).find_split_pos_gready([i * 100 for i in range(20)], 0, 70*100, debug=True)
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        """
        out = []

        if len(subsegment_candidates) > 0:
            i = 0
            pos = segment_start + self.max_len
            while i < len(subsegment_candidates) - 1:
                if subsegment_candidates[i] > pos:
                    out.append(subsegment_candidates[i])
                    pos = subsegment_candidates[i] + self.max_len
                elif subsegment_candidates[i + 1] > pos:
                    out.append(subsegment_candidates[i])
                    pos = subsegment_candidates[i] + self.max_len

                i += 1

            if segment_end > pos:
                out.append(subsegment_candidates[i])

        return out

    def find_split_pos(self, subsegment_candidates, segment_start, segment_end, debug=False):
        """
        Find the best subsegment positions, with the criterions:
         1.: Each subsegment length is smaller or equal to the threshhold
                Exception: In the subsegment is no other subsegment candidate.
         2.: Use a minimum of splits.
         3.: Between the splits that fullfill 1 and 2, select the one,
             where the sum of the quadratic intervall lengthes is minimal.
             This criterion prefares, that the smallest segments are longer.

        Issue:
            Gets very slow, when the signal is too long and there are too many positions to split.
            find_split_pos_partial_gready is solid trade off between speed and optimal solution.

        >>> MaxlenSubsegment(max_len=225).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [200, 320]
        >>> MaxlenSubsegment(max_len=250).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [250]
        >>> MaxlenSubsegment(max_len=499).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [250]
        >>> MaxlenSubsegment(max_len=500).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        []
        >>> MaxlenSubsegment(max_len=100).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500)
        [100, 200, 300, 400]
        >>> MaxlenSubsegment(max_len=99).find_split_pos([10, 100, 200, 250, 300, 310, 320, 400], 0, 500, debug=True)
        Trials: 246
        [10, 100, 200, 250, 320, 400]
        >>> MaxlenSubsegment(max_len=100).find_split_pos([i * 100 for i in range(20)], 0, 70*100, debug=True)
        Trials: 1048574
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        """
        if segment_end - segment_start <= self.max_len or len(subsegment_candidates) == 0:
            return []

        # print('subsegment_candidates', segment_end - segment_start, self.max_len, type(subsegment_candidates), len(subsegment_candidates))

        t = np.diff(subsegment_candidates, prepend=segment_start, append=segment_end)

        best_cost = None
        best_split = None
        i = 0
        for num_splits in range(1, len(subsegment_candidates)+1):

            split_positions = [(0,)]
            for _ in range(num_splits):
                split_positions = (  # generator
                    s + (i,)
                    for s in split_positions
                    for i in range(s[-1] + 1, len(t))
                )
            split_positions = (s[1:] for s in split_positions)  # generator

            for split_position in split_positions:
                i += 1

                splits = np.split(t, split_position)

                num_segments = np.diff(split_position, prepend=0, append=len(t))

                segment_lengths = np.array([sum(s) for s in splits])
                # print(split_position, segment_lengths)
                if any(n > 1 and s > self.max_len for s, n in zip(segment_lengths, num_segments, strict=True)):
                    continue

                cost = np.sum(segment_lengths**2)
                if best_cost is None or best_cost > cost:
                    best_cost = cost
                    best_split = split_position

            # print(thresh ** 2 * (num_splits + 1), best_cost, best_split)
            if best_cost is not None:
                break

        if debug:
            print(f'Trials: {i}')

        assert best_split is not None, 'ToDo: Implement fallback'
        # print(best_split, np.split(subsegment_candidates, best_split))

        best_split = [s[-1] for s in np.split(subsegment_candidates, best_split)[:-1]]
        return best_split

    def discretize(self, sad):
        x = self.max_kernel_fn(sad)
        x = x <= self.threshold
        x = pb.array.interval.ArrayInterval(x)
        return x

    def __call__(
            self,
            sad,
    ):
        """
        >>> import textwrap
        >>> f = '/mm1/boeddeker/deploy/css/egs/extract/66/eval/5000/2/sad/OV40/overlap_ratio_40.0_sil0.1_1.0_session6_actual39.9.npy'
        >>> sad = np.squeeze(pb.io.load(f, unsafe=True), axis=1)[0, 7489:8994]
        >>> pb.utils.pretty.pprint(sad)
        array(shape=(1505,), dtype=float32)
        >>> MaxlenSubsegment(600, 20)
        MaxlenSubsegment(max_len=600, border_margin=20, subsegment_margin=5, max_kernel=21, threshold=0.2)
        >>> MaxlenSubsegment(600, 20).discretize(sad)
        ArrayInterval("0:9, 85:114, 168:176, 310:319, 389:393, 483:485, 613:614, 791:793, 849:856, 936:937, 1281:1316", shape=(1505,))
        >>> MaxlenSubsegment(600, 20)(sad)
        ((0, 479), (489, 931), (941, 1505))
        >>> from css.plot import ascii
        >>> for s, e in MaxlenSubsegment(600, 20)(sad):
        ...     print(s, e)
        ...     d = ascii(sad[None, s:e], values='▁▂▃▄▅▆▇█', file=None)
        ...     print(*textwrap.wrap(d), sep='\\n')
        0 479
        ▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▂▂▅▆▆▆▆▆▅▇█▇▇▆▇▇▇▆▅▄▄▄▄▃▃▄▅▇▆▅▅▄▄▄▄▃▃▃▅▅▄▄▅▃▂▅▅▅▆▇▇▇▆
        ▅▄▃▄▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▂▅▇▇▆▅▃▇▇██▇▇▇▆▇▆
        ▅▅▄▄▆▆▇█▇▇▇▅▄▆▅▄▄▃▂▂▁▁▁▁▂▁▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▂▁▂▁▂▃▂▄███▇▇▇▇▅▅▆▇▇▇▇▆▇▇▆▅▆▆
        ▆▆▆▆▅▄▄▅▃▂▂▂▁▂▁▁▁▁▁▁▁▁▁▂▄▅▆▅▅▅▄▄▆▆▆▅▅▆▆▆▆▇▇▄▅▄▄▅▆▇▆▅▄▃▂▅▃▁▁▂▂▃▅▄▅▇█▇▆▇
        ▇▆▇▇█▇▇▇▆▇█▇▇▇▇▆▅▆▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▆▄▄▇██▇█▇▇▆▄▇▇▇▇▇▇█▇▇
        ▇▇▇▆▅▄▆▇▅▆▆▇▇▇▇▆▄██▇▇▇▇▇▆▅▄▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▇▇▆▇███▇▇▇▆▆▇█▇▆▇
        ▇▅▃▇▆▄▅▅▅▅▅▅▄▂▃▆▆▇▇▇▆▅▄▃▂▂▆▅▆▇▇▇▇▅▆▇▆▅▄▄▅▆▆▄▄▄▄▄▆▄▄▄▃▂▁▁▁▁▁
        489 931
        ▁▁▁▁▂▂▃▅▅▆▅▅▆▇▇▆▇▆▆▇▇▇▄▃▂▂▁▁▁▁▁▁▄▇█▆▄▇██▇▆█▇▇▇▇▆█▇▆▅█▇███▇▇▇▅▃▂▇▇▇▇▇█▇
        ▇▆▆▆▇▇▇▇▇▆██▇▇▆▆█▇▇▆▇▆▇▇▇▇▇▇▄▃▂██▅▅▆▇▇▇▇▇▇▆▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▇▇▇▇
        ▇▇▅▇█▆▅▇▇█▇▆▇▇▇███▇▇▅▇▇▆▅▆▆▇▇▇▇▇▇▅▃▇▇▅▆▇▇▆▅▄▆▇▇▅▄▇▇▇▆█▇▇▆▅▇▄▇▆▄▅▄▃▃▆▅▄
        ▃▄▅▇▇█▇▆▄▃▇▇▇▇▇▇▆▇▇▇▄▄▄▂▂▄█▆▃▂▇▅▆▅▆▆▇▇█▇█▆▆▇▇▆▆▇▅▄▄▄▅▇▇▇▆▆▇▅▄▄▅▅▆▇▆▄▄▄
        ▆▇▆▇▆▆▇▇▇▆▄▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▁▁▂▇▇▆▄▇▇▇▇█▆▆▅█▇▅▅▆▇▆▆▆▆▃▅▅▇███▇▇▆▄▂▃▄
        ▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▂▅█▄▇███████▆▆▆██▇▇▆▆▆▆▇▇▃▂▂▃▆▇▇▇▇▇▅▅▆█▇▆▄█▇
        ▇▆█▇▇▆▆▆▆▅▅▇▇▇▇▄▃▂▂▁▁▁
        941 1505
        ▁▁▁▂▂▂▃▆▇▇▄█████▇▅▅▆▆▅▅▃▂▂▂▇▅▄▅▅▅▇▆▅▃▂▂▁▂▁▁▁▁▁▁▁▁▁▁▅▆▇▇▆▅▄▅▆▇▇▇▅▆▇▇▄▅▅
        ▄▂▄▅▄▃▂▅▅▃▇▅▄▆▇▇█▇▇▇▅▃▄▆▆▇▄▂▅▃▃▅▆▅▅▄▄▃▂▂▂▂▁▆▇▇▅▅█▇▇▇▇▇▇▆▆▆▇█▇▆▆▇▇▇▆▅▆▅
        ▅▄▃▄▅▇▆▆▆▆▆▄▆▇▆▅▆▇▇▆▇▇▅▆▇▇▇▇▇▅▄▅▇▇▇▇▇▇▇▆▅▄▅▄▇█▇▅▆▆▅▅▅▅▃▄▆▅▃▂▂▂▁▁▁▁▁▁▂▁
        ▁▁▁▃▅▆▅▄▆▇█▇▇▇▇▇▇▆▅▄▃▂▂▁▁▂▂▂▂▂▂▁▂▂▄▇▇▇▇▇███▆▅▇▇▆█▇▅▇▇▇▅▄▄▇████▇▄▆▆▅▄▇▇
        ▇▇██▇▇▅▃▄▇▆▆▆▇▇▇▇▇▇▇▇▇▇▆▄▃▆▆▆▅▂▂█▇▅▄▄▇▇▇▇▆▄▄▅▅▄▄▃▃▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▂██▇██▇▇██▆▄▃▆▇▆▅▃▇█▇▇█▇▆▅▃▃▅▇▄▂▁▅▃▂
        ▂▇▇▇▇▇▅▆▇▅▂▁▂▂▂▁▂▄▂▂▂▂▃▃▂▂▂▂▂▅▄▃▃▄▄▂▂▂▂▁▁▁▂▂▂▂▂▂▂▂▃▃▂▂▆▅▄▃▅▆▆▄▆▆▆▅▆▅▄▅
        ▇▇▅▇▇▅▄▄▃▂▅▅▅▅▄▄▄▃▃▃▅▆▄▄▃▂▄▆▆▆▅▃▂▂▁▂▃▂▂▃▃▂▃▄▄▄▇▇▇▆▄▅▆▅▅▄▃▂▂▄▄▃▃▂▁▁▂▂▂▃
        ▃▃▃▃
        >>> 1500 * 256 / 16000
        24.0
        """
        assert sad.ndim == 1, sad.shape

        x = self.discretize(sad)
        # subsegment_candidates = np.mean(x.normalized_intervals, axis=1).astype(int)

        length = len(sad)
        subsegment_candidates = [
            (e + s) // 2
            for s, e in x.normalized_intervals
            if s >= self.border_margin and (length - e) >= self.border_margin
        ]

        # positions = [0] + [self.find_split_pos(subsegment_candidates, 0, length) + [length]
        # return tuple(zip(positions, positions[1:]))

        last = 0
        positions = []

        if self.algorithm == 'optimal':
            find_split_pos = self.find_split_pos
        elif self.algorithm == 'greedy':
            find_split_pos = self.find_split_pos_gready
        elif self.algorithm == 'partial_gready':
            find_split_pos = self.find_split_pos_partial_gready
        else:
            raise ValueError(self.algorithm)

        for s in find_split_pos(subsegment_candidates, 0, length):
            positions.append((last, s - self.subsegment_margin))
            last = s + self.subsegment_margin
        positions.append((last, length))
        return tuple(positions)


@dataclasses.dataclass
class ABCDiscretizeVAD:
    # min_kernel: int = 41
    # max_kernel: int = 81
    # thresh: float = 0.6

    min_frames: int = 0

    # subsegment: MaxlenSubsegment = None
    subsegment: MaxlenSubsegment = dataclasses.field(default_factory=MaxlenSubsegment)

    window_length: int = 1024
    shift: int = 256
    fading: bool = True

    def msg(self):
        raise NotImplementedError()
        return (self.min_kernel, self.max_kernel, self.thresh)

    def core(self, sad):
        raise NotImplementedError()
        x = Kernel1D(self.min_kernel, kernel=np.amin)(
            Kernel1D(self.max_kernel, kernel=np.amax)(sad))
        return x >= self.thresh

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(
            self,
            sad: np.ndarray,
            unit='samples',
            start_samples=0,
    ) -> list:
        """
        >>> from paderbox.utils.pretty import pprint
        >>> sad = pb.io.load('/mm1/boeddeker/deploy/css/egs/extract/30/eval/25000/1/sad/OV40/overlap_ratio_40.0_sil0.1_1.0_session1_actual39.7.npy', unsafe=True)
        >>> sad.shape
        (8, 1, 37540)
        >>> pprint(DiscretizeVAD()(sad[:, 0, :]))
        [ArrayInterval("48896:66815, 77056:229631, 412160:489983, 1016576:1267711, 2648064:2676735, 2701312:2743295, 3249408:3503359, 3513856:3525631, 3830784:3973375, 3985664:3997183, 4338688:4444159, 5030400:5042175, 5853184:5871103, 5961472:6085375, 6219776:6299135, 7044352:7073279, 7086080:7100159, 7124480:7354367, 7367424:7378943, 7713280:7787007, 9221120:9267455", shape=None),
         ArrayInterval("30208:44543, 1040384:1054207, 1212928:1425919, 1444608:1500671, 1934848:2084095, 2744320:2762239, 4499712:4521983, 4609024:4733951, 4953856:5089535, 5867008:5878527, 6283264:6306303, 6331648:6375679, 6390784:6545919, 6564352:6577407, 7498496:7519999, 7596800:7728639, 7765248:7789567, 8819200:8872191, 9222400:9284863, 9500672:9512447", shape=None),
         ArrayInterval("0:23295, 64512:85247, 101888:135167, 199424:220159, 649984:752895, 1018368:1036543, 1327360:1345535, 1871104:1909759, 2667264:2693887, 2704384:2715903, 3072000:3138303, 5044480:5078783, 5090048:5153535, 6364416:6400511, 7090432:7159295, 7598848:7610367, 7662336:7723775, 8690432:8827135, 8852224:8863743, 8921344:8963327, 9171712:9235455", shape=None),
         ArrayInterval("244224:272639, 697856:740095, 798464:894463, 924416:980735, 1642240:1653759, 2254848:2344447, 3459072:3552767, 3592704:3719423, 4587520:4599295, 5894400:5910015, 6550272:6703103, 6741504:6804735, 8819456:8832255, 8869632:8910335, 9013248:9139711, 9294336:9447423", shape=None),
         ArrayInterval("711168:723967, 742656:756991, 798464:866047, 892416:938495, 1336320:1358335, 1398528:1500671, 3756800:3770623, 5145344:5169151, 5420544:5483007, 5607936:5637375, 6094336:6126591, 6284288:6345983, 6505984:6518783, 6529280:6541311, 6729728:6744831, 8401920:8416511, 8665344:8699135", shape=None),
         ArrayInterval("361728:414975, 5142784:5206271, 5220352:5540607, 5617920:5673983, 6081024:6165503, 6298368:6317311, 6331904:6343423, 6498304:6548991, 6686208:6745087, 7456256:7468031, 8121600:8173311, 8419840:8710143", shape=None),
         ArrayInterval("283136:376831, 478976:665087, 2071552:2145279, 2753792:3077375, 3123456:3299071, 3464448:3589887, 3606272:3627263, 3685888:3828223, 3902208:3913727, 4097536:4382719, 4805376:4964863, 5470208:5736191, 5775616:5787135, 5804800:5819391, 5884928:5897727, 6083840:6095359, 6138624:6223103, 6724096:6735615, 6793472:6936063, 7335936:7516927, 7785216:7903487, 8073472:8439039, 9415936:9427711, 9457664:9505791, 9529856:9610495", shape=None),
         ArrayInterval("745728:841983, 919040:935423, 975616:1008383, 1486080:1939199, 2107904:2648319, 3968512:4121855, 4134912:4146687, 4160000:4189183, 4615424:4663551, 4686592:4828671, 6906880:7087615, 7522560:7598847, 7867648:8035071, 8087040:8130303, 8905728:9055999, 9097728:9181183, 9283584:9351423", shape=None)]

        >>> pprint(DiscretizeVAD()(sad[0, 0, :]))
        ArrayInterval("48896:66815, 77056:229631, 412160:489983, 1016576:1267711, 2648064:2676735, 2701312:2743295, 3249408:3503359, 3513856:3525631, 3830784:3973375, 3985664:3997183, 4338688:4444159, 5030400:5042175, 5853184:5871103, 5961472:6085375, 6219776:6299135, 7044352:7073279, 7086080:7100159, 7124480:7354367, 7367424:7378943, 7713280:7787007, 9221120:9267455", shape=None)

        >>> vad: pb.array.interval.ArrayInterval
        >>> for vad in DiscretizeVAD()(sad[:, 0, :]):
        ...     print(vad.sum() / vad.normalized_intervals[-1][-1])
        0.18698153916042753
        0.1365499329457499
        0.09853970378286722
        0.1186575429087911
        0.06629981026849222
        0.12300004718636652
        0.3077942395266841
        0.24517327469840686
        >>> pprint(DiscretizeVAD()(sad[0, 0, :], unit='frames'))
        ArrayInterval("194:260, 304:896, 1613:1913, 3974:4951, 10347:10455, 10555:10715, 12696:13684, 13729:13771, 14967:15520, 15572:15613, 16951:17359, 19653:19695, 22867:22933, 23290:23770, 24299:24605, 27520:27629, 27683:27734, 27833:28727, 28782:28823, 30133:30417, 36023:36200", shape=None)

        """

        if unit.lower() in ['samples', 'sample']:
            samples = True
        elif unit.lower() in ['frames', 'frame']:
            samples = False
            if start_samples != 0:
                raise NotImplementedError(unit, start_samples)
        else:
            raise ValueError(unit)

        # sad = np.squeeze(sad, axis=1)

        x = sad
        x = max_kernel1d(x, self.max_kernel)
        x = min_kernel1d(x, self.min_kernel)
        # x = Kernel1D(self.min_kernel, kernel=np.amin)(Kernel1D(self.max_kernel, kernel=np.amax)(sad))
        x = x >= self.thresh

        data = np.empty(x.shape[:-1], dtype=object)
        for idx in np.ndindex(x.shape[:-1]):

            a = x[idx]
            ai = pb.array.interval.zeros()

            normalized_intervals = pb.array.interval.ArrayInterval(a).normalized_intervals
            if normalized_intervals:  # normalized_intervals is empty, when a has no True value
                if self.min_frames:
                    assert self.min_frames > 0, self.min_frames
                    normalized_intervals = [
                        (s, e)
                        for s, e in normalized_intervals
                        if e - s >= self.min_frames
                    ]
                if self.subsegment is not None:
                    normalized_intervals = [
                        (s + s_rel, s + e_rel)
                        for s, e in normalized_intervals
                        for s_rel, e_rel in self.subsegment(sad[idx][s:e])
                    ]

                if len(normalized_intervals) == 0:
                    starts, ends = [], []
                else:
                    starts, ends = np.array(normalized_intervals).T

                    if samples:
                        starts, ends = self.frames_to_samples(starts, ends, start_samples)
                        # starts = start_samples + stft_frame_index_to_sample_index(
                        #     starts, window_length=self.window_length, shift=self.shift, fading=self.fading, mode='first')
                        # ends = start_samples + stft_frame_index_to_sample_index(
                        #     ends, window_length=self.window_length, shift=self.shift, fading=self.fading, mode='last')

                for start, end in zip(starts, ends):
                    ai[start:end] = True

            data[idx] = ai
        return data.tolist()

    def frames_to_samples(self, starts, ends, start_samples):
        """

        Args:
            starts: Frame indices for the start of the segments
            ends: Frame indices for the end of the segments
            start_samples: Offset in samples (i.e. the ignored part when loading the audio)

        Returns:
            starts, ends in sample resolution

        """
        starts = start_samples + stft_frame_index_to_sample_index(
            starts, window_length=self.window_length, shift=self.shift, fading=self.fading, mode='first')
        ends = start_samples + stft_frame_index_to_sample_index(
            ends, window_length=self.window_length, shift=self.shift, fading=self.fading, mode='last')
        return starts, ends


@dataclasses.dataclass
class DiscretizeVAD(ABCDiscretizeVAD):
    """
    >>> DiscretizeVAD()
    DiscretizeVAD(min_frames=0, subsegment=MaxlenSubsegment(max_len=750, border_margin=20, subsegment_margin=5, max_kernel=21, threshold=0.2, algorithm='optimal'), window_length=1024, shift=256, fading=True, min_kernel=41, max_kernel=81, thresh=0.6)
    >>> DiscretizeVAD(subsegment=None)
    DiscretizeVAD(min_frames=0, subsegment=None, window_length=1024, shift=256, fading=True, min_kernel=41, max_kernel=81, thresh=0.6)
    """
    min_kernel: int = 41
    max_kernel: int = 81
    thresh: float = 0.6


    def msg(self):
        return (self.min_kernel, self.max_kernel, self.thresh)

    def core(self, sad):
        x = sad
        x = max_kernel1d(x, self.max_kernel)
        x = min_kernel1d(x, self.min_kernel)
        # x = Kernel1D(self.min_kernel, kernel=np.amin)(
        #     Kernel1D(self.max_kernel, kernel=np.amax)(sad))
        return x >= self.thresh


@dataclasses.dataclass
class MedianDiscretizeVAD(ABCDiscretizeVAD):
    median_kernel: int = 51
    # min_kernel: int = 41
    # max_kernel: int = 81
    thresh: float = 0.4

    min_frames: int = 13  # Kaldi: 0.2 seconds -> 12.5 with sample_rate = 16000
    min_silence_frames: int = 19  # Kaldi: 0.3 seconds -> 18.75 with sample_rate = 16000

    def msg(self):
        return ('median', self.median_kernel, self.min_frames, self.min_silence_frames, self.thresh)

    def core(self, sad):
        """
        >>> from css.plot import ascii
        >>> def do(k, x):
        ...     x = np.array(x)
        ...     ascii(x[None, :], values='▁█')
        ...     ascii(k.core(x)[None, :], values='▁█')
        >>> k = MedianDiscretizeVAD(median_kernel=1, min_frames=3, min_silence_frames=5)
        >>> x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        >>> do(k, x)
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        >>> k = MedianDiscretizeVAD(median_kernel=1, min_frames=5, min_silence_frames=5)
        >>> do(k, x)  # increase min_frames, will drop first 1 block
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████▁▁▁▁▁
        >>> k = MedianDiscretizeVAD(median_kernel=1, min_frames=3, min_silence_frames=7)
        >>> do(k, x)  # increase min_silence_frames, will drop slience between activity
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁██████████████▁▁▁▁▁
        >>> k = MedianDiscretizeVAD(median_kernel=1, min_frames=7, min_silence_frames=5)
        >>> do(k, x)  # increase min_frames, such that all are dropped
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        """
        # ToDo: Check what the kaldi code does.
        raise NotImplementedError()
        x = sad
        x = Kernel1D(self.median_kernel, kernel=np.amax)(x)
        x = x >= self.thresh
        x = Kernel1D(self.min_silence_frames, kernel=np.amax)(x)
        x = Kernel1D(self.min_frames + self.min_silence_frames - 1, kernel=np.amin)(x)
        x = Kernel1D(self.min_frames, kernel=np.amax)(x)

        return x


@dataclasses.dataclass
class KaldiMedianDiscretizeVAD(MedianDiscretizeVAD):
    def core(self, sad):
        """
        >>> from css.plot import ascii
        >>> def do(k, x):
        ...     x = np.array(x)
        ...     ascii(x[None, :], values='▁█')
        ...     ascii(k.core(x)[None, :], values='▁█')
        >>> k = KaldiMedianDiscretizeVAD(median_kernel=1, min_frames=3, min_silence_frames=5)
        >>> x = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        >>> do(k, x)
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        >>> k = KaldiMedianDiscretizeVAD(median_kernel=1, min_frames=5, min_silence_frames=5)
        >>> do(k, x)  # increase min_frames, will drop first 1 block
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████▁▁▁▁▁
        >>> k = KaldiMedianDiscretizeVAD(median_kernel=1, min_frames=3, min_silence_frames=7)
        >>> do(k, x)  # increase min_silence_frames, will drop slience between activity
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁███████████████████
        >>> k = KaldiMedianDiscretizeVAD(median_kernel=1, min_frames=7, min_silence_frames=5)
        >>> do(k, x)  # increase min_frames, such that all are dropped
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        >>> k = KaldiMedianDiscretizeVAD(median_kernel=1, min_frames=7, min_silence_frames=7)
        >>> do(k, x)  # increase min_frames, such that all are dropped
        ▁▁▁▁▁████▁▁▁▁▁█████▁▁▁▁▁
        ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
        """
        x = sad
        x = median_kernel1d(x, self.median_kernel)
        # x = Kernel1D(self.median_kernel, kernel=np.amax)(x)
        x = x >= self.thresh
        x = pb.array.interval.ArrayInterval(x)
        assert x.inverse_mode is False, x.inverse_mode
        not_x = pb.array.interval.core._combine(operator.__not__, x)
        assert x.inverse_mode is False, x.inverse_mode

        new = pb.array.interval.zeros(x.shape)
        # last_e = 0
        prev_was_speech = None
        for s, e, speech in sorted([
            (s, e, True)
            for s, e in x.normalized_intervals
        ] + [
            (s, e, False)
            for s, e in not_x.normalized_intervals
        ]):
            if s == 0:
                new[s:e] = speech
                prev_was_speech = speech
            else:
                min_length = self.min_frames if speech else self.min_silence_frames

                if e - s >= min_length:
                    new[s:e] = speech
                    prev_was_speech = speech
                else:
                    new[s:e] = prev_was_speech

        # x = Kernel1D(self.min_silence_frames, kernel=np.amax)(x)
        # x = Kernel1D(self.min_frames + self.min_silence_frames - 1, kernel=np.amin)(x)
        # x = Kernel1D(self.min_frames, kernel=np.amax)(x)

        return np.array(new)
