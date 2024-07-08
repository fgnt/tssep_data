import decimal
import os
import io
from pathlib import Path
import collections

import numpy as np


def read_text(file):
    """
    >>> with open(__file__) as fd:
    ...     print(type(read_text(fd)))
    <class 'str'>
    >>> with open(__file__, 'rb') as fd:
    ...     print(type(read_text(fd)))
    <class 'str'>
    >>> print(type(read_text(__file__)))
    <class 'str'>
    """
    if isinstance(file, io.IOBase):
        if isinstance(file, io.TextIOBase):
            return file.read()
        else:
            return file.read().decode()
    else:
        return Path(file).read_text()


def loads_wav_scp(string):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> pprint(load_wav_scp('/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/ivectors/ivector_online.scp'))  # doctest: +ELLIPSIS
    {'1089-134686-0000': '/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/ivectors/ivector_online.1.ark:17',
     '1089-134686-0001': '/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/ivectors/ivector_online.1.ark:11355',
     '1089-134686-0002': '/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/ivectors/ivector_online.1.ark:15493',
     ...}

    """
    def split(line):
        split = line.split(maxsplit=1)
        try:
            k, v = split
        except ValueError:
            # raise Exception(line)
            #     ValueError: not enough values to unpack (expected 2, got 1)
            assert len(split) == 1, split
            k, = split
            v = ''
        return k, v

    return dict([
        split(line)
        for line in string.splitlines()
        # for k, v in [line.split(maxsplit=1)]
    ])


def load_wav_scp(file):
    return loads_wav_scp(read_text(file))


def loads_utt2spk(string):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> pprint(load_utt2spk('/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/utt2spk'))  # doctest: +ELLIPSIS
    {'1089-134686-0000': '1089',
     '1089-134686-0001': '1089',
     ...}

    """
    return loads_wav_scp(string)


def load_utt2spk(file):
    return loads_utt2spk(read_text(file))


def loads_utt2dur(string):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> pprint(load_utt2dur('/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/utt2dur'))  # doctest: +ELLIPSIS
    {'1089-134686-0000': 10.435,
     '1089-134686-0001': 3.275,
     ...}

    """
    return {k: float(v) for k, v in loads_wav_scp(string).items()}


def load_utt2dur(file):
    return loads_utt2dur(read_text(file))


def loads_text(string):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> pprint(load_text('/mm1/boeddeker/deploy/kaldi/egs/libri_css/s5_mono/data/dev/text.bak'))  # doctest: +ELLIPSIS
    {'1089_session0_CH0_0S_000610_000843': 'BEWARE OF MAKING THAT MISTAKE',
     '1089_session0_CH0_0S_004883_005352': 'FOR A FULL HOUR HE HAD PACED UP AND DOWN WAITING BUT HE COULD WAIT NO LONGER',
     ...}

    """
    return loads_wav_scp(string)


def load_text(file):
    return loads_text(read_text(file))


def loads_spk2utt(string):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> pprint(load_spk2utt('/mm1/boeddeker/librispeech_v1_extractor/ivectors/test_clean/spk2utt'))  # doctest: +ELLIPSIS
    {'1089-134686': ['1089-134686-0000',
      '1089-134686-0001',
      ...],
     '1089-134691': ['1089-134691-0000',
      '1089-134691-0001',
     ...}
    """
    return {
        k: v
        for line in string.splitlines()
        for k, *v in [line.split()]
    }


def load_spk2utt(file):
    return loads_spk2utt(read_text(file))


def load_segments(file):
    """
    <utterance-id> <recording-id> <segment-begin> <segment-end>

    >>> from paderbox.utils.pretty import pprint
    >>> file = '/mm1/boeddeker/deploy/kaldi/egs/chime6/s5c_track2_pretrained/data/dev_beamformit_dereverb_diarized_hires/segments'
    >>> pprint(load_segments(file))  # doctest: +ELLIPSIS
    {'S02_U06.ENH-1-011147-011252': ('S02_U06.ENH', '111.47', '112.52'),...}

    """
    def split(line):
        split = line.split()
        try:
            utterance_id, recording_id, segment_begin, segment_end = split
        except ValueError:
            raise split
        return utterance_id, (recording_id, segment_begin, segment_end)

    return dict([
        split(line)
        for line in Path(file).read_text().splitlines()
    ])


def load_per_utt(file):
    """
    >>> import textwrap
    >>> from paderbox.utils.pretty import pprint
    >>> len(load_per_utt('/mm1/boeddeker/deploy/css/egs/extract/66/eval/5000/2/audio/scoring_kaldi/wer_details/per_utt'))  # doctest: +ELLIPSIS
    432

    # Print does not work in doctest. Maybe that file is too large.
    >>> # pprint(load_per_utt('/mm1/boeddeker/deploy/css/egs/extract/66/eval/5000/2/audio/scoring_kaldi/wer_details/per_utt'))
    """
    def split(line):
        split = line.split(maxsplit=2)
        try:
            k1, k2, v = split
        except ValueError:
            # raise Exception(line)
            #     ValueError: not enough values to unpack (expected 2, got 1)
            assert len(split) == 2, split
            k1, k2 = split
            v = ''
        return k1, k2, v

    ret = collections.defaultdict(dict)
    for line in Path(file).read_text().splitlines():
        k1, k2, v = split(line)
        # print(k1, k2)
        if k2 in ['ref', 'hyp', 'op']:
            v = ' '.join(v.strip().split())
        elif k2 in ['#csid']:
            v = [int(e) for e in v.strip().split()]
        else:
            raise RuntimeError(k1, k2, v)
        ret[k1][k2] = v

    return dict(ret)


def load(file):
    file = Path(file)

    {
        'wav.scp': load_wav_scp,
        'utt2spk': load_utt2spk,
        'utt2dur': load_utt2dur,
        'text': load_text,
        'spk2utt': load_spk2utt,
    }[file.name](file)


def dump_wav_scp(data, file):
    wav_scp = '\n'.join([f'{k} {v}' for k, v in sorted(data.items())]) + '\n'
    Path(file).write_text(wav_scp)


def to_wav_value(
        wav_file,
        start=None,
        stop=None,
        # frames=None,  # Use names from soundfile.read
        unit='samples'  # samples, seconds
):
    """
    !!! Deprecated !!!

    ToDo: Replace this function with to_wav_scp_value
    """
    if start is None and stop is None:
        return f'{wav_file}'
    elif stop is None:
        raise NotImplementedError(start, stop)
    elif start is None:
        raise NotImplementedError(start, stop)
    else:
        assert unit in ['sample', 'samples'], unit
        return f'sox -t wav {wav_file} -t wav - trim {start}s ={stop}s |'


def dump_text(data, file):
    return dump_wav_scp(data, file)


def dump_utt2spk(data, file):
    dump_wav_scp(data, file)


def dump_segments(data, file):
    wav_scp = '\n'.join([f'{k} {reco_id} {start} {end}' for k, (reco_id, start, end) in sorted(data.items())]) + '\n'
    Path(file).write_text(wav_scp)


class KaldiDataDumper:
    """
    Low level interface to dump kaldi
    """
    def __init__(
            self,
            wav_scp=None,  # example_id to path or command to read file
            utt2spk=None,  # example_id to speaker_id
            spk2utt=None,  # speaker_id to list of examples_ids
            text=None,     # example_id
    ):
        self.wav_scp = wav_scp
        self.utt2spk = utt2spk
        self.spk2utt = spk2utt
        self.text = text

    @property
    def utt2spk(self):
        # ToDO: Compute from spk2utt
        return self._utt2spk

    @utt2spk.setter
    def utt2spk(self, value):
        self._utt2spk = value

    @property
    def spk2utt(self):
        if self._spk2utt is None and self._utt2spk is not None:
            # assert self._utt2spk is not None, (self._spk2utt, self._utt2spk)
            spk2utt = collections.defaultdict(list)
            for k, v in self._utt2spk.items():
                spk2utt[v].append(k)
            self._spk2utt = spk2utt

        return self._spk2utt

    @spk2utt.setter
    def spk2utt(self, value):
        self._spk2utt = value

    def dump(self, folder):
        folder = Path(folder)
        wav_scp = self.wav_scp
        if wav_scp is not None:
            dump_wav_scp(wav_scp, folder / 'wav.scp')
            # wav_scp = '\n'.join([f'{k} {v}' for k, v in sorted(wav_scp.items())]) + '\n'
            # (folder / 'wav.scp').write_text(wav_scp)

        utt2spk = self.utt2spk
        if utt2spk is not None:
            dump_wav_scp(utt2spk, folder / 'utt2spk')
            # utt2spk = '\n'.join([f'{k} {v}' for k, v in sorted(utt2spk.items())]) + '\n'
            # (folder / 'utt2spk').write_text(utt2spk)

        spk2utt = self.spk2utt
        if spk2utt is not None:
            dump_wav_scp(spk2utt, folder / 'spk2utt')
            # spk2utt = '\n'.join([f'{k} {" ".join(sorted(v))}' for k, v in sorted(spk2utt.items())]) + '\n'
            # (folder / 'spk2utt').write_text(spk2utt)

        text = self.text
        if text is not None:
            dump_wav_scp(text, folder / 'text')


def array_interval_from_rttm(folder, sample_rate=16000):
    """
    """
    folder = Path(folder)

    # import css.io.kaldi
    import paderbox.array.interval.core
    import paderbox as pb
    segments = load_segments(folder / 'segments')
    utt2spk = load_utt2spk(folder / 'utt2spk')

    assert len(segments) == len(utt2spk), (len(segments), len(utt2spk))
    times = collections.defaultdict(pb.array.interval.zeros)

    for utterance_id, (recording_id, segment_begin, segmen_end) in segments.items():
        # recording_id, start, end = segments[utterance_id]
        segment_begin = int(decimal.Decimal(segment_begin) * sample_rate)
        segmen_end = int(decimal.Decimal(segmen_end) * sample_rate)
        times[recording_id, utt2spk[utterance_id]][segment_begin:segmen_end] = True
    return pb.utils.nested.deflatten(times, sep=None)


def to_wav_scp_value(
        file,
        start=None,  # in samples
        end=None,  # in samples
        segments=None,  # start end pairs in segments
        *,
        norm=None,
        unit='samples',
):
    """



    Notes:
        - To test a sox command use (Norm is usefull to have a similar scale between close talk and far fielf signal):
               sox ... norm -0.1 | play -
        - The segments are concaternated.
        - "10s" means 10 samples and "10" means 10 seconds in sox

    >>> file = 'sample.wav'
    >>> to_wav_scp_value(file)
    'sample.wav'
    >>> to_wav_scp_value(file, 16000, 16000 * 2)
    'sox -t wav sample.wav -t wav - trim 16000s =32000s |'
    >>> to_wav_scp_value(file, segments=[(686956, 714157), (760649, 1010729)])
    'sox -t wav sample.wav -t wav - trim =686956s =714157s =760649s =1010729s |'
    >>> to_wav_scp_value(file, segments=[(686956, 714157), (760649, 1010729)], norm=True)
    'sox -t wav sample.wav -t wav - trim =686956s =714157s =760649s =1010729s norm -0.1 |'
    >>> to_wav_scp_value(file, segments=[(1.2, 2), (3, 4)], norm=True, unit="second")
    'sox -t wav sample.wav -t wav - trim =1.2 =2.0 =3.0 =4.0 norm -0.1 |'

    """
    file = os.fspath(file)
    assert not file.startswith('sox '), file

    if norm is None or norm is False:
        norm = ''
    elif norm is True:
        # https://madskjeldgaard.dk/posts/sox-tutorial-batch-processing/#:~:text=To%20normalize%20a%20file%20in,that%20in%20our%20conversion%20process.
        norm = f'norm -0.1 '
    elif isinstance(norm, float):
        norm = f'norm -{norm} '
    else:
        raise TypeError(type(norm), norm)

    if start is None and end is None and segments is None:
        return f'{file}'
    elif start is None and end is None and isinstance(segments, (tuple, list, np.ndarray)):
        if unit in ['sample', 'samples']:
            segments = np.array(segments)

            assert segments.ndim == 2, segments.shape
            assert segments.shape[1] == 2, segments.shape

            is_sorted = lambda a: np.all(a[:-1] <= a[1:])  # https://stackoverflow.com/a/47004507/5766934
            assert is_sorted(segments.ravel()), segments
            assert segments.dtype == int, segments.dtype
            segments = ' '.join([f'={s}s ={e}s' for s, e in segments])
        elif unit in ['seconds', 'second']:
            # ToDO: Implement test, such that decimal.Decimal can be used.
            segments = ' '.join([f'={s} ={e}' for s, e in segments])
        else:
            raise Exception(type(start), type(end), type(segments), start, end, segments, unit)
        return f'sox -t wav {file} -t wav - trim {segments} {norm}|'
    elif unit in ['sample', 'samples'] and isinstance(start, int) and isinstance(end, int):
        assert start < end, (start, end)
        return f'sox -t wav {file} -t wav - trim {start}s ={end}s {norm}|'
    elif unit in ['seconds', 'second']:
        assert start < end, (start, end)
        return f'sox -t wav {file} -t wav - trim {start} ={end} {norm}|'
    else:
        raise Exception(type(start), type(end), type(segments), start, end, segments, unit)
