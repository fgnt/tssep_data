"""
Changes compared to gss.py:
 - Load segments of audio instead of all audio
    - Necessary on CHiME-5/6/7, because on session takes 100 GB memory
 - Changed interface, i.e. it no longer assumes an experiment dir
     - Take per_utt.json and database json as input
 - Remove frame level estimate dependency

/scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/144/eval/10000/1
$ sbatch.py -t 1000 -n 200 --mem-per-cpu 4G --wrap 'srun python -m css.egs.extract.gss_v2 audio/dia_c7.json /scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7.json,/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/sim_libri_css_4spk/sim_libri_css_4spk.json --out_folder=audio_gss --channel_slice=:'
busy: 0: 100%|██████████| 5063/5063 [3:50:30<00:00,  2.73s/it]

sbatch.py -n 200 --time 12h --mem-per-cpu 6G --wrap "srun.py python -m cbj.transcribe.cli chime7 audio_gss/gss_c7.json --model_tag=chime7"

"""
import decimal
from pathlib import Path
import collections
import dataclasses

import numpy as np
import tqdm
import einops

import paderbox as pb
import padertorch as pt
import dlp_mpi.collection
import lazy_dataset.database

from meeteval.io.stm import STM, STMLine
import pb_chime5
from pb_chime5.core_chime6 import get_enhancer as get_enhancer_chime6


def foo(
        Obs,
        ex_array_activity,
        # initialization,
        speaker_id,

        begin_index,
        end_index,
        context_frames,

        gss: pb_chime5.core_chime6.Enhancer,

        ex=None,
        debug=False,

):
    # s = slice(max(0, begin_index - context_frames),
    #           min(Obs.shape[1], end_index + context_frames))
    s = slice(max(0, begin_index - context_frames), end_index + context_frames)

    Obs = Obs[:, s, :]

    ex_array_activity = {
        k: v[s]
        for k, v in ex_array_activity.items()
    }

    # if initialization is not None:
    #     initialization = einops.rearrange(
    #         initialization[..., s, :], 'k 1 t f -> f k t',
    #         k=len(ex_array_activity)-1,  #
    #         f=513,
    #     )
    #
    #     # Add Noise class
    #     noise_class_init = np.maximum(1e-10, 1 - np.sum(initialization, axis=1, keepdims=True))
    #     initialization = np.concatenate([initialization, noise_class_init], axis=1)
    #     initialization = np.maximum(initialization, 1e-10)
    #     initialization /= np.sum(initialization, keepdims=True, axis=1)
    #     # print(initialization.shape, 'initialization', flush=True)
    #
    #     # raise Exception(initialization.shape)

    Obs = gss.wpe_block(Obs, debug=debug)
    masks = gss.gss_block.__call__(
        Obs,
        list(ex_array_activity.values()),
        # initialization=initialization,
        debug=debug,
    )

    # ToDo: remove context for BF

    example_id = ex['example_id'] if ex is not None and 'example_id' in ex else 'Unknown'

    old_shape = masks.shape
    masks = masks[:, begin_index - s.start:end_index - s.start, :]
    Obs = Obs[:, begin_index - s.start:end_index - s.start, :]
    if Obs.shape[1] == end_index - begin_index:
        pass
    elif Obs.shape[1] > (end_index - begin_index) * 0.2:  # ToDO: Fix that this don't happen in Dipco
        print(f'WARNING: {example_id}: Obs.shape[1]({Obs.shape[1]}) != (end_index({end_index}) - begin_index({begin_index}))')
    else:
        assert masks.shape[1] == end_index - begin_index, (old_shape, s, example_id, masks.shape, begin_index, end_index)
        assert Obs.shape[1] == end_index - begin_index, (Obs.shape, begin_index, end_index)

    target_speaker_index = tuple(ex_array_activity.keys()).index(speaker_id)

    target_mask = masks[target_speaker_index]
    distortion_mask = np.sum(
        np.delete(masks, target_speaker_index, axis=0),
        axis=0,
    )

    X_hat = gss.bf_block(
        Obs,
        target_mask=target_mask,
        distortion_mask=distortion_mask,
        debug=debug,
    )

    return X_hat


class DefaultList(collections.UserList):
    """
    >>> l = DefaultList(dict)
    >>> l[3]
    []
    >>> l
    [{}, {}, {}, {}]
    """
    def __init__(self, default_factory):
        super().__init__([])
        assert callable(default_factory), default_factory
        self.default_factory = default_factory

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except IndexError:
            pass
        while item >= len(self.data):
            self.data.append(self.default_factory())
        return super().__getitem__(item)

    def __iter__(self):
        return iter(self.data)


class LazyLoadObservation:
    """

    ToDo: implement a caching, e.g. have an audio cache of 3 minutes
          and reload data in minut chunks.

    >>> from paderbox.utils.pretty import pprint
    >>> f = '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/144/eval/10000/1/audio/dev_chime6/S02_0_0_53.wav'
    >>> o = LazyLoadObservation([f], 0, 100_000, debug=True)
    >>> o.stft
    STFT(shift=256, size=1024, window_length=1024, window='hann', symmetric_window=False, pad=True, fading=True)
    >>> orig = o.stft(pb.io.load_audio(f, frames=16000 * 1))[None, ...]
    >>> orig.shape
    (1, 66, 513)
    >>> a = o[:, 5:10, :]
    LazyLoadObservation read from 512 to 2560: (1, 2048)
    >>> np.testing.assert_equal(orig[:, 5:10, :], a)
    >>> b = o[:, 0:10, :]
    LazyLoadObservation read from 0 to 2560: (1, 2560)
    >>> np.testing.assert_equal(orig[:, 0:10, :], b)
    >>> pprint(a)
    array(shape=(1, 5, 513), dtype=complex128)
    >>> pprint(b)
    array(shape=(1, 10, 513), dtype=complex128)

    """

    def __init__(self, files, start, end, stft, debug=False):
        self.files = files
        # self.start = start
        # self.end = end
        # self.fe: css.egs.extract.feature_extractor.AbsSTFT = fe
        # css.egs.extract.feature_extractor.AbsSTFT(fading=False)
        self.debug = debug

        self.stft = stft

        # ToDo: Add caching

    def __getitem__(self, item):
        assert len(item) == 3, item
        assert item[0] == slice(None) == item[2], item

        start, end = item[1].start, item[1].stop

        assert start >= 0, (start, end)
        assert end >= 0, (start, end)
        assert end - start > 0, (start, end)

        start_sample = pb.transform.module_stft.stft_frame_index_to_sample_index(
            start,
            window_length=self.stft.window_length,
            shift=self.stft.shift,
            pad=self.stft.pad,
            fading=self.stft.fading,
            mode='first'
        )
        end_sample = pb.transform.module_stft.stft_frame_index_to_sample_index(
            end - 1,
            window_length=self.stft.window_length,
            shift=self.stft.shift,
            pad=self.stft.pad,
            fading=self.stft.fading,
            mode='last'
        ) + 1

        # fading_length = window_length - shift

        data = np.array(
            [pb.io.load_audio(f, start=start_sample, stop=end_sample)
             for f in self.files])

        assert data.shape[-1] > 0, (data.shape, start_sample, end_sample)

        if data.dtype == object:
            lens = [len(e) for e in data]
            max_len = max(lens)
            min_len = min(lens)
            if min_len / max_len >= 0.8:
                data = np.array([
                    e[:min_len]
                    for e in data
                ])
            else:
                data = np.array([
                    e
                    for e in data
                    if len(e) == max_len
                ])
                assert len(data) >= 3, (lens, data)

        assert data.dtype != object, (data.shape, start_sample, end_sample, data)
        assert data.size, (data.shape, start_sample, end_sample, data)

        if self.debug:
            print(self.__class__.__name__, f'read from {start_sample} to {end_sample}: {data.shape}')

        if self.stft.fading:
            assert self.stft.fading is True, self.stft
            assert self.stft.window_length // self.stft.shift >= 1, self.stft
            assert self.stft.window_length % self.stft.shift == 0, self.stft

            first_frame = min(self.stft.window_length // self.stft.shift - 1, start)
            return self.stft(data)[:, first_frame:first_frame+end-start, :]
        else:
            return self.stft(data)

        # "start": 649600,
        # "end": 142472247


class ExArrayActivity:
    def __init__(self, stm_lines: lazy_dataset.Dataset, stft):
        # Channel is the dataset
        self.stms = stm_lines.groupby(lambda x: (x.dataset, x.session_id))
        self.last_key = None
        self.last_value = None
        self.stft = stft

    #     self.stft.sample_index_to_frame_index(segment.stop_sample)

    def __getitem__(self, segment):
        item = segment.dataset, segment.session_id

        if item == self.last_key:
            return self.last_value

        stm = self.stms[item]

        # ex_array_activity = DefaultList(pb.array.interval.zeros)
        ex_array_activity = collections.defaultdict(pb.array.interval.zeros)

        for line in stm:
            # speaker_index = int(line.speaker)
            ex_array_activity[line.speaker][
                line.start_sample:line.stop_sample] = True

        # ex_array_activity = {str(k): v for k, v in enumerate(ex_array_activity)}
        ex_array_activity = {
            k: v.from_pairs(
                self.stft.sample_index_to_frame_index(
                    np.array(v.normalized_intervals)),
                shape=v.shape)
            for k, v in ex_array_activity.items()
        }

        assert 'Noise' not in ex_array_activity, ex_array_activity.keys()
        ex_array_activity['Noise'] = pb.array.interval.ones()

        self.last_key = item
        self.last_value = ex_array_activity
        return ex_array_activity


def main(
        # folder='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/extract/77/eval/62000/55',
        # folder='.',
        dia_c7json,
        db_json,
        # config,
        out_folder='audio_gss_mask',
        # use_mask=True,
        # gss=None,
        gss_postfilter=None,  # 'mask_mul.2',
        channel_slice=None,
        sample_rate=16000,
        minimum_segment_length=5000,  # 5000 / 16000 = 0.3125 s
):
    dia_c7json = Path(dia_c7json)
    out_folder = Path(out_folder).absolute()

    if isinstance(minimum_segment_length, str):
        minimum_segment_length = int(minimum_segment_length)

    gss_c7json = out_folder / 'gss_c7.json'
    assert not gss_c7json.exists(), gss_c7json

    # from css.egs.extract.data.sim_libri_css import SimLibriCSS
    # channel_slice = SimLibriCSS.to_slice(channel_slice)

    from tssep.util.utils import str_to_slice
    channel_slice = str_to_slice(channel_slice)

    if ',' in db_json:
        db_json = db_json.split(',')

    db = lazy_dataset.database.JsonDatabase(db_json)
    # db.dataset_names: This contains alias's, which produce duplicates.

    dss = {}

    ds_all_ex = db.get_dataset(sorted(
        set(db.data['datasets'].keys()) - {
            'train_call_mixer6',
            'train_intv_mixer6',
        }
    ))

    gss = get_enhancer_chime6(
        postfilter=gss_postfilter,
    )

    import munch

    stm_lines = []
    for ex in lazy_dataset.from_file(dia_c7json):
        if 'dataset' not in ex:
            ex['dataset'] = ds_all_ex[ex['session_id']]['dataset']
        if 'start_sample' not in ex:
            ex['start_sample'] = round(decimal.Decimal(ex['start_time']) * sample_rate)
        if 'stop_sample' not in ex:
            ex['stop_sample'] = round(decimal.Decimal(ex['end_time']) * sample_rate)
        stm_lines.append(munch.Munch(**ex))

    stm = lazy_dataset.new(stm_lines, immutable_warranty=None)

    stft = pb.transform.STFT(
        size=1024,
        # window_length=1024,
        shift=256,
        # fading=False,  # This will produce some errors at the edges, but it shouldn't have an effect.
        fading=True,
        # axis=-1,
        window='hann',
    )

    activity = ExArrayActivity(stm, stft)

    res = dlp_mpi.collection.UnorderedList()

    # for (dataset, filename), stm in stms.items():
    for segment in dlp_mpi.split_managed(
            stm.sort(
                lambda x: (
                        x.dataset,
                        x.session_id,
                        x.start_time
                ), reverse=True
            ).filter(

                lambda ex: minimum_segment_length <= ex['stop_sample'] - ex['start_sample'],
                lazy=False,
            ),
            progress_bar=True,
            # pbar_prefix=ex['example_id'],
            allow_single_worker=True,
    ):
        dataset = segment.dataset
        filename = segment.session_id
        try:
            ex = dss[dataset][filename]
        except KeyError:
            assert dataset not in dss, (
            dss.keys(), (dataset, filename, stm))
            dss[dataset] = db.get_dataset(dataset)
            ex = dss[dataset][filename]

        ex_array_activity = activity[segment]

        Observation = LazyLoadObservation(
            ex['audio_path']['observation'][channel_slice],
            ex.get('start', 0), ex.get('end', 0),
            stft=stft,
        )

        X_hat = foo(
            Observation,
            ex_array_activity,
            # initialization=initialization,
            begin_index=Observation.stft.sample_index_to_frame_index(segment.start_sample),
            end_index=Observation.stft.sample_index_to_frame_index(segment.stop_sample),
            speaker_id=segment.speaker,
            context_frames=Observation.stft.sample_index_to_frame_index(
                gss.context_samples),
            ex=ex,
            debug=False,
            gss=gss,
        )

        x_hat = Observation.stft.inverse(X_hat)
        del X_hat

        wav_file = f'{out_folder}/{ex["dataset"]}/{ex["example_id"]}_{segment.speaker}_{segment.start_sample}_{segment.stop_sample}.wav'
        Path(wav_file).parent.mkdir(parents=True, exist_ok=True)
        pb.io.dump_audio(x_hat, wav_file, sample_rate=sample_rate)

        segment.audio_path = wav_file

        res.append(segment)

        del x_hat
        del ex

    res = res.gather()

    if dlp_mpi.IS_MASTER:
        res = sorted(res, key=lambda x: x.start_time)  # this ensures, that append order is correct.

        pb.io.dump(res, gss_c7json)
        print(f'Wrote {gss_c7json}')


if __name__ == '__main__':

    from css.slurm import set_memory_limit
    set_memory_limit(10)

    import fire

    import ipdb

    # with ipdb.launch_ipdb_on_exception():
    fire.Fire(main)
