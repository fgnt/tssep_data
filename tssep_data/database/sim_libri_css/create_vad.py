from pathlib import Path
import lazy_dataset.database
# from tssep_data.data.reader_v2 import PBJsonDSMeta
import numpy as np
import dlp_mpi.collection
import lazy_dataset.database
import paderbox as pb
from tssep_data.util.access import ItemAccessor
from paderbox.array.kernel import np_kernel1d


def to_vad(signal, kernel_size=2049):
    x = np.abs(signal)
    x = x / np.amax(x, axis=-1, keepdims=True)
    x = np_kernel1d(x, kernel_size, kernel=np.amax)
    x = x > 0.05
    return [pb.array.interval.ArrayInterval(v) for v in x]


def to_vad_v2(signal, kernel_size=2049):
    """
    Slightly faster than to_vad.
    """
    x = np.abs(signal)
    x = x / np.amax(x, axis=-1, keepdims=True)
    x = x > 0.05
    x = Kernel1D(kernel_size, kernel=np.any)(x)
    return [pb.array.interval.ArrayInterval(v) for v in x]


def to_vad_v3(signal, kernel_size=2049):
    """
    This function is for large kernel_sizes roughly 100 times faster than to_vad and to_vad_v2.

    >>> for a in [
    ...     np.array([int(s) for s in '000111000111000']),
    ...     np.array([int(s) for s in '000111100111000']),
    ...     np.array([int(s) for s in '001001100111100']),
    ...     np.array([int(s) for s in '010001100111110']),
    ...     np.array([int(s) for s in '100001100111111']),
    ...     np.arange(30)**2,
    ... ]:
    ...     v1 = to_vad(a[None], kernel_size=3)
    ...     v2 = to_vad_v2(a[None], kernel_size=3)
    ...     v3 = to_vad_v3(a[None], kernel_size=3)
    ...     to_str = lambda a: ''.join(['_#'[int(e)] for e in a]) if len(set(np.unique(a)) - {0, 1}) == 0 else a
    ...
    ...     print(to_str(a), 'input')
    ...     print(to_str(v1[0]), 'to_vad')
    ...     print(to_str(v2[0]), 'to_vad_v2')
    ...     print(to_str(v3[0]), 'to_vad_v3')
    ...     print('')
    ___###___###___ input
    __#####_#####__ to_vad
    __#####_#####__ to_vad_v2
    __#####_#####__ to_vad_v3
    <BLANKLINE>
    ___####__###___ input
    __###########__ to_vad
    __###########__ to_vad_v2
    __###########__ to_vad_v3
    <BLANKLINE>
    __#__##__####__ input
    _#############_ to_vad
    _#############_ to_vad_v2
    _#############_ to_vad_v3
    <BLANKLINE>
    _#___##__#####_ input
    ###_########### to_vad
    ###_########### to_vad_v2
    ###_########### to_vad_v3
    <BLANKLINE>
    #____##__###### input
    ##__########### to_vad
    ##__########### to_vad_v2
    ##__########### to_vad_v3
    <BLANKLINE>
    [  0   1   4   9  16  25  36  49  64  81 100 121 144 169 196 225 256 289
     324 361 400 441 484 529 576 625 676 729 784 841] input
    ______######################## to_vad
    ______######################## to_vad_v2
    ______######################## to_vad_v3
    <BLANKLINE>

    """
    if len(signal.shape) > 1:
        return [to_vad_v3(s, kernel_size) for s in signal]

    assert len(signal.shape) == 1, signal.shape

    x = np.abs(signal)
    x = x / np.amax(x, axis=-1, keepdims=True)
    x = x > 0.05

    indices = np.squeeze(np.array(np.nonzero(x)), axis=0)

    assert kernel_size % 2 == 1, (kernel_size, 'is nor odd')
    half_kernel = (kernel_size // 2)

    diff = (indices[1:] - half_kernel) - (indices[:-1] + half_kernel) > 0
    nonzerodiff = np.squeeze(np.array(np.nonzero(diff)), axis=0)
    starts = [max(0, indices[0] - half_kernel), *(indices[nonzerodiff + 1] - half_kernel)]

    ends = [*(indices[nonzerodiff] + half_kernel + 1), min(signal.shape[-1], indices[-1] + (half_kernel + 1))]

    ai = pb.array.interval.zeros(shape=signal.shape)
    ai.intervals = tuple(zip(starts, ends))

    return ai


def v1(
        json='/mm1/boeddeker/db/sim_libri_css.json',
        num_speakers=8,
):

    cfg = SimLibriCSS.get_config({
        'cache_loader': {'use_cache': False},
        'db': {'json_path': json},
        'num_speakers': num_speakers,
    })

    if dlp_mpi.IS_MASTER:
        print(cfg)
    reader = SimLibriCSS.from_config(cfg)

    # cfg = VADSigmoidBCE.get_config()
    # if dlp_mpi.IS_MASTER:
    #     print(VADSigmoidBCE())
    # prepare_target = VADSigmoidBCE().prepare_target

    # cfg = AbsSTFT.get_config()
    # if dlp_mpi.IS_MASTER:
    #     print(cfg)
    # fe = AbsSTFT.from_config(cfg)

    ref_channel = 0
    if dlp_mpi.IS_MASTER:
        print('ref_channel', ref_channel)

    data = dlp_mpi.collection.NestedDict()
    for dataset_name in [
        reader.eval_dataset_name,
        reader.train_dataset_name,
        reader.validate_dataset_name,
    ]:
        ds = reader.__call__(
            dataset_name,
            load_keys=['speaker_reverberation_early'], channel_slice=slice(1),
            segment_num_samples=None, minimum_segment_num_samples=None, mixup=False
        )

        if dlp_mpi.IS_MASTER:
            print(repr(ds), len(ds))

        for ex in dlp_mpi.split_managed(ds, pbar_prefix=dataset_name, allow_single_worker=True):
            vad = to_vad_v3(ex['audio_data']['speaker_reverberation_early'][:, ref_channel])

            data[(ex['dataset'], ex['example_id'])] = vad

    data = data.gather()

    output_dir = f'{Path(json).with_suffix("")}/target_vad/{1}'

    if dlp_mpi.IS_MASTER:
        data = pb.utils.nested.deflatten(data, sep=None)

        assert len(data) > 0, data

        for dataset, d in data.items():
            file = f'{output_dir}/{dataset}.pkl'
            print(f'Write {file}')
            pb.io.dump(d, file, mkdir=True, mkdir_exist_ok=True, mkdir_parents=True, unsafe=True)


def v2(
        json='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json',
        key='["audio_path"]["speaker_reverberation_early"]',
):
    """

    sbatch.py -n 24 -t 6h --mem-per-cpu 4GB --wrap 'srun.py python -m tssep.database.sim_libri_css.create_vad v2 /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json'

    """
    if dlp_mpi.IS_MASTER:
        print(f'Use {key} from {json} for {Path(__file__).stem}')

    assert isinstance(json, str), json

    json = Path(json)
    if dlp_mpi.IS_MASTER:
        assert json.exists(), json

    output_dir = Path(f'{Path(json).with_suffix("")}/target_vad/v2')
    if dlp_mpi.IS_MASTER:
        output_dir.mkdir(parents=True, exist_ok=True)

    tmp = pb.io.load(json) if dlp_mpi.IS_MASTER else None
    db = lazy_dataset.database.DictDatabase(dlp_mpi.bcast(tmp))
    dataset_names = dlp_mpi.bcast(db.dataset_names)

    accessor = ItemAccessor(key)

    for dataset_name in dataset_names:
        data = dlp_mpi.collection.NestedDict()

        ds = db.get_dataset(dataset_name)  # if dlp_mpi.IS_MASTER else None
        for ex in dlp_mpi.split_managed(ds, pbar_prefix=dataset_name, allow_single_worker=True):
            path = accessor(ex)
            audio = pb.io.load_audio(
                path,
                channel=slice(None),  # Force mono to have a channel dim (Some examples are buggy and have only one speaker)
            )
            vad = to_vad_v3(audio)
            data[ex['example_id']] = vad

        data = data.gather()

        if dlp_mpi.IS_MASTER:
            file = f'{output_dir}/{dataset_name}.pkl'
            print(f'Write {file}')
            pb.io.dump(data, file, mkdir=True, mkdir_exist_ok=True, mkdir_parents=True, unsafe=True)

            file = f'{output_dir}/{dataset_name}.rttm'
            pb.array.interval.to_rttm(data, file)
            print(f'Wrote {file}')


if __name__ == '__main__':
    import fire
    fire.Fire({
        'v1': v1,
        'v2': v2,
    })
