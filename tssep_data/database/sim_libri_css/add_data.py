import re

import lazy_dataset.database
import paderbox as pb
import numpy as np
from pathlib import Path
import dlp_mpi

from scipy.signal import fftconvolve
from sms_wsj.reverb.reverb_utils import get_rir_start_sample


def convolve(a: np.ndarray, b: np.ndarray, force_loop=False):
    """
    >>> a = np.arange(10).reshape(1, 2, 5)
    >>> b = np.arange(12).reshape(2, 1, 6)
    >>> convolve(a, b).shape
    (2, 2, 10)
    >>> convolve(a, b, force_loop=True).shape
    (2, 2, 10)
    >>> np.testing.assert_allclose(convolve(a, b), convolve(a, b, force_loop=True))

    """
    if force_loop or (a.nbytes + b.nbytes) >= 100 * 1024**2:
        # Loopy version, when arrays are 100MB+ large
        shape = np.broadcast_shapes(a.shape[:-1], b.shape[:-1])
        a = np.broadcast_to(a, shape + a.shape[-1:])
        b = np.broadcast_to(b, shape + b.shape[-1:])
        out = None
        for index in np.ndindex(shape):
            tmp = fftconvolve(a[index], b[index], axes=-1)
            if out is None:
                out = np.zeros(shape + (a.shape[-1] + b.shape[-1] - 1,),
                               dtype=tmp.dtype)
            out[index] = tmp
        return out
    else:
        return fftconvolve(a, b, axes=-1)


def main(
        json='/mm1/boeddeker/db/sim_libri_css.json',
        early_rir_samples: int = int(16_000 * 0.05),  # 50 milli seconds
        # task: '"write_data" | "write_json"'='write_data',
):
    json = Path(json)

    if dlp_mpi.IS_MASTER:
        print(json)
        json_data = pb.io.load(json)
    else:
        json_data = None

    json_data = dlp_mpi.bcast(json_data)
    printed = False

    from tssep_data.util.slurm import set_memory_limit
    set_memory_limit(11)

    def work(exampleid_ex):
        import gc
        gc.collect()
        example_id, ex = exampleid_ex
        ex['audio_data'] = {}
        ex['audio_data']['rir'] = np.array(pb.io.load(ex['audio_path']['rir']))
        ex['audio_data']['speaker_source'] = np.array(pb.io.load(ex['audio_path']['speaker_source']))

        rir_start_sample = np.array([get_rir_start_sample(h_k) for h_k in ex['audio_data']['rir'][:, :, :]])
        rir_stop_sample = rir_start_sample + early_rir_samples

        # use first channel only
        ex['audio_data']['rir'] = ex['audio_data']['rir'][..., :1, :]

        ex['audio_data']['rir_early'] = ex['audio_data']['rir'][..., :np.amax(rir_stop_sample)]
        for i in range(ex['audio_data']['rir_early'].shape[0]):
            ex['audio_data']['rir_early'][i, ..., rir_stop_sample[i]:] = 0

        ex['audio_data']['rir_early'] = ex['audio_data']['rir_early'][:, ..., :max(rir_stop_sample)]

        ex['audio_data']['speaker_reverberation_early_ch0'] = convolve(
            ex['audio_data']['speaker_source'][:, None, :],
            ex['audio_data']['rir_early'],
            # axes=-1
        )[..., :-ex['audio_data']['rir_early'].shape[-1] + 1]
        del ex['audio_data']['speaker_source']
        del ex['audio_data']['rir_early']

        ex['audio_data']['speaker_reverberation_early_ch0'] = np.squeeze(ex['audio_data']['speaker_reverberation_early_ch0'], axis=1)

        p = Path(ex['audio_path']['noise_image'])
        if '::' in p.name:
            slice_str = p.name.split('::')[-1]
            ch = re.match(r'^\[:,(\d+):\d+\]$', slice_str)
            assert ch, slice_str
            ch = ch.group(1)
            p = p.with_name(
                p.name.split('::')[0].replace('_noise', f'speaker_reverberation_early_ch{ch}'))
        else:
            p = p.with_name(p.name.replace('_noise', 'speaker_reverberation_early_ch0'))
        # print(p)

        pb.io.dump_audio(ex['audio_data']['speaker_reverberation_early_ch0'], p, normalize=False, dtype=np.float32)
        del ex['audio_data']
        nonlocal printed
        if not printed:
            printed = True
            print(f'Wrote {p}')

        return example_id, p

    for ds_name, ds in json_data['datasets'].items():
        # ds = db.get_dataset(ds_name)
        for example_id, path in dlp_mpi.map_unordered(
                work, ds.items(),
                indexable=False,
                progress_bar=True, pbar_prefix=ds_name
        ):
            ds[example_id]['audio_path']['speaker_reverberation_early_ch0'] = str(path)

    if dlp_mpi.IS_MASTER:
        p = json.with_stem(json.stem + '_early')
        pb.io.dump_json(json_data, p)
        print(f'Wrote {p}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
