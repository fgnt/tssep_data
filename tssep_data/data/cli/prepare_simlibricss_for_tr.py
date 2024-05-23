"""



sbatch.py -n 50 -t 12h --mem-per-cpu 10GB --wrap 'srun.py python -m tssep.data.cli.prepare_simlibricss_for_tr /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch.json'
...
Wrote /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json

"""

import functools

import dlp_mpi
import paderbox as pb
import lazy_dataset.database
import numpy as np
from scipy.signal import fftconvolve
from sms_wsj.reverb.reverb_utils import get_rir_start_sample
from pathlib import Path


def work(
    ex, early_rir_samples
):
    audio_data = {}
    audio_data['rir'] = np.array(pb.io.load(ex['audio_path']['rir']))
    audio_data['speaker_source'] = np.array(
        pb.io.load(ex['audio_path']['speaker_source']))

    rir_start_sample = np.array(
        [get_rir_start_sample(h_k) for h_k in audio_data['rir'][:, :, :]])
    rir_stop_sample = rir_start_sample + early_rir_samples

    audio_data['rir'] = audio_data['rir'][..., :, :]

    audio_data['rir_early'] = audio_data['rir'][..., :np.amax(rir_stop_sample)]
    for i in range(audio_data['rir_early'].shape[0]):
        audio_data['rir_early'][i, ..., rir_stop_sample[i]:] = 0

    audio_data['rir_early'] = audio_data['rir_early'][:, ..., :max(rir_stop_sample)]

    audio_data['speaker_reverberation_early'] = fftconvolve(
        audio_data['speaker_source'][:, None, :],
        audio_data['rir_early'],
        axes=-1
    )[..., :-audio_data['rir_early'].shape[-1] + 1]

    # The squeeze is necessary because wav doesn't support 3 dims,
    # hence drop the singleton channel dimension
    audio_data['speaker_reverberation_early'] = np.squeeze(
        audio_data['speaker_reverberation_early'], axis=1)

    # print(audio_data['speaker_reverberation_early_ch0'].shape)

    assert len(ex['audio_path']['observation']) == 1, ex['audio_path']['observation']
    observation, = ex['audio_path']['observation']
    p = Path(observation)
    p = p.with_name(p.name.replace('_', '_speaker_reverberation_early_'))
    # print(p)

    pb.io.dump_audio(audio_data['speaker_reverberation_early'], p,
                     normalize=False, dtype=np.float32)

    # ex['audio_path']['speaker_reverberation_early'] = str(p)
    return ex['example_id'], str(p)


def main(
        json,
        early_rir_samples: int = int(16_000 * 0.05),  # 50 milli seconds
):
    json = Path(json)
    if dlp_mpi.IS_MASTER:
        print(json)

    output = json.with_stem(json.stem + '_speaker_reverberation_early')
    output_exists = dlp_mpi.call_on_root_and_broadcast(output.exists)
    assert not output_exists, output

    json_data = dlp_mpi.call_on_root_and_broadcast(pb.io.load, json)
    db = lazy_dataset.database.DictDatabase(json_data)

    printed = False

    for ds_name in db.dataset_names:
        ds = db.get_dataset(ds_name)

        for example_id, speaker_reverberation_early in dlp_mpi.map_unordered(
                functools.partial(work, early_rir_samples=early_rir_samples),
                ds, allow_single_worker=True, progress_bar=True, pbar_prefix=ds_name):
            json_data['datasets'][ds_name][example_id]['audio_path']['speaker_reverberation_early'] = speaker_reverberation_early

            if not printed:
                printed = True
                print(f'Wrote {speaker_reverberation_early}')

    if dlp_mpi.IS_MASTER:
        pb.io.dump(json_data, output)
        print('Wrote', output)
    dlp_mpi.barrier()


if __name__ == '__main__':
    import fire
    fire.Fire(main)
