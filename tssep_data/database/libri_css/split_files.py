import os
import subprocess
from pathlib import Path

import numpy as np

import soundfile
import tqdm
import dlp_mpi

import paderbox as pb
from paderbox.utils.mapping import Dispatcher
from paderbox.utils.iterable import zip
import tssep_data.util.bash


def zip_strict(*args):
    return zip(*args, strict=True)


class Foo:
    _load_mapping = Dispatcher({
        'PCM_16': np.int16,
        'FLOAT': np.float32,
        'DOUBLE': np.float64,
    })

    def channels(self, path):
        with soundfile.SoundFile(os.fspath(path)) as sf:
            return sf.channels

    def load(self, path):
        # from paderbox.utils.mapping import Dispatcher

        with soundfile.SoundFile(os.fspath(path)) as sf:
            dtype = self._load_mapping[sf.subtype]
            obj = sf.read(dtype=dtype)
            return obj.T, sf.samplerate

    def dump(self, obj, path, samplerate):
        dtype_map = Dispatcher({
            np.int16: 'PCM_16',
            np.dtype('int16'): 'PCM_16',
            np.int32: 'PCM_32',
            np.dtype('int32'): 'PCM_32',
            np.float32: 'FLOAT',
            np.dtype('float32'): 'FLOAT',
            np.float64: 'DOUBLE',
            np.dtype('float64'): 'DOUBLE',
        })
        soundfile.write(os.fspath(path), obj.T, samplerate, subtype=dtype_map[obj.dtype])

    def gzip(self, path):
        subprocess.run(['gzip', '-f', '-k', f'{path}'])


def main(
        write=False,  # Weather to write the audio files or just the json.
        json='/mm1/boeddeker/libriCSS/libriCSS_raw.json'
):
    print(tssep_data.util.bash.hostname())

    json = Path(json)
    data = pb.io.load(json)

    foo = Foo()

    def task(example_id):
        example = dataset[example_id]
        speaker_id = list(dict.fromkeys(example['speaker_id']))  # drop duplicates
        speaker_source_indices = example['speaker_source_indices']

        f = Path(example['audio_path']['observation'])

        observation_files = []
        if write:
            y, sample_rate = foo.load(f)
            assert len(y.shape) == 2, y.shape
            channels = y.shape[0]
        else:
            channels = foo.channels(f)

        for i in range(channels):
            f_ch = f.with_stem(f.stem + f'_ch{i}')
            if write:
                y_ = y[i]
                foo.dump(y_, f_ch, sample_rate)
                foo.gzip(f_ch)
            observation_files.append(
                str(f_ch)
                # + '.gz'
            )
        example['audio_path']['observation'] = observation_files

        f = Path(example['audio_path']['speaker_source'])
        if write:
            y, sample_rate = foo.load(f)
            assert len(y.shape) == 2, y.shape
            channels = y.shape[0]
        else:
            channels = foo.channels(f)

        assert len(speaker_source_indices) <= channels, (len(speaker_source_indices), channels, speaker_source_indices)
        # for i, y_ in zip_strict(speaker_source_indices, y):
        # speaker_files = [None] * len(speaker_source_indices)
        speaker_files = []
        for i, spk_id in zip_strict(speaker_source_indices, speaker_id):
        # for i, y_ in zip_strict(speaker_source_indices, y):
            f_ch = f.with_stem(f.stem + f'_{spk_id}')
            if write:
                y_ = y[i]
                foo.dump(y_, f_ch, sample_rate)
                foo.gzip(f_ch)
            speaker_files.append(
                str(f_ch)
                # + '.gz'
            )
            # speaker_files[i] = (
            #     str(f_ch)
            #     # + '.gz'
            # )
        example['audio_path']['speaker_source'] = speaker_files
        return example_id, example

    for dataset_name, dataset in data['datasets'].items():
        for example_id, example in dlp_mpi.map_unordered(task, sorted(dataset.keys()), progress_bar=True):
            dataset[example_id] = example

    if dlp_mpi.IS_MASTER:
        pb.io.dump_json(data, json.with_stem(json.stem + '_chfiles'), sort_keys=False)
        print('Wrote', json.with_stem(json.stem + '_chfiles'))


if __name__ == '__main__':
    import fire
    fire.Fire(main)
