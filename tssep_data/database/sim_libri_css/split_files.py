import os
import re
from pathlib import Path
import subprocess

import numpy as np
import soundfile
import tqdm
import dlp_mpi


from paderbox.utils.mapping import Dispatcher
import paderbox as pb

import tssep_data.util.bash


class Foo:
    _load_mapping = Dispatcher({
        'PCM_16': np.int16,
        'FLOAT': np.float32,
        'DOUBLE': np.float64,
    })

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
        # gzip doesn't support random access, hence partial load didn't work.
        # The number of files is the bottleneck, not the size of the files.
        subprocess.run(['gzip', '-f', '-k', f'{path}'])


def main(*folders, check=False):
    for folder in folders:
        print(f'Compress files in: {folder}, {tssep_data.util.bash.hostname()}')

        p = Path(folder)

        foo = Foo()

        if dlp_mpi.IS_MASTER:
            files = sorted(p.glob('*.wav'))
        else:
            files = None

        files = dlp_mpi.bcast(files)
        assert len(files), p

        for f in dlp_mpi.split_managed(files, progress_bar=True):
            if re.match('\d+.wav', f.name):  # Observation
                observation_files = []
                y, sample_rate = foo.load(f)
                assert len(y.shape) == 2, y.shape
                for i, y_ in enumerate(y):
                    f_ch = f.with_stem(f.stem + f'_ch{i}')
                    foo.dump(y_, f_ch, sample_rate)
                    # foo.gzip(f_ch)
                    observation_files.append(f_ch)
                # foo.gzip(f)

                if check:
                    old = pb.io.load(f, list_to='array')
                    new = pb.io.load(observation_files, list_to='array')
                    np.testing.assert_equal(old, new)

            elif re.match('\d+_ch\d+.wav', f.name):  # Channel X of Observation
                pass
            elif re.match('\d+_source\d+.wav', f.name):
                pass
            elif re.match('\d+_noise.wav', f.name):
                pass
            elif re.match('\d+speaker_reverberation_early_ch0.wav', f.name):
                pass
            elif re.match('\d+_rir\d+.wav', f.name):
                pass
            else:
                raise ValueError(f.name)
        from tssep_data.util.bash import hostname
        print(f'Finished compress files in: {folder}, {hostname()}')


if __name__ == '__main__':
    import fire
    fire.Fire(main)
