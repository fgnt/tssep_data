"""
Split the "multichannel" examples into multiple examples, one for each mic.

python -m tssep.data.cli.json_split_multichannel /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json --clear=False --original_id=original_id
"""

import re
import paderbox as pb
import copy
from pathlib import Path


def delete_keys(d, keys):
    for key in keys:
        if key in d:
            del d[key]
    return d


def get_exid_suffix(file):
    """
    >>> get_exid_suffix('CH1.wav')
    'CH1'
    >>> get_exid_suffix('S25_U01.CH4.wav')  # CHiME6 example
    'U01CH4'
    """
    try:
        m = re.search('(?:(U\d\d)\.)?([Cc][Hh]\d+)', file)
        g1, g2 = m.groups()
        return g2 if g1 is None else g1 + g2
    except (TypeError, AttributeError):
        raise RuntimeError(file)


def main(
        input,  # ='/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2.json'
        clear,
        original_id='embedding_id',
):
    """

    Tested for:
     - /scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2.json
     - /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json
    """
    input = Path(input)
    # input
    output = input.with_stem(input.stem + '_ch')
    assert output.exists() == False, output
    # output = '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2_ch.json'

    input = pb.io.load(input)

    def select_channel(multichannel, index):
        if isinstance(multichannel, list):
            return [multichannel[index]]
        elif isinstance(multichannel, str):
            return f'{multichannel}::[:,{index}:{index+1}]'
        else:
            raise RuntimeError(multichannel)

    for dataset_name, examples in input['datasets'].items():
        # if 'train' in dataset_name:
        #     continue

        new_examples = {}
        for example_id, example in examples.items():
            if clear:
                delete_keys(example, ['num_samples', 'offset', 'transcription',
                                      'speaker_id', 'kaldi_transcription'])
            else:
                pass

            for i, o in enumerate(example['audio_path']['observation']):
                new = {**example}
                new['audio_path'] = {'observation': [o]}
                if not clear:
                    for k, v in example['audio_path'].items():
                        if k in ['observation']:
                            pass
                        elif k in ['noise_image']:
                            new['audio_path'][k] = select_channel(v, i)
                        elif k in ['rir']:
                            new['audio_path'][k] = [select_channel(p, i) for p in v]
                        elif k in ['speaker_source']:
                            new['audio_path'][k] = v
                        else:
                            raise KeyError(k)

                new[original_id] = example_id
                k = example_id + '_' + get_exid_suffix(o)
                if k in new_examples:
                    raise RuntimeError(k, o, example['audio_path']['observation'])
                new_examples[k] = new

        len_new_examples = len(new_examples)
        new_examples = dict(new_examples)
        assert len(new_examples) == len_new_examples, (len(new_examples), len_new_examples, list(new_examples.keys())[:10])
        input['datasets'][dataset_name] = new_examples

    # output = '/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2_ch.json'

    pb.io.dump(input, output)
    print('Wrote', output)


if __name__ == '__main__':
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        import fire
        fire.Fire(main)
