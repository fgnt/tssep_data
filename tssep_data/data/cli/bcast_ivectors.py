from pathlib import Path
import paderbox as pb


def main(file, channels, suffix):
    """
    >>> main('/scratch/hpc-prf-nt2/cbj/deploy/css/egs/chime7/data/chime7_v2.json', channels=[0, 1, 2, 3])
    """
    file = Path(file)
    new_file = file.with_stem(file.stem + '_bcast_ch')
    assert new_file.exists() is False, new_file

    data = pb.io.load(file)

    new = {}
    for dataset_name, examples in data['datasets'].items():
        new[dataset_name] = {}
        for example_id, example in examples.items():
            for channel in range(channels):
                suf = suffix.format(channel) # f'_ch{channel}'
                assert suf != suffix, (suf, suffix)
                ex = example.copy()
                ex['example_id'] = example_id + suf
                new[dataset_name][example_id + suf] = ex

    pb.io.dump({'datasets': new}, new_file)
    print('Wrote', new_file)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
    # python -m tssep.data.cli.bcast_ivectors /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/ivector/14/libriCSS_espnet_ivectors.json 7 '_ch{}'
