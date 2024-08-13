import collections
import decimal
import sys

from meeteval.wer.wer.error_rate import ErrorRate
import paderbox as pb


def example_id_to_subset(k):
    """
    >>> example_id_to_subset('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
    'OV40'
    >>> example_id_to_subset('overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0')
    '0S'
    >>> example_id_to_subset('overlap_ratio_0.0_sil2.9_3.0_session1_actual0.0')
    '0L'
    >>> example_id_to_subset('session0_0L')
    '0L'

    """
    if k.startswith('session'):
        subset = k.split('_')[-1]
    elif k.startswith('overlap_ratio_0.0_sil0'):
        subset = '0S'
    elif k.startswith('overlap_ratio_0.0_sil'):
        subset = '0L'
    else:
        subset = None
        for ratio in [10, 20, 30, 40]:
            if k.startswith(f'overlap_ratio_{ratio}.'):
                subset = f'OV{ratio}'
        assert subset is not None, (k, subset)
    return subset


def example_id_to_dataset(k):
    """
    >>> example_id_to_dataset('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
    'eval'
    >>> example_id_to_dataset('overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0')
    'dev'
    """
    return ['eval', 'dev']['session0' in k]


def example_id_to_session(k):
    """
    >>> example_id_to_session('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
    'session9'
    >>> example_id_to_session('overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0')
    'session0'
    """
    s = 'session'
    return s + k.split(s)[1].split('_')[0]


def main(
        # file='dia_est_v3/gss_c7_words_cpwer_per_reco.json',
        *files,
        session: bool = True,
        per_file=True,
):
    if per_file and len(files) > 1:
        for file in files:
            print(file)
            main(file, session=session, per_file=False)
        return

    print_session = session

    assert files, files

    per_reco = {}
    for file in files:
        per_reco_new = pb.io.load_json(file, parse_float=decimal.Decimal)
        assert set(per_reco_new.keys()).isdisjoint(set(per_reco.keys())), (file, per_reco_new.keys() & per_reco.keys())
        per_reco.update(per_reco_new)

    per_reco = {
        k: ErrorRate.from_dict(v)
        for k, v in per_reco.items()
    }

    per_subset = {
        k: sum([e[1] for e in v])
        for k, v in sorted(pb.utils.iterable.groupby(
            per_reco.items(),
            lambda kv: (example_id_to_dataset(kv[0]), example_id_to_subset(kv[0]), example_id_to_session(kv[0]))
        ).items())
    }

    sessions = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    datasets = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    # dev = {}
    # eval = {}
    # avg = {}
    for (dataset, subset, session), v in per_subset.items():
        sessions[session][subset] += v
        datasets[dataset][subset] += v

    avg = {k: sum([d[k] for d in datasets.values()]) for k in datasets['dev']}

    format_ = lambda x: f'{x.error_rate*100:.2f}' if x is not None else '-'
    # format_ = lambda x: f'{x.errors}'


    data = [
        [
            {
                '': k,
                **{
                    k2: format_(v2)
                    for k2, v2 in v.items()
                },
                'avg': format_(sum(v.values())) if len(v) else None,
            }
            for k, v in d.items()
        ] if not isinstance(d, str) else d
        for d in [
            *([
                sessions,
                '-',
            ] if print_session else []),
            datasets,
            '-',
            {'avg': avg},
        ]
    ]

    data = [e for d in data for e in d]  # flatten

    from tssep.util.table import print_table

    # print(data)
    # print_table(data)
    print_table(data, sep=' & ')

    # data = {
    #     **sessions,
    #     **datasets,
    #     'avg': avg,
    # }
    # data = {
    #     k: {
    #         k2: f'{v2.error_rate*100:.2f}'
    #         for k2, v2 in v.items()
    #     }
    #     for k, v in data.items()
    # }
    #
    # import pandas as pd
    # df = pd.DataFrame(data)
    # print(df.T)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    # main(*sys.argv[1:])
