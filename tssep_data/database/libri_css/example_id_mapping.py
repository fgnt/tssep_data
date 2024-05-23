import paderbox as pb
import re


libri_css_folder_id_to_kaldi_id = pb.utils.mapping.Dispatcher({
    'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0': 'session0_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session1_actual0.0': 'session1_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session2_actual0.0': 'session2_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session3_actual0.0': 'session3_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session4_actual0.0': 'session4_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session5_actual0.0': 'session5_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session6_actual0.0': 'session6_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session7_actual0.0': 'session7_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session8_actual0.0': 'session8_CH0_0L',
    'overlap_ratio_0.0_sil2.9_3.0_session9_actual0.0': 'session9_CH0_0L',
    'overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0': 'session0_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session1_actual0.0': 'session1_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session2_actual0.0': 'session2_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0': 'session3_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session4_actual0.0': 'session4_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session5_actual0.0': 'session5_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session6_actual0.0': 'session6_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session7_actual0.0': 'session7_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session8_actual0.0': 'session8_CH0_0S',
    'overlap_ratio_0.0_sil0.1_0.5_session9_actual0.0': 'session9_CH0_0S',
    'overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1': 'session0_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2': 'session1_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0': 'session2_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session3_actual10.1': 'session3_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session4_actual10.0': 'session4_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session5_actual9.9': 'session5_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session6_actual9.9': 'session6_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session7_actual10.1': 'session7_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session8_actual10.0': 'session8_CH0_OV10',
    'overlap_ratio_10.0_sil0.1_1.0_session9_actual10.0': 'session9_CH0_OV10',
    'overlap_ratio_20.0_sil0.1_1.0_session0_actual20.8': 'session0_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session1_actual20.5': 'session1_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session2_actual21.1': 'session2_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session3_actual20.0': 'session3_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session4_actual20.0': 'session4_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session5_actual19.6': 'session5_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session6_actual20.0': 'session6_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session7_actual20.1': 'session7_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session8_actual19.8': 'session8_CH0_OV20',
    'overlap_ratio_20.0_sil0.1_1.0_session9_actual20.7': 'session9_CH0_OV20',
    'overlap_ratio_30.0_sil0.1_1.0_session0_actual29.7': 'session0_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session1_actual30.4': 'session1_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session2_actual29.6': 'session2_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session3_actual30.2': 'session3_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session4_actual29.8': 'session4_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session5_actual29.7': 'session5_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session6_actual30.1': 'session6_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session7_actual30.2': 'session7_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session8_actual29.7': 'session8_CH0_OV30',
    'overlap_ratio_30.0_sil0.1_1.0_session9_actual29.8': 'session9_CH0_OV30',
    'overlap_ratio_40.0_sil0.1_1.0_session0_actual39.5': 'session0_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session1_actual39.7': 'session1_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session2_actual41.2': 'session2_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session3_actual40.2': 'session3_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session4_actual39.0': 'session4_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session5_actual42.0': 'session5_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session6_actual39.9': 'session6_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session7_actual40.5': 'session7_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session8_actual40.5': 'session8_CH0_OV40',
    'overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9': 'session9_CH0_OV40',
})

libri_css_kaldi_id_to_folder_id = pb.utils.mapping.Dispatcher({
    'session0_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0',
    'session1_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session1_actual0.0',
    'session2_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session2_actual0.0',
    'session3_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session3_actual0.0',
    'session4_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session4_actual0.0',
    'session5_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session5_actual0.0',
    'session6_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session6_actual0.0',
    'session7_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session7_actual0.0',
    'session8_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session8_actual0.0',
    'session9_CH0_0L': 'overlap_ratio_0.0_sil2.9_3.0_session9_actual0.0',
    'session0_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session0_actual0.0',
    'session1_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session1_actual0.0',
    'session2_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session2_actual0.0',
    'session3_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session3_actual0.0',
    'session4_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session4_actual0.0',
    'session5_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session5_actual0.0',
    'session6_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session6_actual0.0',
    'session7_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session7_actual0.0',
    'session8_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session8_actual0.0',
    'session9_CH0_0S': 'overlap_ratio_0.0_sil0.1_0.5_session9_actual0.0',
    'session0_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session0_actual10.1',
    'session1_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session1_actual10.2',
    'session2_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session2_actual10.0',
    'session3_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session3_actual10.1',
    'session4_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session4_actual10.0',
    'session5_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session5_actual9.9',
    'session6_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session6_actual9.9',
    'session7_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session7_actual10.1',
    'session8_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session8_actual10.0',
    'session9_CH0_OV10': 'overlap_ratio_10.0_sil0.1_1.0_session9_actual10.0',
    'session0_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session0_actual20.8',
    'session1_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session1_actual20.5',
    'session2_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session2_actual21.1',
    'session3_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session3_actual20.0',
    'session4_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session4_actual20.0',
    'session5_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session5_actual19.6',
    'session6_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session6_actual20.0',
    'session7_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session7_actual20.1',
    'session8_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session8_actual19.8',
    'session9_CH0_OV20': 'overlap_ratio_20.0_sil0.1_1.0_session9_actual20.7',
    'session0_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session0_actual29.7',
    'session1_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session1_actual30.4',
    'session2_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session2_actual29.6',
    'session3_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session3_actual30.2',
    'session4_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session4_actual29.8',
    'session5_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session5_actual29.7',
    'session6_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session6_actual30.1',
    'session7_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session7_actual30.2',
    'session8_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session8_actual29.7',
    'session9_CH0_OV30': 'overlap_ratio_30.0_sil0.1_1.0_session9_actual29.8',
    'session0_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session0_actual39.5',
    'session1_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session1_actual39.7',
    'session2_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session2_actual41.2',
    'session3_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session3_actual40.2',
    'session4_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session4_actual39.0',
    'session5_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session5_actual42.0',
    'session6_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session6_actual39.9',
    'session7_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session7_actual40.5',
    'session8_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session8_actual40.5',
    'session9_CH0_OV40': 'overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9',
})


class LibriCSSIDMapper:
    @staticmethod
    def to_kaldi(id_):
        """
        >>> LibriCSSIDMapper().to_kaldi('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
        'session9_CH0_OV40'
        >>> LibriCSSIDMapper().to_kaldi('session9_CH0_OV40')
        'session9_CH0_OV40'
        """
        try:
            return libri_css_folder_id_to_kaldi_id[id_]
        except KeyError:
            if id_ in libri_css_kaldi_id_to_folder_id:
                return id_
            else:
                raise

    @staticmethod
    def to_folder(id_, default=None):
        """
        >>> LibriCSSIDMapper().to_folder('session9_CH0_OV40')
        'overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9'
        >>> LibriCSSIDMapper().to_folder('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
        'overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9'
        """
        try:
            return libri_css_kaldi_id_to_folder_id[id_]
        except KeyError:
            if id_ in libri_css_folder_id_to_kaldi_id:
                return id_
            else:
                if default is None:
                    raise
                else:
                    return default

    @classmethod
    def to_dataset(cls, id_):
        """

        >>> LibriCSSIDMapper().to_dataset('session9_CH0_OV40')
        'OV40'
        >>> LibriCSSIDMapper().to_dataset('overlap_ratio_40.0_sil0.1_1.0_session9_actual39.9')
        'OV40'
        """
        return cls.to_kaldi(id_).split('_')[-1]


class LibriCSSIDMapper_v2:
    def __init__(
            self,
            mode='rec',  # rec[ording], utt[erance]
            # mode='utterance_id',
    ):
        self.mode = mode
        if mode == 'rec':
            self.r_folder = re.compile(r'overlap_ratio_(.*)_session(.*)_actual.*')
            self.r_kaldi = re.compile(r'session(\d+)_CH0_(OV\d0|0[LS])')
            self.r_short = re.compile(r'session(\d+)_(OV\d0|0[LS])')
            self.r_desh = re.compile(r'(OV\d0|0[LS])_session(\d+)')
        elif mode == 'rec2utt':
            self.r_folder = re.compile(r'\d+_overlap_ratio_(.*)_session(.*)_actual.*')
            self.r_kaldi = re.compile(r'\d+_session(\d+)_CH0_(OV\d0|0[LS])_\d+_\d+')
            self.r_short = re.compile(r'\d+_session(\d+)_(OV\d0|0[LS])_\d+_\d+')
        else:
            raise NotImplementedError(mode)

        self.verbose2subset = pb.utils.mapping.Dispatcher({
            '0.0_sil2.9_3.0': '0L',
            '0.0_sil0.1_0.5': '0S',
            '10.0_sil0.1_1.0': 'OV10',
            '20.0_sil0.1_1.0': 'OV20',
            '30.0_sil0.1_1.0': 'OV30',
            '40.0_sil0.1_1.0': 'OV40',
        })

    def to_kaldi(self, id_, verbose=False):
        """
        >>> LibriCSSIDMapper_v2().to_kaldi('overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0')
        'session0_CH0_0L'
        >>> LibriCSSIDMapper_v2().to_kaldi('session0_CH0_0L')
        'session0_CH0_0L'
        >>> LibriCSSIDMapper_v2().to_kaldi('session0_0L')
        'session0_CH0_0L'
        >>> LibriCSSIDMapper_v2('rec2utt').to_kaldi('1089_session1_CH0_0S_003830_004658')
        'session1_CH0_0S'
        >>> LibriCSSIDMapper_v2().to_kaldi('0L_session1')  # https://github.com/desh2608/diarizer
        'session1_CH0_0L'
        """
        m = self.r_folder.fullmatch(id_)
        if m:
            verbose_, session = m.groups()
            subset = self.verbose2subset[verbose_]
            if verbose:
                verbose = 'r_folder'
        else:
            m = self.r_short.fullmatch(id_)
            if m:
                session, subset = m.groups()
                if verbose:
                    verbose = 'r_short'
            else:
                m = self.r_kaldi.fullmatch(id_)
                if m:
                    session, subset = m.groups()
                    if verbose:
                        verbose = 'r_kaldi'
                else:
                    m = self.r_desh.fullmatch(id_)
                    if m:
                        subset, session = m.groups()
                        if verbose:
                            verbose = 'r_desh'
                    else:
                        raise ValueError(id_)
        if verbose:
            return f'session{session}_CH0_{subset}', verbose
        else:
            return f'session{session}_CH0_{subset}'

    def to_short(self, id_):
        """
        >>> LibriCSSIDMapper_v2().to_short('overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0')
        'session0_0L'
        >>> LibriCSSIDMapper_v2().to_short('session0_CH0_0L')
        'session0_0L'
        >>> LibriCSSIDMapper_v2().to_short('session0_0L')
        'session0_0L'
        """
        return self.to_kaldi(id_).replace('_CH0', '')

    def to_folder(self, id_):
        """
        >>> LibriCSSIDMapper_v2().to_folder('overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0')
        'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0'
        >>> LibriCSSIDMapper_v2().to_folder('session0_CH0_0L')
        'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0'
        >>> LibriCSSIDMapper_v2().to_folder('session0_0L')
        'overlap_ratio_0.0_sil2.9_3.0_session0_actual0.0'
        """
        return libri_css_kaldi_id_to_folder_id[self.to_kaldi(id_)]


class ExampleIDtoKaldi:
    def __init__(self):
        self.r = re.compile(r'overlap_ratio_(.*)_session(.*)_actual.*')
        self.r_kaldi = re.compile(r'session(.*)_CH.*_(OV\d0|0[LS])')

        self.verbose2short = pb.utils.mapping.Dispatcher({
            '0.0_sil2.9_3.0': '0L',
            '0.0_sil0.1_0.5': '0S',
            '10.0_sil0.1_1.0': 'OV10',
            '20.0_sil0.1_1.0': 'OV20',
            '30.0_sil0.1_1.0': 'OV30',
            '40.0_sil0.1_1.0': 'OV40',
        })

    def __getitem__(self, item):
        """
        >>> mapper = ExampleIDtoKaldi()
        >>> mapper['overlap_ratio_30.0_sil0.1_1.0_session1_actual30.4']
        'session1_CH0_OV30'
        >>> mapper['session1_CH0_OV30']
        'session1_CH0_OV30'

        """

        m = self.r.match(item)

        if not m:
            if self.r_kaldi.match(item):
                return item
            raise Exception(item, m)

        verbose, session = m.groups()
        short = self.verbose2short[verbose]

        return f'session{session}_CH0_{short}'


def calculate():
    from lazy_dataset.database import JsonDatabase
    from paderbox.utils.pretty import pprint
    json = 'ToDO/path/to/libriCSS_raw_compressed.json'
    db = JsonDatabase(json)

    mapper = ExampleIDtoKaldi()

    ex_to_kaldi = {}
    kaldi_to_ex = {}

    for ds_name in db.dataset_names:
        ds = db.get_dataset(ds_name)

        for ex in ds:
            example_id = ex['example_id']
            kaldi_id = mapper[example_id]

            ex_to_kaldi[example_id] = kaldi_id
            kaldi_to_ex[kaldi_id] = example_id

    pprint(ex_to_kaldi)
    pprint(kaldi_to_ex)

    assert libri_css_folder_id_to_kaldi_id == ex_to_kaldi
    assert libri_css_kaldi_id_to_folder_id == kaldi_to_ex


if __name__ == '__main__':
    calculate()
