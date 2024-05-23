import collections
from pathlib import Path
import lazy_dataset.database
from paderbox.utils.mapping import Dispatcher
import paderbox as pb
import tqdm

LibriCSS2LibriSpeech = Dispatcher({
    'SimLibriCSS-test': 'test_clean',
    'SimLibriCSS-dev': 'dev_clean',
    'SimLibriCSS-train': 'train_960',  # Not sure, if train_100, train_360, train_460 or train_960
})


def mixlog_entry_to_example(
        d,
        ds_librispeech,
        compressed=False,
):
    """
    >>> folder = Path('/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test')
    >>> librispeech_json = '/mm1/boeddeker/db/librispeech.json'
    >>> db_librispeech = lazy_dataset.database.JsonDatabase(librispeech_json)
    >>> pb.utils.pretty.pprint(mixlog_entry_to_example(pb.io.load(folder / 'mixlog.json')[0], db_librispeech.get_dataset('test_clean')))
    ('00',
     {'audio_path': {'observation': ['/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch0.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch1.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch2.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch3.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch4.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch5.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_ch6.wav.gz'],
       'rir': ['/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir1188.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir1580.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir2830.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir61.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir6829.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir7176.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir8224.wav',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_rir8455.wav'],
       'speaker_source': ['/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source1188.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source1580.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source2830.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source61.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source6829.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source7176.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source8224.wav.gz',
        '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_source8455.wav.gz'],
       'noise_image': '/mm1/boeddeker/deploy/jsalt2020_simulate/data/data/SimLibriCSS-test/wav/00_noise.wav'},
      'snr': 0.17201338362256696,
      'sound_decay_time': 11.7865097282824,
      'num_speakers': 8,
      'speaker_id': ['1188', '1580', '2830', '61', '6829', '7176', '8224', '8455'],
      'scenario': [['1188_133604', '1188_133604', '1188_133604'],
       ['1580_141084', '1580_141084', '1580_141084'],
       ['2830_3980', '2830_3980', '2830_3979'],
       ['61_70968', '61_70968', '61_70968'],
       ['6829_68769', '6829_68769', '6829_68769'],
       ['7176_92135', '7176_88083', '7176_92135'],
       ['8224_274381', '8224_274381', '8224_274381'],
       ['8455_210777', '8455_210777', '8455_210777']],
      'speaker_activity': [[(647745, 870625),
        (958211, 1006531),
        (1924433, 2189953)],
       [(233577, 361417), (1409027, 1451107), (1811548, 1843868)],
       [(0, 63360), (1421476, 1464516), (2319080, 2448440)],
       [(802537, 883577), (1453167, 1540767), (1882895, 1929695)],
       [(57398, 97478), (880887, 931127), (2254371, 2320211)],
       [(119800, 234520), (1560053, 1594133), (2120928, 2156688)],
       [(340111, 658271), (1002916, 1421476), (1603661, 1889741)],
       [(70061, 120541), (1553016, 1629496), (2213887, 2302607)]],
      'offset': [[647745, 958211, 1924433],
       [233577, 1409027, 1811548],
       [0, 1421476, 2319080],
       [802537, 1453167, 1882895],
       [57398, 880887, 2254371],
       [119800, 1560053, 2120928],
       [340111, 1002916, 1603661],
       [70061, 1553016, 2213887]],
      'source_id': [['1188-133604-0019', '1188-133604-0013', '1188-133604-0016'],
       ['1580-141084-0041', '1580-141084-0035', '1580-141084-0007'],
       ['2830-3980-0071', '2830-3980-0058', '2830-3979-0003'],
       ['61-70968-0057', '61-70968-0019', '61-70968-0031'],
       ['6829-68769-0038', '6829-68769-0010', '6829-68769-0022'],
       ['7176-92135-0017', '7176-88083-0027', '7176-92135-0030'],
       ['8224-274381-0012', '8224-274381-0005', '8224-274381-0014'],
       ['8455-210777-0020', '8455-210777-0042', '8455-210777-0056']],
      'num_samples': {'observation': 2448440,
       'original_source': [[222880, 48320, 265520],
        [127840, 42080, 32320],
        [63360, 43040, 129360],
        [81040, 87600, 46800],
        [40080, 50240, 65840],
        [114720, 34080, 35760],
        [318160, 418560, 286080],
        [50480, 76480, 88720]]},
      'kaldi_transcription': [['THE LARGE LETTER CONTAINS INDEED ENTIRELY FEEBLE AND ILL DRAWN FIGURES THAT IS MERELY CHILDISH AND FAILING WORK OF AN INFERIOR HAND IT IS NOT CHARACTERISTIC OF GOTHIC OR ANY OTHER SCHOOL',
        'IT MUST REMEMBER BE ONE OR THE OTHER',
        "THIS AT ONCE COMPELS YOU TO UNDERSTAND THAT THE WORK IS TO BE IMAGINATIVE AND DECORATIVE THAT IT REPRESENTS BEAUTIFUL THINGS IN THE CLEAREST WAY BUT NOT UNDER EXISTING CONDITIONS AND THAT IN FACT YOU ARE PRODUCING JEWELER'S WORK RATHER THAN PICTURES"],
       ['NO HARM WOULD HAVE BEEN DONE HAD IT NOT BEEN THAT AS HE PASSED YOUR DOOR HE PERCEIVED THE KEY WHICH HAD BEEN LEFT BY THE CARELESSNESS OF YOUR SERVANT',
        'HE COULD EXAMINE THE PAPERS IN HIS OWN OFFICE',
        'TO MORROW IS THE EXAMINATION'],
       ['WE THINK THAT BY SOME LITTLE WORK OR MERIT WE CAN DISMISS SIN',
        'MOHAMMED ALSO SPEAKS HIGHLY OF CHRIST',
        'THE UNDERTAKING WHICH SEEMED SO ATTRACTIVE WHEN VIEWED AS A LITERARY TASK PROVED A MOST DIFFICULT ONE AND AT TIMES BECAME OPPRESSIVE'],
       ['THESE ESCAPADES ARE NOT FOR OLD GAMEWELL LAD HIS DAY HAS COME TO TWILIGHT',
        'IT IS ENOUGH SAID GEORGE GAMEWELL SHARPLY AND HE TURNED UPON THE CROWD',
        'SILENCE YOU KNAVE CRIED MONTFICHET'],
       ["AND IT RUINS A MAN'S DISPOSITION",
        'WE WISH TO TALK WITH HIM ANSWERED KENNETH TALK',
        'WE HAVE HEARD SOMETHING OF YOUR STORY SAID KENNETH AND ARE INTERESTED IN IT'],
       ['IN THE OLD BADLY MADE PLAY IT WAS FREQUENTLY NECESSARY FOR ONE OF THE CHARACTERS TO TAKE THE AUDIENCE INTO HIS CONFIDENCE',
        'THEN THE LEADER PARTED FROM THE LINE',
        'ENTER LORD ARTHUR FLUFFINOSE'],
       ["MONTROSE WEAK IN CAVALRY HERE LINED HIS TROOPS OF HORSE WITH INFANTRY AND AFTER PUTTING THE ENEMY'S HORSE TO ROUT FELL WITH UNITED FORCE UPON THEIR FOOT WHO WERE ENTIRELY CUT IN PIECES THOUGH WITH THE LOSS OF THE GALLANT LORD GORDON ON THE PART OF THE ROYALISTS",
        'DREADING THE SUPERIOR POWER OF ARGYLE WHO HAVING JOINED HIS VASSALS TO A FORCE LEVIED BY THE PUBLIC WAS APPROACHING WITH A CONSIDERABLE ARMY MONTROSE HASTENED NORTHWARDS IN ORDER TO ROUSE AGAIN THE MARQUIS OF HUNTLEY AND THE GORDONS WHO HAVING BEFORE HASTILY TAKEN ARMS HAD BEEN INSTANTLY SUPPRESSED BY THE COVENANTERS',
        'BESIDES MEMBERS OF PARLIAMENT WHO WERE EXCLUDED MANY OFFICERS UNWILLING TO SERVE UNDER THE NEW GENERALS THREW UP THEIR COMMISSIONS AND UNWARILY FACILITATED THE PROJECT OF PUTTING THE ARMY ENTIRELY INTO THE HANDS OF THAT FACTION'],
       ["OH YES SAID JACK AND I'M NOWHERE",
        'YOU HEAR WHAT SIR FERDINANDO BROWN HAS SAID REPLIED CAPTAIN BATTLEAX',
        "THE PECULIAR CIRCUMSTANCES OF THE COLONY ARE WITHIN YOUR EXCELLENCY'S KNOWLEDGE"]],
      'gender': ['male',
       'female',
       'male',
       'male',
       'female',
       'male',
       'male',
       'male']})

    """
    obs = Path(d['output'])

    tmp_speaker_id_source_id = [
        (input['speaker_id'], Path(input['path']).with_suffix('').name, input)
        for input in d['inputs']
    ]

    speaker_source_examples = {
        spk_id: []
        for spk_id in d['speakers']
    }

    # Note: The order of source_ids will be kind of random. Ideally it should be sorted by utterance start time.
    # source_ids = []

    for spk_id, source_id, input in tmp_speaker_id_source_id:
        speaker_source_examples[spk_id].append((ds_librispeech[source_id], input))

    activity = []
    scenario = []
    source_ids = []
    for spk_id, librispeech_exs in speaker_source_examples.items():
        scenario.append([f"{ex['speaker_id']}_{ex['chapter_id']}" for ex, _ in librispeech_exs])
        activity.append([(input['offset_in_samples'], input['offset_in_samples'] + ex['num_samples']) for ex, input in librispeech_exs])
        source_ids.append([ex['example_id'] for ex, input in librispeech_exs])

    channels = pb.io.audioread.audio_channels(obs)

    ex = {
        'audio_path': {
            'observation': [f'{obs.with_stem(obs.stem + f"_ch{c}")}' + ('.gz' if compressed else '') for c in range(channels)],
            'rir': [
                str(obs.with_name(obs.name.replace('.wav', f'_rir{spk}.wav')))
                for spk in d['speakers']
            ],
            'speaker_source': [
                str(obs.with_name(obs.name.replace('.wav', f'_source{spk}.wav'))) + ('.gz' if compressed else '')
                for spk in d['speakers']
            ],
            'noise_image': str(obs.with_name(obs.name.replace('.wav', f'_noise.wav')))
            #             'original_source': [i['path'] for i in d['inputs']],
        },
        #         'offset': {'original_source': [i['offset_in_samples'] for i in d['inputs']]}
        'snr': d['t60'],
        'sound_decay_time': d['snr'],
        'num_speakers': len(d['speakers']),
        'speaker_id': d['speakers'],
        'scenario': scenario,
        'speaker_activity': activity,
        'offset': [[
            s
            for s, e in a
        ] for a in activity],
        'source_id': source_ids,
        'num_samples': {
            'observation': pb.io.audioread.audio_length(d['output']),
            'original_source': [[
                e - s
                for s, e in a
            ] for a in activity],
        },  # Slow on remote filesystem
        'kaldi_transcription': [
            [
                ex['kaldi_transcription'] if 'kaldi_transcription' in ex else ex['transcription']
                for ex, _ in speaker_source_examples[spk_id]
            ]
            for spk_id in d['speakers']
        ],
        'gender': [
            speaker_source_examples[spk_id][0][0]['gender']
            for spk_id in d['speakers']
        ],
    }
    return obs.with_suffix('').name, ex


def main(
        folder='/mm1/boeddeker/deploy/jsalt2020_simulate/data/data',
        librispeech_json='/mm1/boeddeker/db/librispeech.json',
        output_path='/mm1/boeddeker/db/sim_libri_css.json',
):
    folder = Path(folder).resolve()
    db_librispeech = lazy_dataset.database.JsonDatabase(librispeech_json)
    db_librispeech = db_librispeech

    datasets = {}

    assert folder.exists(), folder
    dataset_folders = list(folder.glob('SimLibriCSS*'))

    assert len(dataset_folders), (folder, folder.glob('*'))

    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.name
        librispeech_dataset = db_librispeech.get_dataset(LibriCSS2LibriSpeech[dataset_name])

        examples = [
            mixlog_entry_to_example(d, librispeech_dataset)
            for d in tqdm.tqdm(pb.io.load(dataset_folder / 'mixlog.json'), desc=dataset_name)
        ]
        len_examples = len(examples)
        examples = dict(examples)
        assert len(examples) == len_examples, (len(examples), len_examples)

        datasets[dataset_name] = examples

    print('Write', output_path)
    pb.io.dump({
        'datasets': datasets,
    }, output_path)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
