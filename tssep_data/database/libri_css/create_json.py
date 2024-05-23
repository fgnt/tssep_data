from pathlib import Path
import numpy as np
import paderbox as pb
from tssep_data.database.libri_css.each_spk_assignment import assignment


def create_raw_database_v2(database_path: Path, overlap_conditions=['0S', '0L', 'OV10', 'OV20', 'OV30', 'OV40']):
    all_res = pb.io.load(database_path / 'all_res.json')
    database = {}

    for cond in overlap_conditions:
        dataset = database[f'{cond}'] = {}
        for ite in (database_path / cond).glob('*'):
            meeting_id = ite.name
            # Get transcriptions from segment files
            transcriptions = []

            # The files 'all_res.json' and '*/*/transcription/segments/seg_*.scp' can be combined to obtain start and
            # end times in samples.
            # The files '*/*/transcription/meeting_info.txt' contains the start and end times in seconds.
            # While samples is the better unit, only the meeting_info.txt file contains the original utterance_id.
            # Hence, use that file. A diff between samples from all_res.json+seg_*.scp and meeting_info.txt with
            # conversion to samples showed no difference between the start and end times.
            # => No quantization issue with the times in meeting_info.txt.

            start = np.amin(all_res[meeting_id])
            end = np.amax(all_res[meeting_id])

            for entry in pb.io.load_csv(ite / 'transcription' / f'meeting_info.txt', dialect='excel-tab'):
                start_time = int(float(entry['start_time']) * 16000) - start
                end_time = int(float(entry['end_time']) * 16000) - start
                speaker_id = entry['speaker']
                transcription = entry['transcription']
                utterance_id = entry['utterance_id']

                transcriptions.append(dict(
                    start_time=start_time,
                    end_time=end_time,
                    speaker_id=speaker_id,
                    transcription=transcription,
                    utterance_id=utterance_id,
                ))

            speaker_source_indices = [
                assignment[(cond, meeting_id)].index(t['speaker_id'])
                for t in transcriptions
            ]
            speaker_source_indices = list(dict.fromkeys(speaker_source_indices))  # drop duplicates

            # Construct example dict
            dataset[f'{meeting_id}'] = {
                # Meeting meta information
                # 'segment_id': segment_id,
                'meeting_id': meeting_id,

                # Data. We have the link to the full raw recording and the
                # start and end times in that recording to save memory
                # 'start': segment_data[0],
                # 'end': segment_data[1],
                'audio_path': {
                    'observation': ite / 'record' / 'raw_recording.wav',
                    'speaker_source': ite / 'clean' / 'each_spk.wav',
                },
                'speaker_source_indices': speaker_source_indices,
                'offset': [
                    t['start_time'] for t in transcriptions
                ],
                'start': start,
                'end': end,
                'source_id': [
                    t['utterance_id'] for t in transcriptions
                ],
                'num_samples': {
                    # 'observation': segment_data[1] - segment_data[0],
                    # 'observation': pb.io.audioread.audio_length(ite / 'record' / 'raw_recording.wav'),
                    'observation': end - start,
                    'original_source': [
                        t['end_time'] - t['start_time']
                        for t in transcriptions
                    ]
                },
                'transcription': [
                    t['transcription'] for t in transcriptions
                ],
                'speaker_id': [
                    t['speaker_id'] for t in transcriptions
                ],
            }

    return {'datasets': database}


def main(folder, output_path):
    print(__file__, folder, output_path)
    folder = Path(folder).expanduser().absolute()
    print(__file__, folder, output_path)
    database = create_raw_database_v2(folder)

    pb.io.dump_json(database, output_path, indent=2, sort_keys=False)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    import fire
    if False:
        import paderbox as pb
        with pb.utils.debug_utils.debug_on(Exception):
            fire.Fire(main)
    else:
        fire.Fire(main)
