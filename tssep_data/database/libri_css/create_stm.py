import decimal
from pathlib import Path
import paderbox as pb
import tssep_data


def main(
    for_release_folder=tssep_data.git_root / 'egs/libri_css/libri_css/exp/data-orig/for_release',
    out=tssep_data.git_root / 'egs/libri_css/data/libri_css_ref.stm',
):
    out = Path(out)

    for_release_folder = Path(for_release_folder)
    files = list(for_release_folder.glob('*/*/transcription/meeting_info.txt'))
    assert files, (f"{for_release_folder}/*/*/transcription/meeting_info.txt", files)

    from meeteval.io.stm import STMLine, STM
    stm = []
    for f in files:
        meeting_info = pb.io.load_csv(f, dialect='excel-tab')

        for entry in meeting_info:
            stm.append(STMLine(
                filename=f.parts[-3],
                channel=1,
                speaker_id=entry['speaker'],
                begin_time=decimal.Decimal(entry['start_time']),
                end_time=decimal.Decimal(entry['end_time']),
                # utterance_id=entry['utterance_id'],
                transcript=entry['transcription'],
            ))
    print('Loaded', len(stm), 'lines')
    print('Example:', stm[0])
    STM(stm).dump(out)
    print('Wrote', out)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
