import os
from pathlib import Path
import subprocess
import re


def kaldi(
        src,
        dst,
):
    """

    python -m cbj.transcribe.normalize_transcript kaldi per_utt_whisper_base_en.stm per_utt_whisper_base_en_normalized.stm
    python -m cbj.transcribe.normalize_transcript kaldi per_utt_whisper_base_en_words.stm per_utt_whisper_base_en_words_normalized.stm

    meeteval-wer cpwer -r per_utt_whisper_base_en_normalized.stm -h per_utt_whisper_base_en_words_normalized.stm --average-out -

    python -m cbj.cli.stm diff_v2 per_utt_whisper_base_en_normalized.stm per_utt_whisper_base_en_words_normalized.stm
    python -m cbj.cli.stm diff_v2 per_utt_whisper_base_en_normalized.stm per_utt_whisper_base_en_words_normalized.stm --by-spk

    """
    src = Path(src)
    dst = Path(dst)
    assert src.suffix == '.stm', src
    assert dst.suffix == '.stm', dst
    assert src.exists(), src
    assert not dst.exists(), dst

    from meeteval.io.stm import STM, STMLine

    src = STM.load(src)

    lines = src.lines
    fill = len(str(len(lines)))

    def fix(transcript):
        transcript = transcript.replace('?', '')
        transcript = transcript.replace(',', '')
        transcript = transcript.replace('.', '')
        transcript = transcript.replace('!', '')
        transcript = transcript.replace('-', ' ')
        return transcript.strip()

    text = '\n'.join([
        f'{i:0{fill}} {line.transcript}'
        for i, line in enumerate(lines)
        if line.transcript
    ]) + '\n'

    script = f'{os.environ["KALDI_ROOT"]}/egs/wsj/s5/local/normalize_transcript.pl'

    cp = subprocess.run(
        [script, ''],
        input=text,
        stdout=subprocess.PIPE,
        check=False,
        universal_newlines=True,
    )
    if cp.returncode:
        raise Exception(
            '\n'.join(text.splitlines()[:20])
        )

    text_new = cp.stdout.strip().splitlines()
    # assert len(text_new) == len(lines), (len(text_new), len(lines))
    r = re.compile('([^ ]*) (.*)')
    text_new = {
        int(kv[0]): kv[1]
        for line in text_new
        for kv in [r.fullmatch(line).groups()]
    }

    new_lines = []
    for i, line in enumerate(lines):
        if line.transcript:
            line = line.replace(
                transcript=fix(text_new[i]),
            )
        line = line.replace(
            filename=line.filename.replace('_CH0_', '_'),
        )
        new_lines.append(line)

    # print(new_lines)
    print(f'Write {dst}')
    STM(new_lines).dump(dst)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'kaldi': kaldi
    })
