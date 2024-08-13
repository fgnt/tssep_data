"""



mpiexec -np 20 python -m tssep_data.asr seglst c7.json --model_tag=nemo --key=audio_path
sbatch.py -n 200 --time 12h --mem-per-cpu 6G --wrap "srun.py python -m tssep_data.asr seglst c7.json --model_tag=chime7 --key=audio_path"
sbatch.py -n 40 --time 12h --mem-per-cpu 6G --wrap "srun.py python -m tssep_data.asr seglst c7.json --model_tag=nemo --key=audio_path"

"""
import sys
import os
import re
import numbers
import operator
import contextlib

from tssep_data.asr.pretrained import ESPnetASR, WhisperASR, ESPnetASRBcast, ESPnetASRPartialBcast, NeMoASR
import dlp_mpi.collection
import paderbox as pb
from pathlib import Path


if sys.version_info >= (3, 10):
    def zip_strict(*args):
        return zip(*args, strict=True)
else:
    def zip_strict(*args):
        assert len(set(map(len, args))) == 1, args
        return zip(*args)


def alias_to_apply_asr(alias: str):
    alias_mapping = {
        'espnet': "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
        # Properties:
        #  - Trained for LibriSpeech
        #  - Relative old, slow, but solid performance.
        #  - Sensitive to artifacts in the input.
        # Used in
        #  - An Initialization Scheme for Meeting Separation with Spatial Mixture Models
        #  - TS-SEP: Joint diarization and separation conditioned on estimated speaker embeddings

        'wavlm': 'espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp',
        # Properties:
        #  - Trained for LibriSpeech
        #  - Slow, but good performance.
        #  - Robust to artifacts in the input.
        # Used in
        #  - An Initialization Scheme for Meeting Separation with Spatial Mixture Models
        #  - TS-SEP: Joint diarization and separation conditioned on estimated speaker embeddings

        'chime7': 'popcornell/chime7_task1_asr1_baseline',
        # Properties:
        #  - Trained for CHiME-5/6 data
        #
        # 'nemo': 'nvidia/stt_en_fastconformer_transducer_xlarge',  # Buggy produces two transcripts for one file
        'nemo': 'nvidia/stt_en_conformer_ctc_large',
        # Properties:
        #  - Very fast
        #  - Good performance
        #  - Trained for many datasets, see https://huggingface.co/spaces/hf-audio/open_asr_leaderboard

        'nemo_xxl': 'nvidia/stt_en_fastconformer_transducer_xxlarge',

        # 'whisper/base.en': 'whisper/base.en',
        # Properties:
        #  - Trained for many datasets
    }
    if alias in alias_mapping:
        model_tag = alias_mapping[alias]
        if dlp_mpi.IS_MASTER:
            print(f'Replace model_tag {alias} with {model_tag}')
        alias = model_tag

    if alias in [
        "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
    ]:
        return ESPnetASRBcast(alias).apply_asr
    elif alias in [
        'popcornell/chime7_task1_asr1_baseline',
        'espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp',  # It's now fixed and does not need the hack anymore.
    ]:
        return ESPnetASRPartialBcast(alias).apply_asr
    elif alias.startswith('whisper'):
        assert alias.startswith('whisper/'), alias
        return WhisperASR(alias.split('/', maxsplit=2)[1]).apply_asr
    elif alias.startswith('nvidia'):
        return NeMoASR(alias).apply_asr
    elif alias.startswith('espnet'):
        return ESPnetASRPartialBcast(alias).apply_asr
    else:
        raise NotImplementedError(alias)


class LazyFile2Text:
    """
    Call f2t.to_text(file, length) to
    store the file and the length of the file.
    Note: The length is only used for sorting the files to improve the
          scheduling of the files. It is optional and an approximation
          or something proportional to the length is fine.
    That function will returns a transcription placeholder object, that can be
    inserted into a nested structure.

    Once all calls to f2t.to_text(file, length) are done, call
    f2t.estimate_text(estimator=estimator)
    to calculate the transcriptions for each file.

    Finally, call `nested = f2t.insert_text(nested)` to replace the
    transcription placeholders with the actual transcriptions.

    >>> import json, pickle
    >>> memo = []
    >>> f2t = LazyFile2Text()
    >>> nested = {'a': f2t.to_text('abc', length=None)}
    >>> f2t.estimate_text(estimator=lambda file: f'text of {file}')
    Length of the files is unknown. No sort.
    >>> nested = f2t.insert_text(nested)
    >>> nested
    {'a': 'text of abc'}
    """
    def __init__(self):
        self.files2text = {}

    class LazyText:
        def __init__(
                self,
                file,
                outer: 'LazyFile2Text',
                length=None,  # e.g. os.stat(file).st_size, number of samples, ...
        ):
            self.file = file
            self.outer = outer
            self.outer.files2text.setdefault(file, length)

        def __reduce__(self):
            text = self.outer.files2text[self.file]
            assert text is not None, (self.file, self)
            return operator.itemgetter(0), ([text],)

    def to_text(self, file, length=None):
        return self.LazyText(file, self, length=length)

    @classmethod
    def insert_text(cls, data):
        import pickle
        return pickle.loads(pickle.dumps(data))

    def estimate_text(self, estimator: callable, sort=False):
        files2text = dlp_mpi.bcast(self.files2text)

        text_values_set = set(files2text.values())
        if text_values_set - {None} == set():
            print(f'Length of the files is unknown.', 'Reverse sort files by size.' if sort else 'No sort.')
        else:
            assert None not in text_values_set, text_values_set
            print(f'Reverse sort files by provided length indicator, e.g. {list(files2text.values())[:3]}')
            files2text = dict(sorted(files2text.items(), key=lambda kv: kv[1],
                                     reverse=True))
            sort = False

        files = list(files2text.keys())
        if sort:
            # Is this slow?
            d = dlp_mpi.collection.NestedDict()
            for f in files[dlp_mpi.RANK::dlp_mpi.SIZE]:
                d[f] = os.stat(f).st_size
            d = d.gather()
            if dlp_mpi.IS_MASTER:
                files = sorted(d.items(), key=lambda kv: kv[1], reverse=True)
                files = [f for f, _ in files]
            files = dlp_mpi.bcast(files)

        files2text = dlp_mpi.collection.NestedDict()
        for file in dlp_mpi.split_managed(
                files, allow_single_worker=True, progress_bar=True):
            files2text[file] = estimator(file)
        files2text = files2text.gather()
        if dlp_mpi.IS_MASTER:
            self.files2text = files2text


@contextlib.contextmanager
def logctx(pre, post):
    if dlp_mpi.IS_MASTER or dlp_mpi.RANK == 1:
        if pre:
            print(pre)
    yield
    if dlp_mpi.IS_MASTER or dlp_mpi.RANK == 1:
        if post:
            print(post)


def seglst(
        *jsons,
        model_tag='espnet',
        dry=False,
        key='file',
        out: str = None,
):
    """
    Example for a json:

    [{
        "end_time": "43.82",
        "start_time": "40.60",
        "speaker": "P05",
        "session_id": "S02"
        "file": '.../S02.wav::[649600:701120,0]'
    },{
        "end_time": "43.82",
        "start_time": "40.60",
        "speaker": "P05",
        "session_id": "S02"
        "file": '.../S02_P05_004060_004382.wav'
    }]

    important is the "file" entry. Either the complete file is used or
    with "::[start:end, channel]" you can specify the start and end sample.
    The channel argument is optional.
    Note: the "[start:end]" is intrepreted as python code, so it is an array
    slice.

    >>> import tempfile
    >>> import paderbox as pb
    >>> def apply(model_tag='nemo'):
    ...     data = [{'file': pb.testing.testfile_fetcher.get_file_path('speech.wav')}]
    ...     with tempfile.TemporaryDirectory() as tmpdir:
    ...         tmpdir = Path(tmpdir)
    ...         file = tmpdir / 'data.json'
    ...         pb.io.dump(data, file)
    ...         seglst(file, model_tag=model_tag)
    ...         print(pb.io.load(file.with_name(file.stem + f"_words_{re.sub('[^a-zA-Z0-9]', '_', model_tag)}.json")))
    >>> apply()  # doctest: +ELLIPSIS
    Load estimator
    Replace model_tag nemo with nvidia/stt_en_conformer_ctc_large
    Loaded estimator
    Estimate text
    Length of the files is unknown. No sort.
    Estimated text
    Wrote .../data_words_nemo.json
    [{'file': '.../paderbox/cache/speech.wav', 'words': 'the birch canoe slid on the smooth planks'}]
    >>> apply('nemo')  # doctest: +ELLIPSIS
    Load estimator
    Replace model_tag nemo_xxl with nvidia/stt_en_conformer_ctc_large
    Loaded estimator
    Estimate text
    Length of the files is unknown. No sort.
    Estimated text
    Wrote .../data_words_nemo.json
    [{'file': '/home/cbj/python/paderbox/cache/speech.wav', 'words': 'the birch canoe slid on the smooth planks'}]
    >>> apply('espnet')  # doctest: +ELLIPSIS
    Load estimator
    Replace model_tag espnet with Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best
    Use None as cachedir for model_tag='Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best'
    Loaded estimator
    Estimate text
    Length of the files is unknown. No sort.
    Estimated text
    Wrote .../data_words_espnet.json
    [{'file': '/home/cbj/python/paderbox/cache/speech.wav', 'words': 'THE BIRCH CANOE SLID ON THE SMOOTH PLANKS'}]
    >>> apply('whisper/base.en')  # doctest: +ELLIPSIS
    Load estimator
    Loaded estimator
    Estimate text
    Length of the files is unknown. No sort.
    Estimated text
    Wrote .../data_words_whisper_base_en.json
    [{'file': '/home/cbj/python/paderbox/cache/speech.wav', 'words': ' The birch canoe slid on the smooth planks.'}]
    """
    if out is not None:
        out = Path(out)
        assert not out.exists(), out
        assert out.suffix == '.json', out
        assert len(jsons) == 1, f'Assert only one json with out allowed. Gout: {jsons}'

    f2t = LazyFile2Text()
    datas = {}
    if dlp_mpi.IS_MASTER:
        for json in jsons:
            data = pb.io.load(json)
            datas[json] = data

            data_ = data
            if 'end_time' in data[0] and 'start_time' in data[0]:
                data_ = sorted(
                        data_,
                        key=lambda s: float(s['end_time']) - float(s['start_time']),
                        reverse=True,  # Start with long segments.
                )

            for segment in data_:
                # dur = float(segment['end_time']) - float(segment['start_time'])
                # if dur < 80:
                segment['words'] = f2t.to_text(
                    f"{segment[key]}",
                    # length=dur,
                )
                # else:
                #     print('Skip', segment[key], f'because it is too long ({dur} seconds).')
                #     segment['words'] = ''

    text_estimator = lambda x: ''
    if not dry:
        with logctx('Load estimator', 'Loaded estimator'):
            text_estimator = alias_to_apply_asr(model_tag)

    with logctx('Estimate text', 'Estimated text'):
        f2t.estimate_text(text_estimator)

    if dlp_mpi.IS_MASTER:
        datas = f2t.insert_text(datas)
        for json, data in datas.items():
            json = Path(json)
            if out:
                new = out
            else:
                model_tag = re.sub('[^a-zA-Z0-9]', '_', model_tag)
                new = json.with_stem(json.stem + f'_words_{model_tag}')
            pb.io.dump(data, new)
            print(f'Wrote {new}')


def pbjson(
        *jsons,
        model_tag='espnet',
        key='["audio_path"]["segmented_vad"]',  # input key
        okey='["estimated_transcription"]',  # output key
        skey=None,  # sort key
        dry=False,
        out: str = None,
):
    """
    Assumes the input is a database json (see `lazy_dataset.database`).
    Example:
    {
        "datasets": {
           "first_dataset": {
               "example1": {
                   "audio_path": {"segmented_vad": ["file1.wav", "file2.wav"]},
               },
               ...
           },
           ...
    }
    The "key" argument describes, where the audio files are.

    """
    if out is not None:
        out = Path(out)
        assert not out.exists(), out
        assert out.suffix == '.json', out
        assert len(jsons) == 1, f'Assert only one json with out allowed. Gout: {jsons}'

    from cbj.lib.access import ItemAccessor

    input_key = ItemAccessor(key)
    output_key = ItemAccessor(okey)

    if skey is None:
        sort_key = lambda x: None
    else:
        print(f'Use {skey} to reverse sort the files before estimating the transcriptions (Trigger OOM early and better scheduling at the end).')
        sort_key = ItemAccessor(skey)

    def nested_to_text(obj, length):
        if isinstance(obj, str):
            if skey is None:
                assert length is None, (type(length), length)
            else:
                assert isinstance(length, numbers.Number), (type(length), length, obj)
            return f2t.to_text(obj, length)
        elif isinstance(obj, list):
            if not isinstance(length, (tuple, list)):
                length = [length] * len(obj)
            return [
                nested_to_text(o, l)
                for o, l in zip_strict(obj, length)
            ]
        elif isinstance(obj, dict):
            if not isinstance(length, (tuple, list)):
                length = [length] * len(obj)
            return {k: nested_to_text(v, l)
                    for (k, v), l in zip_strict(obj.items(), length)}
        else:
            raise NotImplementedError(type(obj), obj)

    f2t = LazyFile2Text()
    datas = {}
    if dlp_mpi.IS_MASTER:
        for json in jsons:
            db = pb.io.load(json)
            datas[json] = db
            for _, ds in db['datasets'].items():
                for _, ex in ds.items():
                    output_key.set(ex, nested_to_text(
                        input_key(ex),
                        sort_key(ex),
                    ),)
                    # ex['estimated_transcription'] = [
                    #     [
                    #         f2t.to_text(file)
                    #         for file in files
                    #     ]
                    #     for files in ex['audio_path']['segmented_vad']
                    # ]

    text_estimator = lambda x, sort=False: ''
    if not dry:
        with logctx('Load estimator', 'Loaded estimator'):
            text_estimator = alias_to_apply_asr(model_tag)

    with logctx('Estimate text', 'Estimated text'):
        f2t.estimate_text(text_estimator, sort=True)

    if dlp_mpi.IS_MASTER:
        datas = f2t.insert_text(datas)
        for json, data in datas.items():
            json = Path(json)
            if out:
                new = out
            else:
                new = json.with_stem(json.stem + '_words')
            pb.io.dump(data, new)
            print(f'Wrote {new}')


def kaldi(*datasetdirs, model_tag='espnet', dry=False):
    """

    python -m cbj.transcribe.cli kaldi . --dry --model_tag=chime7

    Does this still work? Probably.

    """
    f2t = LazyFile2Text()
    out = {}

    if dlp_mpi.IS_MASTER:
        for dir in datasetdirs:
            text = {}
            out[dir] = text
            wav_scp = (Path(dir) / f'wav.scp').read_text()
            for line in wav_scp.splitlines():
                example_id, file = line.split(' ', maxsplit=1)
                text[example_id] = f2t.to_text(f"{file}")

    text_estimator = lambda x: 'dummy text'
    if not dry:
        text_estimator = alias_to_apply_asr(model_tag)
    f2t.estimate_text(text_estimator)

    if dlp_mpi.IS_MASTER:
        out = f2t.insert_text(out)
        for dir, text in out.items():
            dir = Path(dir)

            file = dir / f'text_hyp_{model_tag}'

            text = ''.join([f'{k} {v}\n' for k, v in text.items()])
            file.write_text(text)
            print(f'Wrote {file}')


def wav_scp(
        wav_scp,
        model_tag='espnet',
        out='{parent}/{text}_hyp_{model_tag}',
        dry=False,
):
    """

    python -m cbj.transcribe.cli wav_scp wav.scp --dry --model_tag=chime7

    wav_scp=wav.scp
    sbatch.py -n 100 --time 12h --mem-per-cpu 6G --wrap "srun.py python -m tssep_dataasr wav_scp ${wav_scp} --model_tag=chime7"

    Does this still work? Probably.
    """
    f2t = LazyFile2Text()
    text = {}
    wav_scp = Path(wav_scp).absolute()
    out = Path(out.format(
        parent=wav_scp.parent,
        model_tag=model_tag,
        text=wav_scp.name.replace("wav.scp", "text"),
    )).absolute()

    if dlp_mpi.IS_MASTER:
        text = {}
        lines = (wav_scp).read_text()
        for line in lines.splitlines():
            example_id, file = line.split(' ', maxsplit=1)
            text[example_id] = f2t.to_text(f"{file}")

    text_estimator = lambda x: 'dummy text'
    if not dry:
        text_estimator = alias_to_apply_asr(model_tag)
    f2t.estimate_text(text_estimator)

    if dlp_mpi.IS_MASTER:
        text = f2t.insert_text(text)

        text = ''.join([f'{k} {v}\n' for k, v in text.items()])
        out.write_text(text)
        print(f'Wrote {out}', file=sys.stderr)


def cli():
    def doctest_fn():
        import doctest
        doctest.testmod()

    import fire
    fire.Fire({
        'seglst': seglst,  # favorite
        'kaldi': kaldi,
        'wav_scp': wav_scp,
        'pbjson': pbjson,
        'doctest': doctest_fn,
    })


if __name__ == '__main__':
    cli()
