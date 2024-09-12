import functools
import io
import typing
from pathlib import Path
import socket
import os

import cached_property
import numpy as np
import torch

import dlp_mpi
import paderbox as pb
import abc

if typing.TYPE_CHECKING:
    import espnet2.bin.asr_inference


@functools.lru_cache()
def get_device(device=None):
    """
    Returns the device for the current process.

    For the following assume that cuda is available and no device specified:
     - Single process: Use cuda.
     - Multiple MPI process: Use cpu  .

    """
    if (
            device is None
            or device.lower() in ['none', 'null', '']
    ):
        if torch.cuda.is_available():
            if dlp_mpi.SIZE == 1:
                return torch.device('cuda')
            elif dlp_mpi.IS_MASTER:
                print(f'{__file__}: CUDA is available, but MPI_SIZE > 1. Use CPU.')
                return torch.device('cpu')
        return torch.device('cpu')
    else:
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        device = torch.device(device)
        if dlp_mpi.IS_MASTER and dlp_mpi.SIZE > 1 and device.type == 'cuda':
            # Save memory by putting the idling root process on the CPU.
            device = torch.device('cpu')
        return device


class TemplateASR:
    def __init__(self, cwd):
        # For WavLM the cwd must be changed. Hence, allow to change it.
        self.cwd = Path(cwd) if cwd is not None else Path('.').absolute()

    def apply_asr(self, file, start=None, stop=None, channel=None):
        # if start is None and stop is None and channel is None:
        #     return self.transcribe(file)

        kwargs = {}
        if start is not None:
            kwargs['start'] = start
        if stop is not None:
            kwargs['stop'] = stop

        speech = pb.io.load_audio(self.cwd / file, **kwargs)

        if channel is None:
            assert len(speech.shape) == 1, speech.shape
        else:
            assert len(speech.shape) == 2, speech.shape
            speech = speech[channel]

        return self.transcribe(speech)

    @abc.abstractmethod
    def transcribe(self, speech):
        raise NotImplementedError('ToDo', type(self))


class _ESPnetASRBase(TemplateASR):
    speech2text: 'espnet2.bin.asr_inference.Speech2Text'
    model_tag: str

    @cached_property.cached_property
    def model_downloader(self):
        from espnet_model_zoo.downloader import ModelDownloader
        cachedir = os.environ.get('HF_HOME', None)
        print(f'Use {cachedir!r} as cachedir for model_tag={self.model_tag!r}')
        return ModelDownloader(cachedir)

    def get_espnet_model_kwargs(self, model_tag):
        return self.model_downloader.download_and_unpack(model_tag)

    def get_speech2text(self, espnet_model_kwargs):
        from espnet2.bin.asr_inference import Speech2Text

        device = get_device()

        penalty = 0.0
        lm_weight = 0.5

        return Speech2Text(  # Speech2Text.from_pretrained
            **espnet_model_kwargs,
            # Decoding parameters are not included in the model file
            device=device,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=20,
            ctc_weight=0.3,
            lm_weight=lm_weight,
            penalty=penalty,
            nbest=1,
        )

    def transcribe(self, speech):
        try:
            nbests = self.speech2text(speech)
            text, *_ = nbests[0]
            return text
        except RuntimeError as e:
            # from torch.stft:
            # RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (256, 256) at dimension 2 of input [1, 1, 247]
            if 'Padding size should be less' in str(e):
                print(f'Assume nothing was spoken in {speech.shape}, because it is too short for {self.model_tag}.')
                print(f'{e.__module__}.{e.__class__.__qualname__}: {e}')
                return ''
        except Exception as e:
            if e.__class__.__name__ == 'TooShortUttError':
                print(f'Assume nothing was spoken in {speech.shape}, because it is too short for {self.model_tag}.')
                print(f'{e.__module__}.{e.__class__.__qualname__}: {e}')
                return ''

            raise


class ESPnetASRBcast(_ESPnetASRBase):
    """
    This class works only for well written ESPnet models.
    i.e. the model has to support pickle.

    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> file = get_file_path('speech.wav')
    >>> ESPnetASRBcast().apply_asr(file)
    'THE BIRCH CANOE SLID ON THE SMOOTH PLANKS'

    """
    def __init__(
            self,
            model_tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
            cwd=None,
    ):
        super().__init__(cwd=cwd)
        self.model_tag = model_tag

        speech2text = None
        if dlp_mpi.IS_MASTER:
            espnet_model_kwargs = self.get_espnet_model_kwargs(model_tag)
            speech2text = self.get_speech2text(espnet_model_kwargs)
        self.speech2text = dlp_mpi.bcast(speech2text) if dlp_mpi.SIZE > 1 else speech2text


class ESPnetASRPartialBcast(_ESPnetASRBase):
    """

    """
    def __init__(
            self,
            model_tag="popcornell/chime7_task1_asr1_baseline",
            cwd=None,
    ):
        super().__init__(cwd=cwd)
        self.model_tag = model_tag

        espnet_model_kwargs = None
        if dlp_mpi.IS_MASTER:
            espnet_model_kwargs = self.get_espnet_model_kwargs(model_tag)
            # This causes an OOM error, i.e. the memory consumption for the chime7 model is too high.
            # for k in set(espnet_model_kwargs) & {'asr_model_file', 'lm_file'}:
            #     espnet_model_kwargs[k] = io.BytesIO(
            #         Path(espnet_model_kwargs[k]).read_bytes())
        espnet_model_kwargs = dlp_mpi.bcast(espnet_model_kwargs)

        self.speech2text = self.get_speech2text(espnet_model_kwargs)


class ESPnetASR(TemplateASR):
    """
    Deprecated. Use ESPnetASRBcast or ESPnetASRPartialBcast instead.

    This code may have a deadlock.

    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> file = get_file_path('speech.wav')
    >>> ESPnetASR().apply_asr(file)
    'THE BIRCH CANOE SLID ON THE SMOOTH PLANKS'
    >>> # ESPnetASR('espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp').apply_asr(file)  # Very buggy model.
    >>> ESPnetASR('popcornell/chime7_task1_asr1_baseline').apply_asr(file)
    'the birch canoe slid on the smooth planks'
    """

    def __init__(
            self,
            model_tag="Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
            cwd=None,
    ):
        super().__init__(cwd=cwd)
        from espnet_model_zoo.downloader import ModelDownloader, is_url, str_to_hash
        from espnet2.bin.asr_inference import Speech2Text
        self.model_tag = model_tag

        d = ModelDownloader(os.environ.get('HF_HOME', None))

        self.ready = 0

        if model_tag == 'espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp':
            bcast = False
            # The wavlm model is not pickleable.
            #     AttributeError: Can't pickle local object 'UpstreamBase._register_hook_handler.<locals>.generate_hook_handler.<locals>.hook_handler'
            download_dir = f'{d.cachedir}/models--espnet--simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/snapshots/18573cae75cd4ecc952ebbf8c0e22ebbfecb95e0/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1'

            if dlp_mpi.IS_MASTER:
                print(
                    f'Workaround: The model needs the working dir in the espnet download dir (model_tag: {model_tag}).')
                print(f'Hence call os.chdir({download_dir!r}).')
            os.chdir(download_dir)
        elif model_tag == 'popcornell/chime7_task1_asr1_baseline':
            # AttributeError: Can't pickle local object 'UpstreamBase._register_hook_handler.<locals>.generate_hook_handler.<locals>.hook_handler'
            bcast = False
        else:
            bcast = True

        # bcast = False

        if dlp_mpi.IS_MASTER or not bcast:
            if not dlp_mpi.IS_MASTER:
                # Trigger download with main worker.
                # This barrier belongs to the barrier that is later in this code.
                dlp_mpi.barrier()

            if model_tag == 'espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp':
                # ToDO: Get this working for recent ESPnet versions
                # RuntimeError: Error(s) in loading state_dict for ESPnetASRModel:
                # 	Missing key(s) in state_dict: "frontend.upstream.upstream.model.mask_emb", ...
                # 	Unexpected key(s) in state_dict: "frontend.upstream.model.mask_emb",

                if dlp_mpi.IS_MASTER:
                    # Trigger the download.
                    try:
                        espnet_model_kwargs = d.download_and_unpack(model_tag)
                    except FileNotFoundError:
                        # FileNotFoundError: [Errno 2] No such file or directory: '/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/config.yaml'
                        pass

                base = Path(f'{d.cachedir}/models--espnet--simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/snapshots/18573cae75cd4ecc952ebbf8c0e22ebbfecb95e0/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1/',)
                if dlp_mpi.IS_MASTER:
                    if not base.exists():
                        try:
                            d.download_and_unpack(model_tag)
                        except FileNotFoundError:
                            # FileNotFoundError: [Errno 2] No such file or directory: '/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/config.yaml'
                            pass
                    dlp_mpi.barrier()

                espnet_model_kwargs = {
                    'asr_train_config': str(base / 'exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/config.yaml'),
                    'lm_train_config': str(base / 'exp/lm_train_lm_transformer2_en_bpe5000/config.yaml'),
                    'asr_model_file': str(base / 'exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/valid.acc.ave_10best.pth'),
                    'lm_file': str(base / 'exp/lm_train_lm_transformer2_en_bpe5000/4epoch.pth')
                }
            else:
                if bcast:
                    espnet_model_kwargs = d.download_and_unpack(model_tag)
                else:
                    if dlp_mpi.IS_MASTER:
                        espnet_model_kwargs = d.download_and_unpack(model_tag)
                    else:
                        espnet_model_kwargs = None
                    espnet_model_kwargs = dlp_mpi.bcast(espnet_model_kwargs)

            penalty = 0.0
            lm_weight = 0.5

            try:
                device = get_device()

                speech2text = Speech2Text(  # Speech2Text.from_pretrained
                    **espnet_model_kwargs,
                    # Decoding parameters are not included in the model file
                    device=device,
                    maxlenratio=0.0,
                    minlenratio=0.0,
                    beam_size=20,
                    ctc_weight=0.3,
                    lm_weight=lm_weight,
                    penalty=penalty,
                    nbest=1,
                )
            except EOFError as e:
                # Sometimes I get the following Error:
                #     Traceback (most recent call last):
                #       File "/opt/software/pc2/EB-SW/software/Python/3.9.6-GCCcore-11.2.0/lib/python3.9/runpy.py", line 197, in _run_module_as_main
                #         return _run_code(code, main_globals, None,
                #       File "/opt/software/pc2/EB-SW/software/Python/3.9.6-GCCcore-11.2.0/lib/python3.9/runpy.py", line 87, in _run_code
                #         exec(code, run_globals)
                #       File "/scratch/hpc-prf-nt2/cbj/deploy/css/css/egs/extract/per_utt_to_transcription.py", line 118, in <module>
                #         fire.Fire(main)
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/fire/core.py", line 141, in Fire
                #         component_trace = _Fire(component, args, parsed_flag_args, context, name)
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/fire/core.py", line 466, in _Fire
                #         component, remaining_args = _CallAndUpdateTrace(
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/fire/core.py", line 681, in _CallAndUpdateTrace
                #         component = fn(*varargs, **kwargs)
                #       File "/scratch/hpc-prf-nt2/cbj/deploy/css/css/egs/extract/per_utt_to_transcription.py", line 54, in main
                #         apply_asr = ESPnetASR(model_tag=model_tag, cwd=cwd).apply_asr
                #       File "/scratch/hpc-prf-nt2/cbj/deploy/css/css/egs/extract/rttm_to_transcription.py", line 143, in __init__
                #         speech2text = Speech2Text(  # Speech2Text.from_pretrained
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/espnet2/bin/asr_inference.py", line 98, in __init__
                #         asr_model, asr_train_args = task.build_model_from_file(
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/espnet2/tasks/abs_task.py", line 1826, in build_model_from_file
                #         model.load_state_dict(torch.load(model_file, map_location=device))
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/torch/serialization.py", line 795, in load
                #         return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
                #       File "/upb/departments/pc2/groups/hpc-prf-nt2/cbj/software_n2/pythonuserbase_2022_11/lib/python3.9/site-packages/torch/serialization.py", line 1002, in _legacy_load
                #         magic_number = pickle_module.load(f, **pickle_load_args)
                #     EOFError: Ran out of input
                # I don't know why this happens.
                # I observed this, when I started multiple evals, but it is a
                # read operation, to this should be no issue.

                raise Exception(socket.gethostname(), espnet_model_kwargs.get('asr_model_file', '???')) from e

            except RuntimeError as e:
                # Sometimes I get the following Error:
                #     Traceback (most recent call last):
                #       File ".../python/2022_04/anaconda/lib/python3.9/runpy.py", line 197, in _run_module_as_main
                #         return _run_code(code, main_globals, None,
                #       File ".../python/2022_04/anaconda/lib/python3.9/runpy.py", line 87, in _run_code
                #         exec(code, run_globals)
                #       File ".../egs/extract/rttm_to_transcription.py", line 277, in <module>
                #         fire.Fire(main)
                #       File ".../python/2022_04/anaconda/lib/python3.9/site-packages/fire/core.py", line 141, in Fire
                #         component_trace = _Fire(component, args, parsed_flag_args, context, name)
                #       File ".../python/2022_04/anaconda/lib/python3.9/site-packages/fire/core.py", line 466, in _Fire
                #         component, remaining_args = _CallAndUpdateTrace(
                #       File ".../python/2022_04/anaconda/lib/python3.9/site-packages/fire/core.py", line 681, in _CallAndUpdateTrace
                #         component = fn(*varargs, **kwargs)
                #       File ".../egs/extract/rttm_to_transcription.py", line 206, in main
                #         apply_asr = ESPnetASR().apply_asr
                #       File ".../egs/extract/rttm_to_transcription.py", line 109, in __init__
                #         speech2text = Speech2Text(  # Speech2Text.from_pretrained
                #       File ".../espnet/espnet2/bin/asr_inference.py", line 85, in __init__
                #         asr_model, asr_train_args = ASRTask.build_model_from_file(
                #       File ".../espnet/espnet2/tasks/abs_task.py", line 1832, in build_model_from_file
                #         model.load_state_dict(torch.load(model_file, map_location=device))
                #       File ".../python/2022_04/anaconda/lib/python3.9/site-packages/torch/serialization.py", line 713, in load
                #         return _legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)
                #       File ".../python/2022_04/anaconda/lib/python3.9/site-packages/torch/serialization.py", line 938, in _legacy_load
                #         typed_storage._storage._set_from_file(
                #     RuntimeError: unexpected EOF, expected 495801 more bytes. The file might be corrupted.
                # I don't know why. A online search only yield issue reports, where it is reproducible.

                print('#' * 79)
                print('WARNING: Load failed. Retry a second time in a few seconds.')
                print('#' * 79)
                import time
                time.sleep(2)

                device = get_device()
                speech2text = Speech2Text(  # Speech2Text.from_pretrained
                    **espnet_model_kwargs,
                    # Decoding parameters are not included in the model file
                    device=device,
                    maxlenratio=0.0,
                    minlenratio=0.0,
                    beam_size=20,
                    ctc_weight=0.3,
                    lm_weight=lm_weight,
                    penalty=penalty,
                    nbest=1,
                )
        else:
            espnet_model_kwargs = None
            speech2text = None

        if dlp_mpi.IS_MASTER:
            # Trigger download with main worker.
            # This barrier belongs to the barrier that is earlier in this code.
            dlp_mpi.barrier()

        if bcast and dlp_mpi.SIZE > 1:
            self.speech2text = dlp_mpi.bcast(speech2text)
        else:
            self.speech2text = speech2text

    def transcribe(self, speech):
        try:
            nbests = self.speech2text(speech)
            text, *_ = nbests[0]
            return text
        except Exception as e:
            if e.__class__.__name__ == 'TooShortUttError':
                print(f'Assume nothing was spoken in {speech.shape}, because it is too short for {self.model_tag}.')
                print(f'{e.__module__}.{e.__class__.__qualname__}: {e}')
                return ''
            raise


class WhisperASR(TemplateASR):
    """
    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> file = get_file_path('speech.wav')
    >>> WhisperASR().apply_asr(file)
    'THE BIRCH CANOE SLID ON THE SMOOTH PLANKS'
    """

    def __init__(
            self,
            model_tag="base.en",  # 'tiny', 'base', 'base.en', 'small', 'medium', 'large'
            cwd=None,
            # download_root=os.environ.get('HF_HOME', None),  # e.g. /scratch/hpc-prf-nt2/cbj/cache
    ):
        super().__init__(cwd=cwd)

        import whisper  # openai-whisper
        bcast = False

        load_model = functools.partial(whisper.load_model, model_tag, download_root=os.environ.get('HF_HOME', None))

        if bcast:
            self.model = dlp_mpi.call_on_root_and_broadcast(load_model)
        else:
            if dlp_mpi.IS_MASTER:
                # Trigger download
                self.model = load_model()
            dlp_mpi.barrier()
            if not dlp_mpi.IS_MASTER:
                # On worker, use the already downloaded model
                self.model = load_model()

    def transcribe(self, speech: 'np.ndarray'):
        if speech.dtype == np.float64:
            speech = speech.astype(np.float32)
        mode = ['full', 'segments', 'words'][0]

        try:
            result = self.model.transcribe(
                speech,
                word_timestamps=True,
                fp16=False,  # without it raises an annoying warning in each call.
            )

            if mode == 'full':
                result = result['text']
            elif mode == 'segments':
                result = [
                    {
                        'transcript': segment['text'],
                        'begin_time': segment['start'],
                        'end_time': segment['end'],
                    }
                    for segment in result['segments']
                ]
            elif mode == 'words':
                result = [
                    {
                        'transcript': word['word'],
                        'begin_time': word['start'],
                        'end_time': word['end'],
                    }
                    for segment in result['segments']
                    for word in segment['words']
                ]
            else:
                raise ValueError(mode)
            # result = {
            #     'text': '...',
            #     'segments': [
            #         {
            #             'id': 0, 'seek': 0,
            #             'start': 0.24, 'end': 5.24,
            #             'text': '...',
            #             'tokens': [50364, ...],
            #             'temperature': 0.0,
            #             'avg_logprob': -0.15138470649719238,
            #             'compression_ratio': 1.0864197530864197,
            #             'no_speech_prob': 0.002736134687438607,
            #             'words': [
            #                 {
            #                     'word': '...', 'start': 0.24, 'end': 0.74,
            #                     'probability': 0.8676040172576904
            #                 },
            #                 ...
            #             ]
            #         },
            #         ...  # Happens when utterance length > 30 seconds.
            #     ],
            #     'language': 'en'
            # }

            return result
        except Exception as e:
            raise


def _call(func, *args, **kwargs):
    if dlp_mpi.IS_MASTER:
        # Trigger download
        r = func(*args, **kwargs)
    dlp_mpi.barrier()
    if not dlp_mpi.IS_MASTER:
        # On worker, use the already downloaded model
        r = func(*args, **kwargs)
    return r


class NeMoASR(TemplateASR):
    """
    Supports only transcribing full files.

    You may select an nvidia model from
    https://huggingface.co/spaces/hf-audio/open_asr_leaderboard .

    >>> from paderbox.testing.testfile_fetcher import get_file_path
    >>> file = get_file_path('speech.wav')
    >>> NeMoASR().apply_asr(file)
    'the birch canoe slid on the smooth planks'

    >>> NeMoASR('nvidia/stt_en_fastconformer_transducer_xxlarge').apply_asr(file)
    'the birch canoe slid on the smooth planks'

    >>> # NeMoASR('chime-dasr/nemo_baseline_models').apply_asr(file)
    >>> NeMoASR('chime-dasr/nemo_baseline_models/FastConformerXL-RNNT-chime7-GSS-finetuned').apply_asr(file)
    'the birch canoe slid on the smooth planks'

    See https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
    for model tags.

    """
    def __init__(
            self,
            # model_tag='nvidia/stt_en_fastconformer_transducer_xlarge',
            model_tag="nvidia/stt_en_conformer_ctc_large",
            cwd=None,
    ):
        super().__init__(cwd=cwd)
        import logging
        # The following doesn't suppress all logging messages.
        # logging.getLogger('nemo_logger').setLevel(logging.ERROR)

        try:
            import nemo.utils  # pip install nemo_toolkit or pip install git+https://github.com/NVIDIA/NeMo
        except ImportError:
            raise ImportError('pip install nemo_toolkit transformers pytorch_lightning youtokentome webdataset pyannote.audio jiwer datasets lhotse')
        nemo.utils.logging.setLevel(logging.ERROR)
        import nemo.collections.asr as nemo_asr

        if model_tag.count('/') > 1:
            # Hack to add support for file names different to the repo name
            model_tag = model_tag.split('/')
            file = '/'.join(model_tag[2:])
            model_tag = '/'.join(model_tag[:2])

            class ModelTag(str):
                def split(self, sep, maxsplit=-1):
                    r = super().split(sep, maxsplit=maxsplit)
                    if sep == '/':
                        return r + [file]
                    return r

            model_tag = ModelTag(model_tag)

        self.asr_model: nemo_asr.models.asr_model.ASRModel = _call(
            nemo_asr.models.EncDecCTCModelBPE.from_pretrained, model_tag,
            map_location=get_device(),
        )

    def apply_asr(self, file, start=None, stop=None, channel=None):
        assert start is None, (start, stop, channel)
        assert stop is None, (start, stop, channel)
        assert channel is None, (start, stop, channel)
        transcribts = self.asr_model.transcribe([os.fspath(file)], verbose=False)
        try:
            words, = transcribts
            return words
        except Exception:
            try:
                w1, w2 = transcribts
                assert w1 == w2, (w1, w2)
                assert len(w1) == 1 and isinstance(w1[0], str), (w1, w2)
                return w1[0]
            except Exception:
                raise Exception(transcribts, file)

    def transcribe(self, speech: 'np.ndarray'):
        # NeMo requires a file path.
        return self.asr_model.transcribe([speech])


def toy_example():
    """
    >>> toy_example()
    """
    from espnet_model_zoo.downloader import ModelDownloader, is_url, str_to_hash
    from espnet2.bin.asr_inference import Speech2Text
    d = ModelDownloader()
    try:
        d.download_and_unpack('espnet/simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp')
    except FileNotFoundError:
        # FileNotFoundError: [Errno 2] No such file or directory: '/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1/exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/config.yaml'
        pass

    base = Path(f'{d.cachedir}/models--espnet--simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/snapshots/18573cae75cd4ecc952ebbf8c0e22ebbfecb95e0/projects/tir5/users/xuankaic/experiments/espnet/egs2/librispeech/asr1/')
    espnet_model_kwargs = {
        'asr_train_config': str(base / 'exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/config.yaml'),
        'lm_train_config': str(base / 'exp/lm_train_lm_transformer2_en_bpe5000/config.yaml'),
        'asr_model_file': str(base / 'exp/asr_train_asr_conformer7_wavlm_large_raw_en_bpe5000_sp/valid.acc.ave_10best.pth'),
        'lm_file': str(base / 'exp/lm_train_lm_transformer2_en_bpe5000/4epoch.pth'),
    }
    speech2text = Speech2Text(**espnet_model_kwargs)
