"""


salloc.py -n 20 -p cpu,cpu_extra mpirun python -m css.egs.extract.estimate_feature_mean


"""
import numpy as np
import torch

import lazy_dataset
import paderbox as pb
import tqdm.auto


class MeanStd:
    """
    >>> _ = torch.random.manual_seed(0)
    >>> x = torch.randn(10000) * 2 + 3
    >>> torch.mean(x)
    tensor(2.9786)
    >>> torch.std(x)
    tensor(2.0039)

    >>> ms = MeanStd()
    >>> _ = ms.add(torch.sum(x), torch.sum(x**2), len(x))
    >>> ms.mean_std()
    (tensor(2.9786), tensor(2.0038))


    >>> ms = MeanStd()
    >>> for s in [slice(100), slice(100, 1000), slice(1000, None)]:
    ...     ms.add(torch.sum(x[s]), torch.sum(x[s]**2), len(x[s]))
    >>> ms.mean_std()

    """

    def __init__(self):
        self.summation: torch.Tensor = 0
        self.power: torch.Tensor = 0
        self.count = 0

    def add(self, new_sum, new_power, count=1):
        self.summation = self.summation + new_sum
        self.power = self.power + new_power
        self.count += count
        return self

    def mean(self):
        assert self.count > 0, self.count
        return (self.summation / self.count)

    def mean_std(self):
        assert self.count > 0, self.count
        mean = self.summation / self.count
        std = torch.sqrt(self.power / self.count - (mean**2))
        return mean, std

    def std(self):
        assert self.count > 0, self.count
        mean = self.summation / self.count
        return torch.sqrt(self.power / self.count - (mean**2))


def estimate_mean_std(
        reader,
        dataset_name,
        model,
        device='cpu',
        dtype=None,
        allow_thread=True,
        channel_slice=None,
        channel_wise=False,
):
    with torch.no_grad():
        # sig = inspect.signature(reader)

        if False and isinstance(reader, css.egs.extract.data.MMSMSG):
            kwargs = {}
        else:
            kwargs = dict(
                # segment_num_samples=None,
                # minimum_segment_num_samples=None,
                # mixup=False,
                # reverberate=None,
                # channel_slice=channel_slice,
            )

        ds = reader(
            dataset_name,
            load_audio=True,
            pre_load_apply=None,
            load_keys=['observation'],
            **kwargs,
        )

        def to_feature(observation: np.ndarray):
            if len(observation.shape) >= 2 and observation.shape[0] > 2 and observation.nbytes >= 100_000_000:
                # Reduce the memory consumption.
                out = None
                for i, o in enumerate(observation):
                    F: torch.Tensor = to_feature(o)
                    if out is None:
                        out = torch.empty((len(observation),) + F.shape, dtype=F.dtype, device=F.device)
                    out[i, ...] = F
                return out

                # return np.stack([
                #     to_feature(o)
                #     for o in observation
                # ])
            Observation = model.fe.stft(torch.tensor(observation, device=device))
            Feature = model.fe.stft_to_feature(Observation)  # .cpu().numpy()
            return Feature

        model.to(device=device)

        def task(ex):
            feature = to_feature(ex['audio_data']['observation'])
            if len(feature.shape) == 2:
                assert not channel_wise, (channel_wise, feature.shape)
                return (
                    torch.sum(feature, dim=0),
                    torch.sum(feature**2, dim=0),
                    feature.shape[0],
                )
            elif len(feature.shape) == 3:
                if channel_wise:
                    # channel, frames, frequencies
                    return (
                        torch.sum(feature, dim=1),
                        torch.sum(feature ** 2, dim=1),
                        feature.shape[1],
                    )
                else:
                    return (
                        torch.sum(feature, dim=(0, 1)),
                        torch.sum(feature ** 2, dim=(0, 1)),
                        feature.shape[0] * feature.shape[1],
                    )
            else:
                raise ValueError(feature.shape)

        from lazy_dataset.parallel_utils import lazy_parallel_map

        if allow_thread:
            print(f'Estimate statistics: Use lazy_parallel_map with threads. {reader.__class__.__name__} {dataset_name}')

            def pmap(func, iterable: lazy_dataset.Dataset):
                return tqdm.auto.tqdm(iterable.map(func).prefetch(4, 10))
        else:
            print(f'Estimate statistics: Disabled parallel execution. {reader.__class__.__name__} {dataset_name}')

            def pmap(func, iterable: lazy_dataset.Dataset):
                return tqdm.auto.tqdm(iterable.map(func))
                # return tqdm.auto.tqdm(lazy_parallel_map(func, iterable, max_workers=4, buffer_size=10), total=len(iterable))

        meanstd = MeanStd()

        with pb.utils.debug_utils.debug_on(Exception):
            for m, p, count in pmap(task, ds):
                meanstd.add(m, p, count)

        mean, std = meanstd.mean_std()

        if dtype is not None:
            mean = mean.to(dtype=dtype)
            std = std.to(dtype=dtype)

        return mean, std


def estimate_aux_mean_std(
        aux_data, dataset_name,
        model,
        # device='cpu',
        dtype=None,
):
    """
    >>> from paderbox.utils.pretty import pprint
    >>> from lazy_dataset.database import JsonDatabase
    >>> from css.egs.extract.data_ivector import MeetingIVectorReader, SimpleIVectorReader
    >>> # ex = JsonDatabase('/mm1/boeddeker/libriCSS/libriCSS_raw_compressed.json').get_dataset('0L')[0]
    >>> iv_reader = MeetingIVectorReader()
    >>> pprint(estimate_aux_mean_std(iv_reader, '0L', None))
    (array(shape=(100,), dtype=float32), array(shape=(100,), dtype=float32))
    >>> iv_reader = MeetingIVectorReader()
    >>> pprint(estimate_aux_mean_std(iv_reader, '0L', None))
    (array(shape=(100,), dtype=float32), array(shape=(100,), dtype=float32))
    >>> iv_reader = SimpleIVectorReader()
    >>> pprint(estimate_aux_mean_std(iv_reader, 'test_clean', None))
    (array(shape=(100,), dtype=float32), array(shape=(100,), dtype=float32))

    """
    from css.egs.extract.data_ivector import MeetingIVectorReader, SimpleIVectorReader
    from tssep_data.data.data_hooks import SpeakerEmbeddings
    with torch.no_grad():
        if isinstance(aux_data, (MeetingIVectorReader, SimpleIVectorReader)):

            ds = aux_data[dataset_name]
            # Keeping all embeddings inside the memory shouldn't be a problem. At least for dev/eval/test datasets.
            if isinstance(aux_data, MeetingIVectorReader):
                speaker_embeddings = np.concatenate([ex['speaker_embedding'] for ex in ds])
            elif isinstance(aux_data, SimpleIVectorReader):
                speaker_embeddings = np.stack([ex['speaker_embedding'] for ex in ds])
            else:
                raise NotImplementedError(type(aux_data), aux_data)

            mean = np.mean(speaker_embeddings, axis=0, keepdims=False)
            std = np.std(speaker_embeddings, axis=0, keepdims=False)

            return mean, std
        else:
            raise NotImplementedError(type(aux_data), aux_data)


import tssep.train.experiment
import tssep.train.model
import tssep_data.eval.experiment


def main(config, device='cpu'):
    with torch.no_grad():

        config_flat = pb.utils.nested.FlatView(config)

        eg = tssep.train.eexperiment.Experiment.from_config(config_flat['eg'])
        eeg = tssep_data.eval.experiment.EvalExperiment.from_config(config_flat['eeg'])
        model: 'tssep.train.model.Model' = eg.trainer.model

        # if 'eeg' in config_flat and config_flat['eeg.reader'] is not None:
        eval_reader: 'tssep_data.data.LibriCSSRaw' = eeg.reader
        train_reader: 'tssep_data.data.SimLibriCSS' = eg.trainer.model.reader

        for mean, std in [
            estimate_mean_std(reader=eval_reader, dataset_name=eval_reader.eval_dataset_name, model=model, device=device),
            estimate_mean_std(reader=train_reader, dataset_name=train_reader.validate_dataset_name, model=model, device=device),
            estimate_mean_std(reader=train_reader, dataset_name=train_reader.train_dataset_name, model=model, device=device),
        ]:
            print(mean, std)


if __name__ == '__main__':
    main(pb.io.load('/mm1/boeddeker/deploy/css/egs/extract/45/eval/12000/1/config.yaml'))
