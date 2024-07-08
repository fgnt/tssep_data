import operator
import pickle

import numpy as np


class ReduceMaskAsUint8:
    """
    Hack for pickle to store a mask that has values between 0 and 1 as uint8.
    The mask is scaled up to the range 0 to 255 and given to pickle.

    mulaw:
        The load is manipulated to reverse the mulaw compression.
        This results in a loaded float64 array.
        mulaw has a better representation for small numbers.
    8bit:
        The load is manipulated to calculate a truediv of the array and 255.
        This results in a loaded float64 array.

    Note the loaded array is quantized.

    # mulaw compression
    >>> a = np.linspace(0, 1, 10000)
    >>> binary = pickle.dumps(ReduceMaskAsUint8(a, compress='mulaw'))
    >>> len(binary)
    10220
    >>> a_ = pickle.loads(binary)
    >>> np.amax(np.abs(a - a_))  # Max error
    0.01080108010801073
    >>> np.amax(np.abs(a - a_)[a < 0.01])  # Max error for small numbers.
    0.00013308910582645814

    # Simple compression
    >>> binary = pickle.dumps(ReduceMaskAsUint8(a, compress='8bit'))
    >>> len(binary)
    10181
    >>> a_ = pickle.loads(binary)
    >>> np.amax(np.abs(a - a_))  # Max error
    0.0019601960196019563
    >>> np.amax(np.abs(a - a_)[a < 0.01])  # Max error for small numbers.
    0.00195784284310784

    # No compression -> Takes 8 times more memory
    >>> binary = pickle.dumps(ReduceMaskAsUint8(a, compress=False))
    >>> len(binary)
    80160
    >>> a_ = pickle.loads(binary)
    >>> np.amax(np.abs(a - a_))  # Max error
    0.0
    >>> np.amax(np.abs(a - a_)[a < 0.01])  # Max error for small numbers.
    0.0
    """

    def __init__(self, array, compress: 'str | False' = 'mulaw'):
        self.array: np.ndarray = np.asarray(array)
        self.compress = compress

    @staticmethod
    def _error_msg(array):
        return (
            f'min: {array.min()},'
            f'max: {array.max()},\n'
            f'nanquantile(array, [0, 0.5, 1]): {np.nanquantile(array, [0, 0.5, 1])},\n'
            f'infs: {np.isinf(array).sum()},'
            f'nans: {np.isnan(array).sum()},\n'
            f'finites: {np.isfinite(array).sum()},'
            f'elements: {np.size(array)},\n'
            f'array: {array}.'
        )

    def __reduce__(self):
        if not (self.array >= 0).all():
            if np.nanmin(self.array) >= 0:
                print(f'Error in {self.__class__.__name__}. Input of shape {self.array.shape} has only {np.isfinite(self.array).mean() * 100} % finite values. Replace them with 0.')
                self.array = np.nan_to_num(self.array, nan=0, neginf=0, posinf=0)
            else:
                raise AssertionError(self._error_msg(self.array))
        if not (self.array <= 1).all():
            if np.nanmax(self.array) <= 1:
                print(f'Error in {self.__class__.__name__}. Input of shape {self.array.shape} has only {np.isfinite(self.array).mean() * 100} % finite values. Replace them with 0.')
                self.array = np.nan_to_num(self.array, nan=0, neginf=0, posinf=0)
            else:
                raise AssertionError(self._error_msg(self.array))

        if self.compress == 'mulaw':
            mu = 255

            array = np.log(1 + mu * abs(self.array)) / np.log(1 + mu)

            return (
                eval,
                ('((1 + 255)**abs(a / 255) - 1) / 255',
                 {'a': np.rint(array * 255).astype(np.uint8)}),

            )
        elif self.compress == '8bit':
            return operator.truediv, (
                np.rint(self.array * 255).astype(np.uint8),
                255,
            )
        elif self.compress is False:
            return self.array.__reduce__()
        else:
            raise NotADirectoryError(self.compress)


class Audio:
    """
    >>> a = np.linspace(0, 1, 10000)
    >>> binary = pickle.dumps(Audio(a))
    >>> len(binary)
    20194
    >>> a_ = pickle.loads(binary)
    >>> a_.dtype
    dtype('float64')
    >>> np.amax(np.abs(a - a_))  # Max error
    6.062923186855862e-05
    """

    def __init__(self, array):
        self.array = array

    def __reduce__(self):
        import paderbox as pb
        import io
        import soundfile

        b = io.BytesIO()
        pb.io.dump_audio(self.array, b, format='WAV')
        b.seek(0)
        return (eval, (
            'soundfile_read(bytesio)[0]',
            {'soundfile_read': soundfile.read, 'bytesio': b}
        ))
