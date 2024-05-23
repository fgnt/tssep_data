import kaldi_io


class Loader:
    """
    Assumes, that you read ivectors always with the same style (i.e. matrix or vector)

    >>> Loader()
    tssep.data.kaldi.Loader()
    """

    _load_fn = None

    def __repr__(self):
        return f'{type(self).__module__}.{type(self).__name__}()'

    def __call__(self, file_path):
        if self._load_fn is None:
            try:
                _load_fn = lambda p: kaldi_io.read_mat(p)[0]
                vec = _load_fn(file_path)
                self._load_fn = _load_fn
            except kaldi_io.UnknownMatrixHeader:
                _load_fn = lambda p: kaldi_io.read_vec_flt(p)
                vec = _load_fn(file_path)
                self._load_fn = _load_fn
        else:
            vec = self._load_fn(file_path)
        return vec
