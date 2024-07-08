import re


def str_to_slice(channel_slice):
    """
    >>> str_to_slice('[:1]')
    slice(None, 1, None)
    >>> str_to_slice(slice(1))
    slice(None, 1, None)
    >>> str_to_slice(':')
    slice(None, None, None)
    >>> str_to_slice('[::-1]')
    slice(None, None, -1)
    >>> str_to_slice('[-1:]')
    slice(-1, None, None)
    >>> str_to_slice('[1]')
    Traceback (most recent call last):
    ...
    AssertionError: Expect something that represents a slice, got '[1]'.
    """
    channel_slice_orig = channel_slice
    if isinstance(channel_slice, str):
        class Dummy:
            def __getitem__(self, item):
                return item
        m = re.fullmatch(f'\[?(-?\d*:\d*:?-?\d*)\]?', channel_slice)
        assert m, f'Expect something that represents a slice, got {channel_slice!r}.'
        channel_slice = eval(f'Dummy()[{m.group(1)}]')

    assert isinstance(channel_slice, slice), (type(channel_slice), channel_slice_orig)
    return channel_slice
