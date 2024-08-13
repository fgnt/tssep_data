
def _repr_pretty_(self, p, cycle):
    """

    >>> from IPython.lib.pretty import pprint
    >>> import dataclasses
    >>> Point = dataclasses.make_dataclass('Point', ['x', 'y'])
    >>> PrettyPoint = dataclasses.make_dataclass(
    ...     'Point', ['x', 'y'], namespace={'_repr_pretty_': _repr_pretty_})

    >>> pprint(Point([1] * 5, [2] * 5))
    Point(x=[1, 1, 1, 1, 1], y=[2, 2, 2, 2, 2])
    >>> pprint(PrettyPoint([1] * 5, [2] * 5))
    Point(x=[1, 1, 1, 1, 1], y=[2, 2, 2, 2, 2])

    >>> print('#' * 30)  # show desired max line width
    ##############################
    >>> pprint(Point([1] * 5, [2] * 5), max_width=30)
    Point(x=[1, 1, 1, 1, 1], y=[2, 2, 2, 2, 2])
    >>> pprint(PrettyPoint([1] * 5, [2] * 5), max_width=30)
    Point(
        x=[1, 1, 1, 1, 1],
        y=[2, 2, 2, 2, 2]
    )

    >>> @dataclasses.dataclass
    ... class PrettyPointCls:
    ...     x: int
    ...     y: int
    ...     _repr_pretty_ = _repr_pretty_
    >>> pprint(PrettyPointCls([1] * 5, [2] * 5))
    PrettyPointCls(x=[1, 1, 1, 1, 1], y=[2, 2, 2, 2, 2])
    >>> pprint(PrettyPointCls([1] * 5, [2] * 5), max_width=30)
    PrettyPointCls(
        x=[1, 1, 1, 1, 1],
        y=[2, 2, 2, 2, 2]
    )


    Note: A "mixin" class does not work.

    """
    if cycle:
        p.text(f'{self.__class__.__name__}(...)')
    else:
        txt = f'{self.__class__.__name__}('
        with p.group(4, txt, ''):
            keys = self.__dataclass_fields__.keys()
            for i, k in enumerate(keys):
                if i:
                    p.breakable(sep=' ')
                else:
                    p.breakable(sep='')
                p.text(f'{k}=')
                p.pretty(getattr(self, k))
                if i != len(keys) - 1:
                    p.text(',')
        p.breakable('')
        p.text(')')
