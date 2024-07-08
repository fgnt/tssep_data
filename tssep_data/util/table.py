import re

class StripANSIEscapeSequences:
    """
    https://stackoverflow.com/a/2188410/5766934

    >>> StripANSIEscapeSequences()('\x1b[1m0.0\x1b[0m')
    '0.0'

    """
    def __init__(self):
        self.r = re.compile(r"""
            \x1b     # literal ESC
            \[       # literal [
            [;\d]*   # zero or more digits or semicolons
            [A-Za-z] # a letter
            """, re.VERBOSE).sub

    def __call__(self, s):
        try:
            return self.r("", s)
        except TypeError:
            raise TypeError(type(s), s)


def print_table(
        data: 'list[dict, str]',
        header=(),
        just: 'str | dict' ='lr',
        sep='  ',
        missing='-',
):
    """

    Args:
        data: list of dict or str
            dict: Keys indicate the column, values the values in the table
            str:
                char (i.e. length 1): row separator
                str (i.e. length != 1): printed as is, might break table layout
        header:
            Optional keys for the header. Will be filled with the keys from the dicts.
            Usecase: Enforce an ordering.
        just:
            Left or right just of the columns.
            Last one will be repeated.
        sep:
            Separator for the columns.
        missing:
            Placeholder for missing values.

    Returns:

    >>> print_table([{'a': 1, 'b': 2}, {'a': 10, 'c': 20}])
    =========
    a   b   c
    =========
    1   2   -
    10  -  20
    =========
    >>> print_table([{'a': 1, 'b': 2}, 'd', 'ef', {'a': 10, 'c': 20}])
    =========
    a   b   c
    =========
    1   2   -
    ddddddddd
    ef
    10  -  20
    =========
    """

    if isinstance(just, dict):
        assert header == (), header
        header = just.keys()
        just = ''.join(just.values())

    # Take header as suggestion for ordering, fill with remaining keys.
    keys = list(dict.fromkeys(list(header) + [
        k for d in data if isinstance(d, dict) for k in d.keys()]))

    data = [{k: str(v) for k, v in d.items()} if isinstance(d, dict) else d
            for d in data]

    strip_ANSI_escape_sequences = StripANSIEscapeSequences()

    def str_width(s):
        return max(map(len, strip_ANSI_escape_sequences(s).splitlines()),
                   default=0)

    def str_height(s: str):
        return len(s.splitlines())

    widths = {
        k: max([str_width(d.get(k, ''))
                for d in data if isinstance(d, dict)] + [str_width(k)])
        for k in keys
    }
    # header_lines = max([
    #     str_height(k)
    #     for k in keys
    # ])

    def just_fn(position, value, width):
        invisible_width = len(value) - len(strip_ANSI_escape_sequences(value))

        if just[min(position, len(just)-1)] == 'l':
            return value.ljust(width + invisible_width, ' ')
        else:
            return value.rjust(width + invisible_width, ' ')

    def format_line(widths, d: '[dict, str]' = None):
        if d is None:
            # if header_lines == 1:
                d = dict(zip(widths.keys(), widths.keys()))
            # else:
            #     lines = []
            #     for i in range(header_lines):
            #         d = dict(zip(
            #             widths.keys(),
            #             [(k.splitlines()[i:i+1] or [''])[0]
            #              for k in widths.keys()],
            #         ))
            #         lines.append(format_line(widths, d))
            #     return '\n'.join(lines)

        if isinstance(d, str):
            if len(d) == 1:
                return d * (
                        sum(widths.values()) + len(sep)
                        * (len(widths.values()) - 1)
                )
            else:
                return d

        num_lines = max([
            str_height(v)
            for v in d.values()
        ])
        if num_lines == 1:
            return sep.join([just_fn(pos, d.get(k, missing), w)
                             for pos, (k, w) in enumerate(widths.items())])
        else:
            lines = []
            for i in range(num_lines):
                lines.append(
                    sep.join([just_fn(
                        pos,
                        (d.get(k, missing).splitlines()[i:i+1] or [''])[0],
                        w
                    ) for pos, (k, w) in enumerate(widths.items())])
                )
        return '\n'.join(lines)

    for i, d in enumerate(data):
        if i % 40 == 0:
            print(format_line(widths, '='))
            print(format_line(widths))
            print(format_line(widths, '='))
        print(format_line(widths, d))
    print(format_line(widths, '='))

