import io
from pathlib import Path
import decimal
import paderbox as pb


def load_json(json_file):
    """
    Use decimal for floats to keep the number as it is.
    Additionally, this prevents issues with the precision of floats.

    """
    return pb.io.load_json(json_file, parse_float=decimal.Decimal)


def dump_json(data, json_file, indent=2):
    """
    Support for decimal.Decimal

    >>> print(dumps_json({'a': decimal.Decimal('100000000000.0173455555'),
    ...                    'b': 100000000000.0173455555}))
    {
      "a": 100000000000.0173455555,
      "b": 100000000000.01735
    }
    >>> import json
    >>> print(json.dumps({'a': decimal.Decimal('100000000000.0173455555'),
    ...                    'b': 100000000000.0173455555}))
    Traceback (most recent call last):
    ...
    TypeError: Object of type Decimal is not JSON serializable
    >>> print(json.dumps({'b': 100000000000.0173455555}))
    {"b": 100000000000.01735}


    """
    import simplejson
    if isinstance(json_file, (str, Path)):
        with open(json_file, 'w') as f:
            return dump_json(data, f, indent=indent)

    return simplejson.dump(data, json_file, indent=indent)


def dumps_json(data, indent=2):
    fd = io.StringIO()
    dump_json(data, fd, indent=indent)
    return fd.getvalue()
