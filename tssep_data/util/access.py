"""

Licence MIT
Origin: Communications Department, Paderborn University, Germany

"""

import re
from paderbox.utils.mapping import DispatchError


class ItemAccessor:
    """ obj[item] is the same as GetItemGetter(item)(obj), where
    GetItemGetter supports more complex item specifications.

    Some examples:
     - obj[:2] == ItemAccessor[:2](obj)
     - obj[:2] == ItemAccessor(':2')(obj)
     - obj[:2][:2] == ItemAccessor('[:2][:2]')(obj)

    The main purpose is to specify the item as a string.
    Further, the nested access is usually tricky to read and write, e.g.:
        one = operator.itemgetter(1)
        special = operator.itemgetter('Special number')
        getter = lambda x: special(one(x)))
    from https://stackoverflow.com/a/62797048/5766934 vs
        getter = ItemAccessor('[1][Special number]')

    """

    def __class_getitem__(cls, item):
        return cls(item)

    def __getitem__(self, item):
        return self.__class__(self.items + self.to_slices(item))

    def __init__(self, item):
        self.items = self.to_slices(item)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.items})'

    def __call__(self, obj):
        """
        >>> obj = [1, 2, 3]
        >>> ItemAccessor[:2](obj)
        [1, 2]
        >>> ItemAccessor[':2'](obj), ItemAccessor['[:2]'](obj)
        ([1, 2], [1, 2])
        >>> ItemAccessor['[1:][:1]']
        ItemAccessor((slice(1, None, None), slice(None, 1, None)))
        >>> ItemAccessor((slice(1, None, None), slice(None, 1, None)))
        ItemAccessor((slice(1, None, None), slice(None, 1, None)))
        >>> ItemAccessor['[1:][:1]'](obj)
        [2]

        >>> obj = {'a': {'b': {'c': 1}}}
        >>> ItemAccessor['a']['b']['c'](obj)
        1
        >>> ItemAccessor("['a']['b']['c']")(obj)
        1
        """
        for item in self.items:
            try:
                obj = obj[item]
            except KeyError:
                raise DispatchError(item, obj.keys()) from None
        return obj

    def set(self, obj, value):
        for item in self.items[:-1]:
            obj = obj[item]
        obj[self.items[-1]] = value

    @classmethod
    def literal_eval(cls, item):
        """
        Assume item represents an item specification, e.g.: '["abc"][:2]'
        Returns a tuple with the items, e.g.: ('abc', slice(2))

        This functions has some fallbacks for lazy users like me, that write
        e.g. 'abs' instead of '["abc"]' and shell users, where the shell
        eats the quotes.
        Note: If an item specification should for example be the string '1',
        then you have to use quotes, e.g. '["1"]', otherwise it will be
        converted to an integer.

        >>> ItemAccessor.literal_eval('[1]')
        (1,)
        >>> ItemAccessor.literal_eval('[1]["as"][asd]')
        (1, 'as', 'asd')
        >>> ItemAccessor.literal_eval('[1]["as"][asd][:3][...][[1, 2]]')
        (1, 'as', 'asd', slice(None, 3, None), Ellipsis, [1, 2])
        >>> ItemAccessor.literal_eval('a')
        ('a',)
        >>> ItemAccessor.literal_eval('1')
        (1,)
        >>> ItemAccessor.literal_eval('::3')
        (slice(None, None, 3),)
        """
        import ast

        def convert(node):
            if isinstance(node, ast.Subscript):
                return convert(node.value) + convert(node.slice)
            elif isinstance(node, ast.Name):
                # Variables are not allowed. Probably the shell removed the
                # quotes, hence assume it is a string.
                return node.id,
            elif isinstance(node, ast.Slice):
                def fn(value):
                    return None if value is None else ast.literal_eval(value)
                return slice(*map(fn, [node.lower, node.upper, node.step])),
            else:
                return ast.literal_eval(node),

        if isinstance(item, str):
            if not item.startswith('['):
                item = '[' + item + ']'
            ret = convert(ast.parse('dummy' + item, mode='eval').body)
            assert ret[0] == 'dummy', (ret, item)
            return ret[1:]
        else:
            assert isinstance(item, (list, tuple)), item
            return item

    @classmethod
    def to_slices_v2(cls, item):
        """
        >>> ItemAccessor.to_slices_v2('[:1]')
        [slice(None, 1, None)]
        >>> ItemAccessor.to_slices_v2('[1][a]["b"][...]')
        [1, 'a', 'b', Ellipsis]
        >>> ItemAccessor.to_slices_v2('c')
        ['c']
        >>> ItemAccessor.to_slices_v2('"c"')
        ['c']
        """

        if isinstance(item, str):
            if '[' == item[0]:
                assert ']' == item[-1], item
                # item = item[1:-1]
                # ms = item.split('][')
                r = re.compile(r'(?:\[([^\[\]]+)\])+')
                assert r.fullmatch(item), (item, 'Expected something like "[...][...]".')
                # ms = r.findall(item)
                item = item[1:-1]
                ms = item.split('][')

                # ms = re.findall(r'\[([^\[]+)\]+', item)
                # return ms
                assert ms, (item, 'Expected something like "[...][...]".')
            else:
                assert '[' not in item, item
                ms = [item]
            ret = []
            for m in ms:
                if ':' in m:
                    m_ = re.fullmatch(r'(\d*):(\d*)(?::(\d*))?', m)
                    if m_:
                        start, stop, step = m_.groups()
                        start = int(start) if start else None
                        stop = int(stop) if stop else None
                        step = int(step) if step else None
                        ret.append(slice(start, stop, step))
                    continue

                import ast
                try:
                    ret.append(ast.literal_eval(m))
                except ValueError:
                    ret.append(m)
        return ret

    @classmethod
    def to_slices(cls, item):
        """
        >>> ItemAccessor.to_slices('[:1]')
        (slice(None, 1, None),)
        >>> ItemAccessor.to_slices(slice(1))
        (slice(None, 1, None),)
        >>> ItemAccessor.to_slices(':')
        (slice(None, None, None),)
        >>> ItemAccessor.to_slices('[::-1]')
        (slice(None, None, -1),)
        >>> ItemAccessor.to_slices('[-1:]')
        (slice(-1, None, None),)
        >>> ItemAccessor.to_slices('[1]')
        (1,)
        >>> ItemAccessor.to_slices("['U06'][:1]")
        ('U06', slice(None, 1, None))
        >>> ItemAccessor.to_slices("a")
        ('a',)
        >>> ItemAccessor.to_slices("item")
        ('item',)
        >>> ItemAccessor.to_slices("Dummy")
        ('Dummy',)
        """
        if isinstance(item, str):
            class Dummy:
                def __init__(self, prev=()):
                    self.prev = prev

                def __getitem__(self, item):
                    if item == self.__class__:
                        # Without this, the key Dummy would cause an issue.
                        item = self.__class__.__name__
                    return Dummy(self.prev + (item,))

            if '[' in item:
                assert ']' in item, item
                assert item.count('[') == item.count(']'), item
                m = re.fullmatch('(\[[^\[\]]+\])+', item)
                assert m, (m, item, 'Expected something like "[...][...]".')
                items = cls.to_slices(item.strip('[]').split(']['))
            else:
                # ToDo: Use ast something to evaluate the string.
                #       Fallback is then the string that is checked to be valid.
                #       Alternative: Regex that finds all entries.

                # Support the following syntax:
                #  - 'a'  # happens, when using ['a']
                #  - '"a"'  # happens, when using ['["a"]']
                #  - '1'
                #  - ':1'
                class Local(dict):
                    def __missing__(self, key):
                        return key
                items = eval(
                    f'Dummy()[{item}].prev', None, Local({'Dummy': Dummy})
                )
        elif isinstance(item, slice):
            items = (item,)
        elif isinstance(item, list):
            ret = []
            for i in item:
                slices = cls.to_slices(i)
                assert len(slices) == 1, (slices, item)
                ret.append(slices[0])
            return tuple(ret)
        elif isinstance(item, tuple):
            return tuple(cls.to_slices(list(item)))
        else:
            raise NotImplementedError(item)

        assert isinstance(items, tuple), (items, type(items), item)
        return items
