# PyCharm bug: Doctests in __main__.py don't work

if __name__ == '__main__':
    from .__main__ import cli
    cli()