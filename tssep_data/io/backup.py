import os
from pathlib import Path


def _get_backup_file(
        file: [str, Path],
        backup_dir: [None, str, Path] = None,
        mkdir: bool = False,
):
    """
    >>> _get_backup_file('config.json')  # doctest: +ELLIPSIS
    PosixPath('config_202....json')
    """
    file = Path(file)
    import datetime
    now = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')
    if backup_dir is None:
        backup_dir = file.parent
    else:
        backup_dir = Path(backup_dir)
        if mkdir:
            backup_dir.mkdir(parents=True, exist_ok=True)

    file_backup = backup_dir / f'{file.stem}_{now}{file.suffix}'
    return file_backup


def _backup_file(
        file: [str, Path],
        print_fn: [None, callable] = None,
        backup_dir: [None, str, Path] = None,
):
    file_backup = _get_backup_file(file, backup_dir, mkdir=True)
    file.rename(file_backup)
    if print_fn is not None:
        print_fn(f"Created backup of {file.name}: {file_backup}")
    return file_backup


def write_text_with_backup(
        file: [str, Path],
        text: str,
        print_fn: [None, callable] = None,
        backup: bool=True,
        backup_dir: [None, str, Path] = None,
):
    """
    Similar to `file.write_text(text)`, but creates a backup of file, if
    it exists as `file.rename(backup_dir / f'file_{now}')` where `now` is
    `datetime.datetime.today().strftime('%Y_%m_%d_%H_%M_%S')`.

    Args:
        file: File, that may be overwritten.
        text: The text, that should be written to that file.
        print_fn: A logging function, e.g. `log.info`
        backup: Whether to create a backup.
        backup_dir: Write the backup to this folder, default `file.parent`.

    Returns:

    """
    file = Path(file)
    file_backup = None
    if backup and file.exists():
        if file.read_text() == text:
            if print_fn is not None:
                print_fn(f"File didn't changed: {file}")
            return
        file_backup = _backup_file(file, print_fn, backup_dir)

    file.write_text(text)
    if print_fn is not None:
        if file_backup is not None:
            import subprocess
            subprocess.run(
                ['icdiff', os.fspath(file_backup), os.fspath(file)],
            )
        print_fn(f"Wrote {file.name}: {file}")
