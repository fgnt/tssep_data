from pathlib import Path
import paderbox as pb


def main(folder='.'):
    folder = Path(folder).absolute()
    assert folder.is_dir(), folder
    if (folder / 'checkpoints').exists():
        folder = folder / 'checkpoints'
    assert (folder / 'ckpt_latest.pth').exists(), folder

    ckpt = pb.io.load(folder / 'ckpt_latest.pth', unsafe=True)

    ranking = ckpt['hooks']['BackOffValidationHook']['ckpt_ranking']

    from tssep_data.util.cmd_runner import user_select, run
    try:
        file = user_select('Select a checkpoint', {
            f'{file} (loss={loss})': folder / file
            for file, loss in ranking
        })
    except KeyboardInterrupt:
        print('Aborted by Ctrl+C')
        return
    assert file.exists(), file

    run(
        f'python -m tssep_data.eval.run init with config.yaml default eeg.ckpt={file}'
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
