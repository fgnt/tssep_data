"""
python -m tssep.libricss.fix_exampleid rttm vbx_ovl/*.rttm --out=vbx_ovl_fix

"""
import meeteval
from pathlib import Path


def rttm(*rttms, out):

    from tssep_data.database.libri_css.example_id_mapping import LibriCSSIDMapper_v2

    id_mapper = LibriCSSIDMapper_v2()
    out = Path(out)
    if not out.exists():
        out.mkdir(parents=True)

    for file in rttms:
        file = Path(file)
        rttm = meeteval.io.RTTM.load(file)
        new_file = out / file.name
        if new_file.exists():
            print(f'File {new_file} already exists. Skipping.')
            continue
        new = []
        for line in rttm.lines:
            new.append(line.replace(filename=id_mapper.to_folder(line.filename)))
        new = meeteval.io.RTTM(new)

        new.dump(new_file)
        print('Wrote', new_file)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'rttm': rttm,
    })
