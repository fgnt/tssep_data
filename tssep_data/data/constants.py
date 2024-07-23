from pathlib import Path
import tssep_data

egs_dir = tssep_data.git_root / 'egs'
eg_dir = egs_dir / 'libri_css'
json_dir = eg_dir / 'data/jsons'

if not json_dir.exists():
    for p in [
        '/scratch/hpc-prf-nt1/cbj/deploy/tssep_data/egs/libri_css/data/jsons',
        '/net/vol_old/boeddeker/deploy/tssep_data/egs/libri_css/data/jsons',
        '/scratch/hpc-prf-nt2/cbj/deploy/tssep_data/egs/libri_css/data/jsons',
        '/scratch-n1/hpc-prf-nt1/cbj/deploy/tssep_data/egs/libri_css/data/jsons',
        '/scratch-n2/hpc-prf-nt2/cbj/deploy/tssep_data/egs/libri_css/data/jsons',
    ]:
        p = Path(p)
        if p.exists():
            json_dir = p
            eg_dir = json_dir.parent.parent
            egs_dir = eg_dir.parent
            break