


# [sim_libri_css_ch_speaker_reverberation_early.json](`/scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json`)
 - `sbatch.py -n 50 -t 12h --mem-per-cpu 10GB --wrap 'srun.py python -m tssep.data.cli.prepare_simlibricss_for_tr /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch.json'`


sbatch.py -n 24 -t 6h --mem-per-cpu 4GB --wrap 'srun.py python -m css.database.sim_libri_css.create_vad v2 /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json'

# /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_fix8spk.json
python -m css.database.sim_libri_css.fix_json /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css.json

# /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early_fix8spk.json

python -m css.database.sim_libri_css.fix_json /scratch/hpc-prf-nt2/cbj/deploy/css/egs/libricss/data/sim_libri_css_ch_speaker_reverberation_early.json