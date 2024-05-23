
# TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings

This repository contains the data preparation code for the TS-SEP paper.
The training and evaluation code is available at
https://github.com/merlresearch/tssep .


# Installation

Using an existing environment, you can install the data preparation code with:
```
git clone https://github.com/fgnt/tssep_data.git
cd tssep_data
pip install -e .
cd ..
git clone https://github.com/merlresearch/tssep.git
cd tssep
pip install -e .
```

If you want so setup a fresh environment, see [tools/README.md](tools/README.md).
Once you have installed a fresh environment, you can activate it with `. tools/path.sh` (It will also setup some environment variables).

Note: Kaldi and MPI are required for the recipes.
For ASR, you can use
`openai-whisper`, `espnet` or `nemo_toolkit` as alternatives.
ToDo: Limit this to whisper, it has less dependencies.

# Data preparation

WARNING: The data preparation will generate roughly 5.2 TB of data.
Make sure, you have enough space on your hard drive.

Note: The scripts are interactive and will ask at some positions what it should
      do.

Steps:

 - Follow the instructions from https://github.com/chenzhuo1011/libri_css.
   Clone the repository to `egs/libri_css/libri_css` or create a symlink 
   `ln -s /path/to/libri_css egs/libri_css/libri_css`.
 - Once that is done, go to the `egs/libri_css` folder and run `make sim_libri_css prepare_sim_libri_css prepare_libri_css ivector` to create sim_libri_css (stage 2), prepare sim_libri_css (stage 3), prepare libri_css (stage 4) and create ivector (stage 5).
     - Alternatively, you can run `python make.py` and select stage 2, 3, 4 or 5.


# MPI issues

MPI is an old library and very common in HPC setups, nevertheless, some HPC setups don't work out of the box.
(Workstations make less issues, just type `sudo apt install libopenmpi-dev` (or the counterpart of your package manager)
and it will install MPI).

## A high performance Open MPI point-to-point messaging module was unable to find any relevant network interfaces

The warning "A high performance Open MPI point-to-point messaging module was unable to find any relevant network interfaces
[...] Another transport will be used instead, although this may result in lower performance."
is an annoying warning, but uncritical. MPI is mainly used to manage the workers.
The broadcast and gather operations don't need a fast communication, because they are rarely used.

## MPI doesn't work at all. What can I do?

Sometimes the setup of the computing machines are messed up for MPI and a fix might be nontrivial
(e.g. missing permissions to install MPI development packages).
In such a case, you can uninstall `mpi4py` and run everything with a single core (i.e. fix the calls to use 1 MPI job, for SLURM:`-n X` -> `-n 1`)
The code that uses MPI will be slower (roughly `X-1` times slower).
The training time will be unaffected and for evaluation a GPU can compensate MPI (For LibriCSS GPU based eval is default).

A missing `mpi4py` is untested and the code might require a few minor changes 
(e.g. when an exception complains, that `allow_single_worker` should be `True`, change it to `True`).
