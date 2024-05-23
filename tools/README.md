# Dependencies

[Kaldi](https://github.com/kaldi-asr/kaldi) is required for the I-Vector
extraction. The installation is described in the
[Kaldi documentation](https://kaldi-asr.org/doc/install.html).
After installation, set the `KALDI_ROOT` environment variable to the Kaldi 
folder.

Install or activate MPI, see `Install MPI` later in this file.

# Install a fresh miniconda environment with all dependencies.

First clone this repository with `git clone https://github.com/fgnt/tssep_data.git` and
`git clone https://github.com/merlresearch/tssep.git`
then go to the tools folder (i.e. `cd tssep_data/tools`).

Pytorch will be installed with the default CUDA version that is defined by the 
PyTorch team can can be found on https://pytorch.org/get-started/locally/.
If you need a different CUDA version, you have to change the installation
command in the `install_miniconda.py` and search for `pip install torch`.
Replace the pip install command with the command that installs pytorch with
you desired CUDA version. You can find those commands on 
https://pytorch.org/get-started/locally/.

Type `python install_miniconda.py` to install miniconda and all dependencies.

Type `source path.sh` (or `source tools/path.sh` if you are in the root folder
of the git) to activate the python environment.

Go to `Is mpi4py properly installed?` in this file to check if `mpi4py` is
properly installed.

# Manual installation without conda

Activate your python environment and install the following packages:
```
pip install torch torchvision torchaudio  # check https://pytorch.org/get-started/locally/ for alternative CUDA versions
pip install openai-whisper  # ASR with minimal dependencies
# pip install espnet espnet_model_zoo  # alternative ASR
# pip install nemo_toolkit  # alternative ASR

git clone https://github.com/merlresearch/tssep.git
cd tssep
pip install -e .
```

Go to `Is mpi4py properly installed?` in this file to check if `mpi4py` is
properly installed.

# FAQ Troubleshooting

## Install MPI

On Ubuntu, install the following packages:
```bash
sudo apt install openmpi-bin openmpi-dev openmpi-common openmpi-doc libopenmpi-dev
```

On the MERL cluster, make sure to `module add openmpi` before running the
installation, or the mpi4py installation will fail.

## Is mpi4py properly installed?

`mpi4py` has sometimes issues with a cluster setup, hence here explicit the
installation and test.
```bash
$ conda install mpi4py  # Uses anaconda openmpi
$ #  pip install --force --no-cache mpi4py  # Uses usually system mpi (Recommended for Paderborn cluster)

$ salloc -n 4 mpirun python -m mpi4py.bench helloworld
salloc: Granted job allocation 226827
srun: Step created for job 226827
Hello, World! I am process 0 of 4 on node06.
Hello, World! I am process 1 of 4 on node06.
Hello, World! I am process 2 of 4 on node06.
Hello, World! I am process 3 of 4 on node06.
salloc: Relinquishing job allocation 226827
```
Check that it is `... process X of 4 on ...` and not `... process X of 1 on ...`.

Note that a Kaldi installation is required prior to running the recipes.

If an error message complains about missing GLIBCXX_3.4.29, install it as follows:
```
sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11
```

