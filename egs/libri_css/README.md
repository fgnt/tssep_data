# Steps to run the recipe

## Download and prepare external data

WARNING: The data preparation will generate roughly 5.2 TB of data.
Make sure, you have enough space on your hard drive.

Note: The scripts are interactive and will ask at some positions what it should
      do.

Go to the `<tssep_data>/egs/libri_css/data` folder and run
`make libri_css sim_libri_css prepare_sim_libri_css prepare_libri_css` to 
 - download libri_css (stage 1),
 - create sim_libri_css (stage 2),
 - prepare sim_libri_css (stage 3) and 
 - prepare libri_css (stage 4).

Alternatively, you can run `python make.py` in `<tssep_data>/egs/libri_css/data` and select stage 1, 2, 3 or 4.

## Training

The training is done in two steps. First, the TS-VAD model is trained and then
the TS-SEP model.

To start the training of the TS-VAD model, run the following command:
```bash
make tsvad
```
This will create the file `tsvad/config.yaml` and ask how to start the
training, i.e. start the training on the current machine or use slurm to submit
a job (We recommend the first option, but the Paderborn cluster requires the
second).
If you want to change some parameters before the training, you can press Ctrl-C
instead of selecting a start option and edit the `config.yaml` file. Then 
simply type again `make tsvad` and select the start option.

With [tensorboard](https://www.tensorflow.org/tensorboard), you can monitor the training progress.

Once the training loss goes down, you can start the training of the TS-SEP
model. Just run the following command:
```bash
make tssep
```
This will again create the file `tssep/config.yaml` and ask to select a
checkpoint, before asking how to start the training. Again, you can change the
parameters before starting the training, just as with the TS-VAD model.

## Evaluation

To evaluate the TS-SEP model, go in the `tssep` folder and run the following
command:
```bash
make eval_init  # or make eval_init_<checkpointnumber> e.g. make eval_init_62000
```
where `<checkpointnumber>` is the number of the checkpoint you want to
evaluate.

Next, go to the newly created folder (e.g. `eval/62000/1`) and run the
following command:
```bash
make run  # make sbatch if you want to submit the job to slurm
```

This will create an audio folder with the separated sources and a `c7.json`
file, which contains the results of the separation.

Next, you have to apply an ASR model to the enhance data.
To use whisper `tiny.en`, run the following command:
```bash
transcribe_tiny.en
```
This will create a `asr` folder with the transcriptions and calculate the 
`WER`. Note, that `tiny.en` is a weak, but fast ASR model. So you will get
a `WER` in the range of 10 % to 20 %, while stronger ASR models will give you
`WER`'s below 10 %.

# Steps to evaluate a pretrained model

## Download and prepare external data

While you could follow the instructions from the previous section to download
and prepare the data, it does more than necessary for the evaluation of a
pretrained model.

It is sufficient to download the LibriCSS data and prepare it. To do so, run
the following command in the `<tssep_data>/egs/libri_css/data` folder:
```bash
make libri_css prepare_libri_css
```

## Evaluation

To evaluate a pretrained model, run the following command in the `<tssep_data>/egs/libri_css` folder:
```bash
make tssep_pretrained_eval
```
That will:
 - initialize a training folder from `cfg/tssep_pretrained_77_62000.yaml`,
 - download a checkpoint of a pretrained model from https://huggingface.co/boeddeker/tssep_77_62000,
 - download feature_statistics from https://huggingface.co/boeddeker/tssep_77_62000 for the domain adaptation,
   - without the downloaded statistics, the statistics will be calculated,
     hence sim_libri_css would be required.
 - run the evaluation,
 - transcribe the separated audio with an ASR from the nemo-toolkit.

Note, the WER should be around `6.32 %` with the ASR model from the nemo-toolkit.

# MPI issues

MPI is an old library and very common in HPC setups, nevertheless, some HPC setups don't work out of the box.
(Workstations make less issues, just type `sudo apt install libopenmpi-dev`, or the counterpart of your package manager, 
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
