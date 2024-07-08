# Steps to run the recipe

## Download and prepare external data preparation

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
