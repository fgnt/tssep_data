
# TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings

This repository contains the data preparation and evaluation code for the TS-SEP paper.
The core and training code is available at https://github.com/merlresearch/tssep .

# Installation

Using an existing environment, you can install the data preparation code with:
```
git clone https://github.com/merlresearch/tssep.git
cd tssep
pip install -e .
cd ..
git clone https://github.com/fgnt/tssep_data.git
cd tssep_data
pip install -e .
```

If you want so setup a fresh environment, see [tools/README.md](tools/README.md).
Once you have installed a fresh environment, you can activate it with `. tools/path.sh` (It will also setup some environment variables).

Note: Kaldi and MPI are required for the recipes.
For ASR, you can use
`openai-whisper`, `espnet` or `nemo_toolkit` as alternatives.
ToDo: Limit this to whisper, it has less dependencies.

# LibriCSS data preparation, training and evaluation

[egs/libri_css/README.md#steps-to-run-the-recipe](egs/libri_css/README.md#steps-to-run-the-recipe) contains the instructions
for the LibriCSS data preparation, training and evaluation.

# LibriCSS evaluation with pretrained model

[egs/libri_css/README.md#steps-to-evaluate-a-pretrained-model](egs/libri_css/README.md#steps-to-evaluate-a-pretrained-model)
contains the instructions for the LibriCSS evaluation with a [pretrained model](https://huggingface.co/boeddeker/tssep_77_62000/tree/main).
