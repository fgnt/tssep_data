
# TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings

[![IEEE DOI](https://img.shields.io/badge/IEEE/DOI-10.1109/TASLP.2024.3350887-blue.svg)](https://doi.org/10.1109/TASLP.2024.3350887)
[![arXiv](https://img.shields.io/badge/arXiv-2303.03849-b31b1b.svg)](https://arxiv.org/abs/2303.03849)

This repository contains the data preparation and evaluation code for the TS-VAD
and TS-SEP experiments in our 2024 IEEE/ACM TASLP article,
**TS-SEP: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings**
by Christoph Boeddeker, Aswin Shanmugam Subramanian, Gordon Wichern, Reinhold Haeb-Umbach, Jonathan Le Roux
([IEEE Xplore](https://doi.org/10.1109/TASLP.2024.3350887), [arXiv](https://arxiv.org/abs/2303.03849)).

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

# Cite

If you are using this code please cite our paper ([![IEEE DOI](https://img.shields.io/badge/IEEE/DOI-10.1109/TASLP.2024.3350887-blue.svg)](https://doi.org/10.1109/TASLP.2024.3350887)
[![arXiv](https://img.shields.io/badge/arXiv-2303.03849-b31b1b.svg)](https://arxiv.org/abs/2303.03849)):

```
@article{Boeddeker2024feb,
    author = {Boeddeker, Christoph and Subramanian, Aswin Shanmugam and Wichern, Gordon and Haeb-Umbach, Reinhold and Le Roux, Jonathan},
    title = {{TS-SEP}: Joint Diarization and Separation Conditioned on Estimated Speaker Embeddings},
    journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    year = 2024,
    volume = 32,
    pages = {1185--1197},
    month = feb,
}
```
