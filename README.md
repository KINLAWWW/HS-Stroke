# HS Stroke dataset: an upper-limb motor imagery EEG dataset of chronic stroke patients

## Overview

This repository provides code for experiments on the HS Stroke dataset, an
upper-limb motor imagery EEG dataset collected from chronic stroke patients.
The dataset contains left- and right-hand motor imagery recordings together
with clinical and neurophysiological measures, including FMA scores. The complete dataset is available from Figshare. 

## Purpose and Usage

The code supports:

1. MI classification under cross-trial and cross-session evaluation.
2. FMA score prediction under LOSO and LOSSO evaluation.

Example commands:

```bash
python mi_classification/cross_trial/run.py --model eegnet
python fma_regression/loso/run.py --model ifnet
```

## Acknowledgments and Data Update

We would like to express our gratitude to Zhiwei Deng from the University of Science & Technology of China for pointing out the existence of repeated trials across different runs. Following a thorough quality review, these duplicate trials within the same session have been excluded from the current release. Please note, however, that this issue does not exist across different sessions or subjects. The updated dataset contains 23,382 trials and 93,528 samples (with a 1-second window). We sincerely apologize for any inconvenience caused if you have downloaded the previous version. A detailed update notice for the experiment results is provided in [dataset/updateNotice.pdf](dataset/updateNotice.pdf).
