# HS Stroke dataset: an upper-limb motor imagery EEG dataset of chronic stroke patients

## Overview

The **HS Stroke dataset** contains a large-scale collection of motor imagery (MI) EEG recordings from 14 stroke patients performing left- and right-hand MI tasks. Each participant completed 20 recording sessions over approximately 20 days.

In addition to EEG recordings, the dataset includes pre- and post-training clinical and neurophysiological measures, including FMA scores, sEMG, and MEPs. These measures provide complementary insights into behavioral performance and underlying neural activity, enabling a holistic evaluation of motor recovery.

All data are organized according to the EEG-BIDS standard, including:

- Experimental stimuli
- Source data
- Derivatives data
- Patient information
- Analysis code

## Purpose and Usage

This dataset enables:

1. Examination of neurophysiological differences between left- and right-hand MI in stroke patients.
2. Development and benchmarking of MI-based brain-computer interface algorithms.
3. Investigation of the relationship between EEG features and motor recovery, as measured by FMA scores.

## Data Highlights

- **EEG Classification:** Baseline MI data were analyzed using state-of-the-art methods, demonstrating the quality and discriminability of the EEG signals.
- **FMA Score Regression.:** Exploratory analyses indicate that MI-related EEG features are significantly associated with FMA scores, suggesting that EEG signals can reflect post-stroke motor function.

## Ethical Compliance

All procedures were approved by the Ethics Review Committee of Huashan Hospital and conducted in accordance with the principles of the Declaration of Helsinki. Informed consent was obtained from all participants prior to data collection.

## Acknowledgments and Data Update

We would like to express our gratitude to Zhiwei Deng from the University of Science & Technology of China for pointing out the existence of repeated trials across different runs. Following a thorough quality review, these duplicate trials within the same session have been excluded from the current release. Please note, however, that this issue does not exist across different sessions or subjects. The updated dataset contains 23,382 trials and 93,528 samples (with a 1-second window). We sincerely apologize for any inconvenience caused if you have downloaded the previous version. A detailed update notice for the experiment results is provided in [updateNotice.pdf](updateNotice.pdf).
