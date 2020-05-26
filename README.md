# MitigateFileInjectionAttacks

This repository contains the source code and data for the paper "Mitigating File-Injection Attacks with Natural Language Processing"

**The dataset is public dataset and the code is for research purpose only**. The results of this study are published in the following paper:

Hao Liu, Boyang Wang, *"Mitigating File-Injection Attacks with Natural Language Processing,"* The 6th ACM International Workshop on Security and Privacy Analytics 2020 (**IWSPA 2020**).

## Content

This repository contains separate directories for the attack, defense, and the datasets. A brief description of the contents of these directories is below.  More detailed usage instructions are found in the individual directories' README.md.

#### Attack

##### attack with n-gram
![attack model 1](https://github.com/haoliutj/MitigateFileInjectionAttacks/blob/master/ngramFileInjection.jpg)

##### attack with rnn
![attack model 2](https://github.com/haoliutj/MitigateFileInjectionAttacks/blob/master/rnnFileInjection.jpg)

The ```attack``` directory contains the code for the deep learning classification models, data preparation of the input to the models, and some related utility functions.

#### Defense

The `defense` directory contains the code for the proof-of-concept defense against our attack.   

#### Voice Traffic Collection Tool

![collection tool](https://github.com/SmartHomePrivacyProject/DeepVCFingerprinting/blob/master/collection%20tool.png)

The ```collection_tool``` directory contains the code and setup instructions for the collection tool that we use to collect our voice traffic data.  There are also utilities to generate synthetic voice audio files that use text-to-speech APIs and the list of common smart speaker queries in CSV format.  

#### Datasets

The `datasets` has information about where you can find and download the datasets.

#### Additional Information

The `additional_info` has information about the structures of neural networks, lists of voice commands we studied, tuned hyperparameters.  

## Requirements

This project is entirely written in Python 3.  Some third party libraries are necessary to run the code.  

## Usage

See the project's directories for usage information.

## Citation

When reporting results that use the dataset or code in this repository, please cite:

Chenggang Wang, Sean Kennedy, Haipeng Li, King Hudson, Gowtham Atluri, Xuetao Wei, Wenhai Sun, Boyang Wang, *“Fingerprinting Encrypted Voice Traffic on Smart Speakers with Deep Learning,”* ACM Conference on Security and Privacy in Wireless and Mobile Network (**ACM WiSec 2020**), July, 2020. (The first two authors contribute equally in this paper)

## Contacts

Chenggang Wang, wang2c9@mail.uc.edu, University of Cincinnati

Sean Kennedy, kenneds6@mail.uc.edu, Leidos

Boyang Wang, boyang.wang@uc.edu, University of Cincinnati
