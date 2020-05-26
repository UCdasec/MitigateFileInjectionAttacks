# MitigateFileInjectionAttacks

This repository contains the source code and data for the paper "Mitigating File-Injection Attacks with Natural Language Processing"

**The dataset is public dataset and the code is for research purpose only**. The results of this study are published in the following paper:

Hao Liu, Boyang Wang, *"Mitigating File-Injection Attacks with Natural Language Processing,"* The 6th ACM International Workshop on Security and Privacy Analytics 2020 (**IWSPA 2020**).

## Content

This repository contains separate directories for the attack, defense, and the datasets. A brief description of the contents of these directories is below.  More detailed usage instructions are found in the individual directories' README.md.

#### Attack

##### 1.Attack with N-Gram
![attack model 1](https://github.com/haoliutj/MitigateFileInjectionAttacks/blob/master/ngramFileInjection.jpg)

##### 2.Attack with RNN
![attack model 2](https://github.com/haoliutj/MitigateFileInjectionAttacks/blob/master/rnnFileInjection.jpg)

The ```attack``` directory contains two attack methods 'ngrams' and 'rnn'. The code in each method directory includes text generation model, post process function of generated text, and corresponding utility functions.

#### Defense

The `defense` directory contains the code for the text_semantic based defense against the attack.   


#### Datasets

The `datasets` has information about where you can find and download the datasets.


## Requirements

This project is entirely written in Python 3.  Some third party libraries are necessary to run the code.  

## Usage

See the project's directories for usage information.

## Citation

When reporting results that use the dataset or code in this repository, please cite:

Hao Liu, Boyang Wang, *“Mitigating File-Injection Attacks with Natural Language Processing,”* The 6th ACM International Workshop on Security and Privacy Analytics 2020 (**IWSPA 2020**).

## Contacts

Hao Liu, liu3ho@mail.uc.edu, University of Cincinnati

Boyang Wang, boyang.wang@uc.edu, University of Cincinnati
