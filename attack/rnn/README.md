# Usage


#### Train text generator model

##### 1.Train lstm based text generator model in directory `model_train`

-`python3 model_training.py`-


#### Generate text with trained generator model

##### 1.Generate raw text from trained lstm model in directory `model_train`

-`python3 gen_text.py`-



#### Post process raw text

In order to perform File-Injection Attacks, we need to insert a set of keywords into each raw text generated with lstm model. Thus, we perform this text post process step.

##### 1. perform text post process step in directory `text_post_process`

-`python3 text_post_process.py`-
