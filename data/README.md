# Usage

This folder contains three public datasets, you can find the details of each dataset in following sections


#### Amazon Review Dataset

You can download the dataset from the link https://www.kaggle.com/bittlingmayer/amazonreviews. We extract a small sub-dataset from the 'test.ft.txt.bz2' since we didn't have enough computational power to process the whole dataset when we trained LSTM model.


#### Enron Email Dataset   

You can download the dataset from the link https://www.cs.cmu.edu/~enron/. We provide the top keywords file of this dataset `enron_top_keywords_meaningful.txt` by extract top 5000 frequency keywords and remove all un-meaningful words which can't find in English dictionary. This file is used to generated sets of keywords that inject to files.


#### Science Dataset

We obtain this dataset from the library `nltk`. We provide this dataset and corresponding top 5000 frequency keywords file. The usage of this dataset is similar to Enron Email Dataset. 
