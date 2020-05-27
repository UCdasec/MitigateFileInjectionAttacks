# Usage

Instead of separating model training, text generation and text post process steps, we integrate all those steps into one function with ngram model, since it easier to customized inside the model compare to rnn model.


#### Generate text

##### 1.Generate text from ngram model in directory `ngrams`

-`python3 gen_text.py`-

The texts generated here can be used to perform File-Injection Attacks directly.
