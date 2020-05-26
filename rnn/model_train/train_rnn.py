from textgenrnn import textgenrnn
import time


start_time = time.time()

model_cfg = {
    'word_level': False,    # set to True if want to train a word_level model (requires more data and smaller max_length)
    'rnn_size': 256,    #number of LSTM cells of each layer (128/256 recommended)
    'rnn_layers': 3,  # num of LSTM layers (>= recommended)
    'rnn_bidirectional': True,  # consider text both forwards and backward, can give a training boost
    'max_length': 5,   # num of tokens to consider before predicting the next (20-40 for characters, 5-10 for words recommended)
    'max_words': 10000000,     # maximum number of words to model;
    'dim_embeddings': 100,  # dimensionality of the character/word embedings (default 100)
}


train_cfg = {
    'line_delimited': True,    # set to True if each text has its own line in the source file
    'batch_size': 256, # with GPU, could use larger batch size, in the mean time, the epochs may be also need to be increased
    'num_epochs': 50,   # set higher to train the model for longer
    'gen_epochs': 10,    # once the given num of epochs finished, generate samples based on the trained model
    'train_size': 1.0,  # proportion of input data to train on: setting < 1.0 limits model from learning perfectly
    'dropout': 0.0,     # ignore a random proportion of source tokens each epoch, allowing model to generalize better
    'validation': False,    # if train_size < 1.0, test on holdout dataset; will make overall training slower
    'is_csv': False,     # set to True if file is a CSV exported from Excel/BigQuery/pandas

    'multi_gpu':True,
    'num_gpus': 1,
}


input_file = '../datasets/lifestyle_progresspics_processed.txt'
model_name = './models/lifestyle_epoch50'
textgen = textgenrnn(name = model_name)
textgen.reset()
train_function = textgen.train_from_file if train_cfg['line_delimited'] else textgen.train_from_largetext_file
train_function(
    file_path = input_file,
    new_model = True,
    num_epochs = train_cfg['num_epochs'],
    gen_epochs = train_cfg['gen_epochs'],
    batch_size = train_cfg['batch_size'],
    train_size = train_cfg['train_size'],
    dropout = train_cfg['dropout'],
    validation = train_cfg['validation'],
    is_csv = train_cfg['is_csv'],
    rnn_layers = model_cfg['rnn_layers'],
    rnn_bidirectional = model_cfg['rnn_bidirectional'],
    rnn_size = model_cfg['rnn_size'],
    max_length = model_cfg['max_length'],
    dim_embeddings = model_cfg['dim_embeddings'],
    word_level = model_cfg['word_level'],

    multi_gpu = train_cfg['multi_gpu'],
    num_gpus = train_cfg['num_gpus']
)

terminate_time = time.time()
running_time = (terminate_time - start_time) / 60.0
print('running time is {} seconds.'.format((float(running_time))))
