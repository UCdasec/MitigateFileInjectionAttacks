from textblob import TextBlob
import pandas as pd
import os,copy
import nltk
from collections import Counter
import random
import glob


import sys
reload(sys)
sys.setdefaultencoding('utf-8')



def retrieve_filelist(input_path):
    items = os.listdir(input_path)

    #for benign ordering
    #items.sort(key = lambda x: int(x[5:-4]))

    #for faked mails ordering
    items.sort(key = lambda x: int(x[9:-4]))

    return items


def read_file(filename):
    with open(filename,'r+') as f:
        input_file = f.readlines()
        print('num of texts {}'.format(len(input_file)))
    return input_file


def init():
    numwrds_list = []
    numstens_list = []
    avg_sten_len_list = []
    avg_wrd_len_list = []
    numNN = []
    numVB = []
    numJJ = []
    numRB = []
    numPRP = []
    paras = [numwrds_list,numstens_list,avg_sten_len_list,avg_wrd_len_list,
    numNN,numVB,numJJ,numRB,numPRP]
    return paras


def syntactic_feature(input_file):
    """
    nouns: NN
    verbs: VB
    adjectives : JJ
    adverb: RB
    personal pronouns: PRP
    """
    synt_name = ['NN','VB','JJ','RB','PRP']
    token = nltk.word_tokenize(input_file)
    tags = nltk.pos_tag(token)
    counts = Counter(tag for word,tag in tags)
    synt_stat_dict = dict((tag,count) for tag,count in counts.items())
    synt_results = [synt_name,synt_stat_dict]
    return synt_results


def structure_feature(input_file):
    """
    # of words;
    # of sentences;
    the average sentence length (# of words)
    the average word length (# of character)
    """
    struc_name = ['# of words','# of sentences','avg_stence_length','avg_word_length']
    corpus = TextBlob(input_file)
    wrds = corpus.words
    numwrds = len(wrds)
    stens = corpus.sentences
    numstens = len(stens)
    str_len = len(input_file)
    if numstens == 0:
        avrg_sten_len = 0
    else:
        avrg_sten_len = '%.2f' % (float(numwrds) / float(numstens))
    if numwrds == 0:
        avrg_wrd_len = 0
    else:
        avrg_wrd_len = '%.2f' % (float(str_len) / float(numwrds))
    struc_stat = [numwrds,numstens,avrg_sten_len,avrg_wrd_len]
    struc_results = [struc_name,struc_stat]
    return struc_results



def features_append(paras,struc_results,synt_results):
    struc_name = struc_results[0]
    struc_stat = struc_results[1]
    for i in range(len(struc_stat)):
        paras[i].append(struc_stat[i])
    synt_name = synt_results[0]
    synt_stat_dict = synt_results[1]
    for i in range(len(struc_stat),len(struc_stat) + len(synt_name)):
        paras[i].append(synt_stat_dict.get(synt_name[i-len(struc_stat)],0))
    return paras


def label_create(label_symbol,num_labels):
    # label symbol: False means '0' malicious; True means '1' benign

    label_dic = {}
    label_name = 'label'
    if label_symbol == False:
        labels = [0] * num_labels
    else:
        labels = [1] * num_labels
    if label_name not in label_dic:
        label_dic[label_name] = []
    label_dic[label_name] = labels
    return label_dic


def write_to_file(struc_name,synt_name,paras,label_dic):
    "output results in a csv file"
    "features should corresponding to paras"
    dic = {}
    features = copy.deepcopy(struc_name)
    features.extend(synt_name)
    #for_loop output: dic = {features[i]:results[i]};
    for i in range(len(features)):
        if features[i] not in dic:
            dic[features[i]] = []
        dic[features[i]] = paras[i]
    print(label_dic)
    dic.update(label_dic)
    print(dic.keys())
    df = pd.DataFrame(dic)
    features.extend(['label'])
    df = df[features]
    df.index = range(1,len(df) + 1)
    df.to_csv('../result/features.csv')


def writeMultiFile(struc_name,synt_name,paras,label_dic,i):
    "output results in a csv file"
    "features should corresponding to paras"
    output = 'features_'+str(i)+'.csv'
    dic = {}
    features = copy.deepcopy(struc_name)
    features.extend(synt_name)
    #for_loop output: dic = {features[i]:results[i]};
    for i in range(len(features)):
        if features[i] not in dic:
            dic[features[i]] = []
        dic[features[i]] = paras[i]
    print(label_dic)
    dic.update(label_dic)
    print(dic.keys())
    df = pd.DataFrame(dic)
    features.extend(['label'])
    df = df[features]
    df.index = range(1,len(df) + 1)
    df.to_csv('../result/' + output)


def random_choose_file4extract(texts_list,num_file):
    chosen_texts = random.sample(texts_list,num_file)
    return chosen_texts


def retrieve_filepaths(input):
    # format:  '/*.txt' or '/*.csv'
    all_files = glob.glob(input + '/*.txt')
    return all_files


def main(textfilepath,num_output_file, label_symbol):

    # filename = '../data/rnn_rc_epoch100/replaced_rnn_rc_epoch100_num250_len50.txt'
    # filename = '../data/ngram_generated_texts/num250_len50_injected20/generated_texts_0.txt'
    # filename = '../data/reddit_relationshipadvice_legaladvice_2000.txt'

    # if it's orginal corpus, means that need randome choose fixed num of texts from it
    # if textfilepath == '../data/reddit_relationshipadvice_legaladvice_2000.txt':
    if textfilepath == '../data/amazon_review_part1.txt':
        flag_chosen_texts = True
    else:
        flag_chosen_texts = False

    paras = init()
    texts_list = read_file(textfilepath)
    print('the number of input file is {}'.format(len(texts_list)))
    if flag_chosen_texts:
        num_chosen_file = num_output_file  # choose the num of files for extraction
        chosen_texts = random_choose_file4extract(texts_list,num_chosen_file)
        texts_list = chosen_texts
    i = 0
    for text in texts_list:
        print(i)
        print(text)
        struc_results = structure_feature(text)
        synt_results = syntactic_feature(text)
        paras = features_append(paras,struc_results,synt_results)
        i += 1
    struc_name = struc_results[0]
    synt_name = synt_results[0]
    label_dic = label_create(label_symbol,len(paras[0]))
    # write_to_file(struc_name,synt_name,paras,label_dic)
    return struc_name,synt_name,paras,label_dic


def main_loop(inputpath,num_output_file,label_symbol):
    " multi-features files generating "
    all_files = retrieve_filepaths(inputpath)
    print(all_files)
    for i in range(len(all_files)):
        print('feature extration {}....'.format(i))
        struc_name, synt_name, paras, label_dic = main(all_files[i],num_output_file, label_symbol)
        writeMultiFile(struc_name,synt_name,paras,label_dic,i)



if __name__ == '__main__':
    print('feature_extrat.py is running...')

    " single features-file generating "
    # filename = '../data/rnn_rc_epoch100/replaced_rnn_rc_epoch100_num250_len50.txt'
    # filename = '../data/ngram_generated_texts/num1000_len50_injected20/generated_texts.txt'
    # filename = '../data/reddit_relationshipadvice_legaladvice_2000.txt'
#     filename = '../data/rnn_raw_text_epoch100/num320_len160_inject80/generated_texts.txt'
    #
#     num_output_file = 1000       # choose num_output_file from rc_corpus
#     label_symbol = False        # malicious: False 0; benign: True 1
#     struc_name, synt_name, paras, label_dic = main(filename,num_output_file, label_symbol)
#     write_to_file(struc_name, synt_name, paras, label_dic)


    " extract features for benign texts "
    # inputpath = '../data/reddit_relationshipadvice_legaladvice_2000.txt'
    # inputpath = '../data/amazon_review_part1.txt'
    # num_output_file = 1749
    # label_symbol = True
    # struc_name, synt_name, paras, label_dic = main(inputpath, num_output_file, label_symbol)
    # write_to_file(struc_name, synt_name, paras, label_dic)



    " multi features-files generating "
    num_output_file = 1  # choose num_output_file from benign corpus, only needed when extract features for benign data, no impact for malicious feature extraction
    label_symbol = False  # malicious: False 0; benign: True 1

    # inputpath = '../data/reddit_relationshipadvice_legaladvice_2000.txt'
    # inputpath = '../data/ngram_generated_texts/keyword_science_remove_stopword/num316_len160_injected80'
    # inputpath = '../data/rnn_rc_epoch200/keyword_science_remove_stopword/num317_len160'
    # inputpath = '../data/amazon_review/science/rnn/epoch40/inject40'
    inputpath = '../data/amazon_review/science/ngram/inject40'


    main_loop(inputpath,num_output_file,label_symbol)
