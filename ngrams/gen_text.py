# generated text for different length of inset_keyword sets
# Apr 25 2019

import textGenRestriction
import re,os,random,copy,shutil,time
import n_gramAuto, filter


def pre_process_text(text):
    text = re.sub(r'(/m)', "", text)
    text = re.sub(r'(/f)', "", text)
    text = re.sub(r'(f/)', "", text)
    text = re.sub(r'/', "", text)
    text = re.sub(r'\[(?:[^\]|]*\|)?([^\]|]*)\]', r'\1', text)  # remove []
    text = re.sub(r'\((?:[^\]|]*\|)?([^\]|]*)\)', r'\1', text)  # remove ()
    text = re.sub('\d', "", text)  # remove number
    return text


def read_textlist(filename):
    with open(filename,'r') as f:
        output = f.readlines()
    return output


def preprocess_corpus(filename):
    "preprecessed the corpus and join the content list as a single sentence, such that can be well match for Markov model"
    texts_list = read_textlist(filename)
    texts_list_processed = []
    for text in texts_list:
        text = pre_process_text(text)
        texts_list_processed += text
    corpus_processed = ' '.join(texts_list_processed)
    return corpus_processed


def write2file(gen_texts_total):
    items = os.listdir('../result')
    output = 'generated_texts.txt'
    if output in items:
        os.remove('../result/' + output)
    for text in gen_texts_total:
        text = pre_process_text(text)
        with open('../result/' + output,'a') as f:
            f.write(text + '\n')


def writeMultiFile(gen_texts_total,i,t_size):
    items = os.listdir('../result')
    output = 'generated_texts_' + str(t_size) + '_' + str(i) + '.txt'
    if output in items:
        os.remove('../result/' + output)
    for text in gen_texts_total:
        text = pre_process_text(text)
        with open('../result/' + output,'a') as f:
            f.write(text + '\n')


def remove_insert_keywords(insert_keywords,keywordlist):
    for key in insert_keywords:
        keywordlist.remove(key)
    return keywordlist


def choose_keyword_0(keywordlist,n):
    " in a random way, no overlap between all sets"
    insert_keywords = random.sample(keywordlist,n)
    keywordlist = remove_insert_keywords(insert_keywords,keywordlist)
    return insert_keywords,keywordlist


def read_keywords_from_file(file_name):
    '''
    Given a file_name, which has a list of keywords, read all the keywords
    '''

    print('reading keywords...')
    file = open(file_name, 'r')
    words = file.readlines()
    keywords = []
    for word in words:
        word = word.strip('\n')
        keywords.append(word)
    print('keyword pool as: ', keywords)

    return keywords


def choose_keyword(t_size):
    '''
    choose in File-injection way
    Given a keyword pool and a parameter threshold T ( num of injected keywords in each sentence)
    divide all the keywords into K/T subsets, K total num of keywords
    and output the injected keywords for all the K/2T\logT + K/T files
    t_size: num inserted keyword in each sentence
    '''

    dir = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))
    file_name = dir + '/data/top_keyword_science_remove_stopword.txt'
    keyword_pool = read_keywords_from_file(file_name)
    random.shuffle(keyword_pool)

    # this will output all the words for dividing keyword_pool
    # groups is a part of injected keywords for the K/T files
    groups = filter.group_keywords(keyword_pool, t_size)
    group_pairs = filter.pair_groups(groups)
    print('\n')
    inject_groups = []
    for x in range(0, len(groups)):
        inject_groups.append(groups[x])
    for x in range(0, len(group_pairs)):
        filter.get_inject_keywords(group_pairs[x], inject_groups)
    print('the num of inserted keyword sets is {}'.format(len(inject_groups)))

    return inject_groups


def read_keyword(keyword_file):
    with open(keyword_file,'r') as f:
        # output = f.readlines()
        output = f.read().splitlines()
    return output


def buildMarchovModel(corpus):
    MarchovChain = n_gramAuto.MarkovChain(2)
    MM = MarchovChain.learn(corpus)
    print('Marchov Model is built successfully.')
    print ('\n')
    return MM


init_cfg = {
    'corpusName': 'amazon_review_part1.txt',
    'num_of_gram': 2,
    'num_of_injectedkey': 10
}


def main_test():
    'injected keyword randomly'

    dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), (os.path.pardir)))
    # keywordlist = read_keyword('../data/top_keyword_science.txt')
    keywordlist = read_keyword('../data/enron_top_keywords_meaningful.txt')
    keywordlist_copy = copy.deepcopy(keywordlist)
    # corpus_processed = preprocess_corpus(init_cfg['corpusName'])
    start_time = time.time()
    MarkovModel = textGenRestriction.buildMarchovModel(dir,init_cfg['corpusName'])
    end_time = time.time()
    print("training time for training n-gram model is {}".format(end_time-start_time))
    numOutputFile = len(keywordlist) / init_cfg['num_of_injectedkey']
    lenThreshold,lenTerminal = textGenRestriction.genExpRandomList(numOutputFile)
    genTexts = []
    for i in range(numOutputFile):
        print('generating text...{}'.format(i))
        injected_keyword,keywordlist = choose_keyword(keywordlist,init_cfg['num_of_injectedkey'])
        print(injected_keyword)
        genText = textGenRestriction.textGen(MarkovModel,injected_keyword,keywordlist_copy,lenThreshold[i],lenTerminal[i],init_cfg['num_of_gram'])
        genText = pre_process_text(genText)
        genTexts.append(genText)
    return genTexts


def main(t_size,text_len_para):
    'inject keyword in file injection way'
    # text_len_para: control the length of the length of output text (para = 0.08, len(87 to 131))


    # keywordlist = read_keyword('../data/top_keyword_science_remove_stopword.txt')
    keywordlist = read_keyword('../data/enron_top_keywords_meaningful.txt')
    print('keywords length is {}'.format(len(keywordlist)))
    keywordlist_copy = copy.deepcopy(keywordlist)
    # corpus_processed = preprocess_corpus(init_cfg['corpusName'])
    injected_keyword_sets = choose_keyword(t_size)
    start_time = time.time()
    MarkovModel = textGenRestriction.buildMarchovModel(dir, init_cfg['corpusName'])
    end_time = time.time()
    print("training time for training n-gram model is {}".format(end_time-start_time))
    numOutputFile = len(injected_keyword_sets)
    print('length of injected_keyword_sets is {}'.format(numOutputFile))
    lenThreshold, lenTerminal = textGenRestriction.genExpRandomList(numOutputFile,text_len_para)
    genTexts = []
    for i in range(numOutputFile):
        print('generating text...{}'.format(i))
        print(injected_keyword_sets[i])
        genText = textGenRestriction.textGen(MarkovModel, injected_keyword_sets[i], keywordlist_copy, lenThreshold[i],
                                             lenTerminal[i], init_cfg['num_of_gram'])
        genText = pre_process_text(genText)
        genTexts.append(genText)
    return genTexts


def main_loop(looptime,t_size,text_len_para):
    " generate multi-files "

    for i in range(looptime):
        print('generating file {}......'.format(i))
        genTexts = main(t_size,text_len_para)
        writeMultiFile(genTexts,i,t_size)


if __name__ == '__main__':

    " single file generation "
    # t_size = 20     # num of inserted words
    # genTexts = main(t_size)
    # write2file(genTexts)
    # print(genTexts)

    " multi-files generation "
    " inject_20: text_len_para=0.015, add_on=50; inject_40: text_len_para=0.04; inject_60: text_len_para=0.1; inject_80: 0.16; inject_100: 0.2"
    start_time_1 = time.time()
    t_size = 40                 # num of keywords injected to each text
    text_len_para = 0.098    #to generate text with least length=80
    looptime = 1
    main_loop(looptime,t_size,text_len_para)
    end_time_1 = time.time()
    print("running time is {}".format(end_time_1-start_time_1))

    " test 1"
    # ****** test 1
    # groups = choose_keyword(20)
    # y = []
    # for item in groups:
    #     print(len(item))
    #     y.append(len(item))
    # print'*****'
    # print(min(y))

    " test 2 "
#     numOutputFile = 565
#     t = 0.04
#     lenThreshold, lenTerminal = textGenRestriction.genExpRandomList(numOutputFile,t)
