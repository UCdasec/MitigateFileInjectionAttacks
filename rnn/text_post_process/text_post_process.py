import sent_process
import nltk
import random, os,copy,re

from nltk.corpus import wordnet
import random
import nltk.stem as ns
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import filter




def read_file(filename):
    with open(filename,'r') as f:
        output = f.read()
    return output


def read_gentext(filename):
    with open(filename,'r') as f:
        output = f.readlines()
    return output


def join_file(input_list):
        input_file= " "
        input_file = input_file.join(input_list)
        print(input_file)
        return input_file


def read_keyword(keyword_file):
    with open(keyword_file,'r') as f:
        # output = f.readlines()
        output = f.read().splitlines()
    return output


def post_process_text(text):

    text = re.sub(r'(/m)', "", text)
    text = re.sub(r'(/f)', "", text)
    text = re.sub(r'(f/)', "", text)
    text = re.sub(r'/', "", text)
    text = re.sub(r'\[(?:[^\]|]*\|)?([^\]|]*)\]', r'\1', text)  # remove a pair of []
    text = re.sub(r'\((?:[^\]|]*\|)?([^\]|]*)\)', r'\1', text)  # remove a pair of ()
    text = re.sub('\d', "", text)  # remove number
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')

    return text


def choose_keyword0(keywordlist,n):
    " choose keyword in a random way"
    insert_keywords = random.sample(keywordlist,n)
    keywordlist = remove_insert_keywords(insert_keywords,keywordlist)
    return insert_keywords,keywordlist


def choose_keyword(t_size,filepath):
    '''
    choose in File-injection way
    Given a keyword pool and a parameter threshold T ( num of injected keywords in each sentence)
    divide all the keywords into K/T subsets, K total num of keywords
    and output the injected keywords for all the K/2T\logT + K/T files
    t_size: num inserted keyword in each sentence
    '''

    # dir = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))

    keyword_pool = read_keyword(filepath)
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


def remove_insert_keywords(insert_keywords,keywordlist):
    for key in insert_keywords:
        keywordlist.remove(key)
    return keywordlist


def extract_POS(input_file):
    token = nltk.word_tokenize(input_file)
    tags = nltk.pos_tag(token)
    # print(tags)
    dic = {}
    for word,tag in tags:
        if tag not in dic:
            dic[tag] = []
        dic[tag].append(word)

    return dic


def get_same_replace_word(insert_word,sent):
    sent_token = nltk.word_tokenize(sent)
    count = 0
    for token in sent_token:
        if token == insert_word:
            same_word = token
            break
        else:
            count += 1
    if count == len(sent_token):
        same_word = ''

    return same_word


def get_word_synonyms_from_sent(word, sent):
    word_synonyms = []
    word_stem = word_lemmatizer(word)
    sent_token = nltk.word_tokenize(sent)
    for synset in wordnet.synsets(word_stem):
        for lemma in synset.lemma_names():
            if lemma in sent_token and lemma != word:
                word_synonyms.append(lemma)
    # print('synonyms are {}'.format(word_synonyms))
    return word_synonyms


def get_word_antonym_from_sent(word,sent):
    word_antonyms = []
    word_stem = word_lemmatizer(word)
    sent_token = nltk.word_tokenize(sent)
    for synset in wordnet.synsets(word_stem):
        for lemma in synset.lemmas():
            if lemma.antonyms():
                # if lemma.antonyms()[0].name() in sent_token:
                    word_antonyms.append(lemma.antonyms()[0].name())
    # print('antonyms are {}'.format(word_antonyms))
    return word_antonyms


def check_substitution_list(keywrdlist,candidates):
    # if candidates is a sub-set of keywrdlsit
    # then return True
    i = 0
    for x in candidates:
        if x in keywrdlist:
            i += 1
    if i == len(candidates):
        flag = True
    else:
        flag = False
    return flag


def caps_I(sent):
    # replace i to I.
    sent_token = nltk.word_tokenize(sent)
    for i in range(len(sent_token)):
        if sent_token[i] == 'i':
            sent_token[i] = 'I'
    sent = (' '.join(sent_token))
    return sent


def word_lemmatizer(word):
    "stem words"
    lc_stemmer = lc.LancasterStemmer()
    lc_stem = lc_stemmer.stem(word)
    return lc_stem


def word_replace(old,new,sent):
    sent_tokens = nltk.word_tokenize(sent)
    for i in range(len(sent_tokens)):
        if sent_tokens[i] == old:
            sent_tokens[i] == new
            break
    sent = ' '.join(sent_tokens)
    return sent


def visual_similar_word(word, sent_token, len_threshold, similar_threshold):
    "1st and last letter should be exact same"
    visual_similar_word = []
    if len(word) < len_threshold:
        return visual_similar_word,sent_token
    else:
        for word_sent in sent_token:
            count = 0
            length = len(word)
            # if word_sent[0] == word[0] and word_sent[length - 1] == word[length - 1]:
            if len(word_sent) == length:
                if word_sent[0] == word[0]:
                    for i in range(length):
                        if word_sent[i] == word[i]:
                            count += 1
                    similar_ratio = float(count)/float(length)
                    if similar_ratio >= similar_threshold:
                        similar_word_tupe = (word_sent,similar_ratio)
                        visual_similar_word.append(similar_word_tupe)
        visual_similar_word = sorted(visual_similar_word,key=lambda x:(x[1]))
        if visual_similar_word:
            most_visual_similar_word = visual_similar_word[0][0]
            sent_token.remove(most_visual_similar_word)
        else:
            most_visual_similar_word = visual_similar_word
        return most_visual_similar_word, sent_token


def replacement(keywordlist,sent,len_word, similar_thresh):
    """

    :param keywordlist:
    :param sent:
    :param len_word: the least num of letters consider in the visually similar checking function
    :param similar_thresh: the similar threshold between two words
    :return:
    """
    i = 0
    j = 0
    e = 0
    f = 0
    sent = caps_I(sent)
    rest_word = []
    sent_tokens = nltk.word_tokenize(sent)
    for word in keywordlist:
        sent_tags = extract_POS(sent)
        allkey = sent_tags.keys()
        word_synonyms = get_word_synonyms_from_sent(word,sent)
        word_antonyms = get_word_antonym_from_sent(word,sent)
        most_visual_similar_word,sent_tokens = visual_similar_word(word,sent_tokens,len_word,similar_thresh)
        same_word = get_same_replace_word(word,sent)
        if len(same_word) != 0:
            sent = sent
            print('the same word is {}.'.format(same_word))
            i += 1
            f += 1
        elif word_synonyms != []:
            # sent = sent.replace(random.choice(word_synonyms),word,1)
            sent = word_replace(random.choice(word_synonyms),word,sent)
            i += 1
        elif word_antonyms:
            # sent = sent.replace(random.choice(word_antonyms), word, 1)
            sent = word_replace(random.choice(word_antonyms), word,sent)
            i += 1
        elif most_visual_similar_word:
            sent = word_replace(most_visual_similar_word,word,sent)
            i += 1
            e += 1
            print('"{}" in sentence can be replaced by inserted word "{}" given they are visually similar'.format(most_visual_similar_word,word))
        else:
            tmp = []
            tmp.append(word)
            tag = nltk.pos_tag(tmp)[0][1]
            if tag in allkey:
                candidates = sent_tags[tag]
                if not check_substitution_list(keywordlist,candidates):
                    if candidates is not None:
                        substitute = random.choice(sent_tags[tag])
                        while substitute in keywordlist:
                            substitute = random.choice(sent_tags[tag])
                        sent = sent.replace(substitute,word,1)
                        i += 1
                else:
                    # sent = sent + ' ' + word
                    rest_word.append(word)
                    j += 1
            else:
                # sent = sent + ' ' + word
                rest_word.append(word)
                j += 1
    print('the rest word is: {}'.format(rest_word))
    sent_rest = sent_process.main(' '.join(rest_word))
    print('the re-build sentence for rest-words is: {}.'.format(sent_rest))
    sent = sent + ' ' + sent_rest
    print('There are {} times of replacement.'.format(i))
    print('There are {} times of same word replacement.'.format(f))
    print('There are {} times of visual_similar replacement.'.format(e))
    print('There are {} rest words need to be re-processed.'.format(j))
    return sent


def write2file(replace_texts):
    items = os.listdir(os.getcwd() + '/result')
    output = 'replaced_texts.txt'
    if output in items:
        os.remove('./result/' + output)
    for text in replace_texts:
        with open('./result/' + output,'a') as f:
            f.write(text + '\n')


def writeMultiFile(replace_texts,i,num_injected_keyword):
    items = os.listdir(os.getcwd() + '/result')
    output = 'replaced_texts_' + str(num_injected_keyword) + '_' + str(i) + '.txt'
    if output in items:
        os.remove('./result/' + output)
    for text in replace_texts:
        with open('./result/' + output, 'a') as f:
            f.write(text + '\n')


def main_test():
    test_path = '../data/test'
    items = os.listdir(test_path)
    items.remove('.DS_Store')
    keywordlist = read_keyword('../data/enron_top_keywords_meaningful.txt')
    i = 0
    for item in items:
        if item != '.DS_Store':
            i += 1
            print('************************* {} ************************'.format(i))
            text = read_file(test_path + '/' + item)
            insert_kwrds,keywordlist = choose_keyword(keywordlist,20)
            print('the length of keywords_list is: {}'.format(len(keywordlist)))
            new_text = replacement(insert_kwrds, text)
            caps_I(new_text)
            print('\n', insert_kwrds, '\n')
            print('Original text is: "\n"  {}'.format(text))
            print('-' * 30)
            print('Replaced text is: "\n"  {}'.format(new_text))
            print('\n')


def main(replaced_filepath,inject_keyword_sets,text_len_threshold):
    "n is the num of files that need to be input"
    gentexts = read_gentext(replaced_filepath)
    num_output_file = len(inject_keyword_sets)
    print('num of input is {}'.format(len(gentexts)))
    print('num of output is {}'.format(num_output_file))
    len_word = 4                                                        # least_length  for the visually_similar word
    similar_thresh = 0.6                                                # for the visually_similar word similar rate check
    i = 0
    replace_texts = []
    for text in gentexts:
        if len(text) > text_len_threshold:
            i += 1
            if i < num_output_file:
                print('replacing text...{}'.format(i))
                text = post_process_text(text)
                new_text = replacement(inject_keyword_sets[i], text, len_word, similar_thresh)
                caps_I(new_text)
                replace_texts.append(new_text)
                print('\n', inject_keyword_sets[i])
                print('Original text is: "\n"  {}'.format(text))
                print('-' * 30)
                print('Replaced text is: "\n"  {}'.format(new_text))
                print('\n')
            else:
                break
    return replace_texts


def main_loop(looptime,keyword_filepath,num_injected_keyword,text_len_threshold):

    for i in range(looptime):
        print('generating file {}.......'.format(i))
        inject_keyword_sets = choose_keyword(num_injected_keyword,keyword_filepath)
        replace_texts = main(input_textpath, inject_keyword_sets, text_len_threshold)
        writeMultiFile(replace_texts,i,num_injected_keyword)


if __name__ == '__main__':

    num_injected_keyword = 10
    text_len_threshold = 34                     # the least length of text for replacing function. to filter the unqualified raw texts
    # keyword_filepath = '../data/top_keyword_science_remove_stopword.txt'
    keyword_filepath = '../data/enron_top_keywords_meaningful.txt'

    " case1: single file generation "
    # inject_keyword_sets = choose_keyword(num_injected_keyword,filepath)
    # generated_textpath = '../data/model_rc_epoch100_batchsize128/generated_texts_num500_len50.txt'
    # replace_texts = main(generated_textpath,inject_keyword_sets)
    # write2file(replace_texts)


    " case2: mutil-files generation "
    looptime = 5
    input_textpath = '../data/amazon_review/epoch40/gen_texts_1650_34.txt'
    main_loop(looptime,keyword_filepath,num_injected_keyword,text_len_threshold)
