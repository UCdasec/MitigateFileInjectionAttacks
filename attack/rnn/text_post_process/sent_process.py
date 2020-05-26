import nltk
import random

token = ['test']

def raise_error(exception):
    pass


def gen_POS(token):
    tags = nltk.pos_tag(token)
    return tags


def extract_POS(input_file):
    token = nltk.word_tokenize(input_file)
    tags = nltk.pos_tag(token)
    dic = {}
    for word,tag in tags:
        if tag not in dic:
            dic[tag] = []
        dic[tag].append(word)
    return dic


def parse_interjection(dic):
    """
    extract words POS_UH like "wow..."
    :param dic:
    :return:
    """
    if 'UH' in dic:
        uh = dic['UH']
        return uh
    else:
        raise_error('UH list is empty.')


def parse_determiner(dic):
    """
    extract words like 'the, a ,no'
    :param dic:
    :return:
    """
    tag_dt = ['DT','PDT']
    dts = []
    for key in tag_dt:
        if key in dic:
            dts += dic[key]
    if dts:
        return dts
    else:
        raise_error('Determiner list is empty.')


def parse_adjective(dic):
    tag_jj = ['JJ','JJS','JJR']
    jj = []
    for key in tag_jj:
        if key in dic:
            jj += dic[key]
    if jj:
        return jj
    else:
        raise_error('Adjective list is empty.')



def parse_subject(dic):
    tag_sub = ['NN','NNS','NNP','NNPS','PRP']
    subs = []
    for key in tag_sub:
        if key in dic:
            subs += dic[key]
    if subs:
        return subs
    else:
        raise_error('Subject list is empty.')


def parse_predicate(dic):
    tag_pred = ['VB','VBD','VBG','VBN','VBP','VBZ']
    pred = []
    for key in tag_pred:
        if key in dic:
            pred += dic[key]
    if pred:
        return pred
    else:
        raise_error('Predicate list is empty')


def parse_conjunction(dic):
    tag_conj = ['RP','TO','IN','WDT','WP']
    conj = []
    for key in tag_conj:
        if key in dic:
            conj += dic[key]
    if conj:
        return conj
    else:
        raise_error('Conjunctin list is empty.')


def parse_object(subjects):
    """
    share with parse_subject
    :param subjects:
    :return:
    """
    objs = subjects
    if objs:
        return objs
    else:
        raise_error('Object list is empty.')


def parse_comma(dic):
    tag_mark = ['.',',']
    marks = []
    for key in tag_mark:
        if key in dic:
            marks += dic[key]
    if marks:
        return marks
    else:
        raise_error('Marks list is empty.')


def concatenate_word(pos,sent):
    if pos:
        pos_temp = random.choice(pos)
        pos.remove(pos_temp)
        sent.append(pos_temp)
    return sent


def test_empty(lists):
    i = 0
    for list in lists:
        if list:
            i += 1
            return True
            break
    if i == 0:
        return False


def concatnate_words(uh,dts,jj,subs,pred,conj,objs,marks):
    sent = []
    pos_group = [uh,dts,jj,subs,pred,conj,dts,jj,objs,marks]
    flag = test_empty(pos_group)
    while flag:
        for pos in pos_group:
            sent = concatenate_word(pos,sent)
            flag = test_empty(pos_group)
    return sent


def main(input):

    dic = extract_POS(input)
    # print(dic)
    uh = parse_interjection(dic)
    dts = parse_determiner(dic)
    jj = parse_adjective(dic)
    subs = parse_subject(dic)
    pred = parse_predicate(dic)
    conj = parse_conjunction(dic)
    objs = parse_object(subs)
    marks = parse_comma(dic)
    sent = concatnate_words(uh,dts,jj,subs,pred,conj,objs,marks)
    sent = ' '.join(sent)
    return sent
    # print(sent)



if __name__ == '__main__':
    input = 'lawyer seen property operational determine.'
    main(input)