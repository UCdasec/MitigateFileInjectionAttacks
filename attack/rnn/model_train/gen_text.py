from textgenrnn import textgenrnn
import nltk
import random
import os,re,time,sys,argparse



textgen = textgenrnn(weights_path='./models/amazon_part1_3layer_epoch50_weights.hdf5',vocab_path='./models/amazon_part1_3layer_epoch50_vocab.json',
                     config_path='./models/amazon_part1_3layer_epoch50_config.json')


def prefix_choose():
    "char_level, choose one letter randomly"
    letter = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    prefix = random.choice(letter)
    return prefix


def check_sent_length(gen_texts_raw, gen_texts_filtered, num_text_threshold, len_text_threshold):
    count = len(gen_texts_filtered)
    flag = True
    for i in range(len(gen_texts_raw),0,-1):    # retrieve word in a reverse order, because there is a remove function which change the order of the list each time
        text_token = nltk.word_tokenize(gen_texts_raw[i-1])
        if len(text_token) < len_text_threshold:
            gen_texts_raw.remove(gen_texts_raw[i-1])
        else:
            count += 1
    if count >= num_text_threshold:
        flag = False
    else:
        flag = True
    return flag, gen_texts_raw


def text_gen(num_text_threshold, len_text_threshold):
    num_sent = 0
    gen_texts_filtered = []
    flag = True
    while(flag):
        num_sent += 3
        prefix_letter = prefix_choose()
        gen_texts_raw = textgen.generate(n = num_sent,return_as_list= True, prefix=prefix_letter,temperature=0.8,max_gen_length=1000000)
        flag,gen_texts_filtered_eachtime = check_sent_length(gen_texts_raw, gen_texts_filtered, num_text_threshold, len_text_threshold)
        gen_texts_filtered += gen_texts_filtered_eachtime
        print('generating text...{}'.format(len(gen_texts_filtered)))
        num_sent = 0
    return gen_texts_filtered


def text_gen_realtime(num_text_threshold, len_text_threshold):
    num_sent = 0
    gen_texts_filtered = []
    flag = True
    while(flag):
        num_sent += 3
        prefix_letter = prefix_choose()
        gen_texts_raw = textgen.generate(n = num_sent,return_as_list= True, prefix=prefix_letter,temperature=0.6,max_gen_length=1000000)
        flag,gen_texts_filtered_eachtime = check_sent_length(gen_texts_raw, gen_texts_filtered, num_text_threshold, len_text_threshold)
        gen_texts_filtered += gen_texts_filtered_eachtime
        print('generating text...{}'.format(len(gen_texts_filtered)))
        num_sent = 0
        write2file_realtime(gen_texts_filtered_eachtime,num_text_threshold,len_text_threshold)
    return gen_texts_filtered


def post_process_text(text):
    text = re.sub('(/m)', r'', text)
    text = re.sub('(/f)', r'', text)
    text = re.sub('(f/)', r'', text)
    text = re.sub('(m/)', r'', text)
    text = re.sub('/', r'', text)
    text = re.sub('\[(?:[^\]|]*\|)?([^\]|]*)\]', r'1',text) # remove a pair of []
    text = re.sub('\((?:[^\]|]*\|)?([^\]|]*)\)', r'1',text) # remove a pair of ()
    text = re.sub('\d',r'', text) #remove number
    text = text.replace('(','')
    text = text.replace(')','')
    text = text.replace('[','')
    text = text.replace(']','')
    return text


def write2file(gen_texts_filtered,num_texts,len_text):
    items = os.listdir(os.getcwd())
    output = 'generated_texts_' + str(num_texts) +'_'+ str(len_text) + '.txt'
    if output in items:
        os.remove(output)
    for text in gen_texts_filtered:
        text = post_process_text(text)
        with open(output,'a') as f:
          f.write(text + '\n')


def write2file_realtime(gen_text,num_texts,len_text):
    output = 'gen_texts_' + str(num_texts) +'_'+ str(len_text) + '.txt'
    out_path = './output_texts/'
    def write_file(text):
        with open(out_path + output, 'a+') as f:
            f.write(text + '\n')
            f.flush()
    if len(gen_text) == 0:
       pass
    else:
    	for text in gen_text:
            write_file(text)



def main(num_text_threshold,len_text_threshold):

    gen_texts_filtered = text_gen(num_text_threshold,len_text_threshold)
    write2file(gen_texts_filtered,num_text_threshold,len_text_threshold)


def main_realtime_writing(num_text_threshold,len_text_threshold):
    gen_texts_filtered = text_gen_realtime(num_text_threshold,len_text_threshold)


if __name__ == '__main__':
    print("'gen_texts.py' is running ...")
    start_time = time.time()
    arg1 = int(sys.argv[1]) 	# num of texts
    arg2 = int(sys.argv[2])		# least length of each text
    #main(arg1,arg2)        		# n is the num of texts that need to be generated; m is the least length of the each generated text
    main_realtime_writing(arg1,arg2)
    end_time = time.time()
    print('The running time is {} minutes.'.format(str(float(end_time - start_time)/60.0)))
