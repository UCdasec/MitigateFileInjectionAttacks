
#!/usr/bin/python
#coding:utf-8

# Boyang Wang
# University of Cincinnati 
# boyang.wang@uc.edu
# 10/04/2017

# from __future__ import division
import os, re, operator, nltk, random
from collections import Counter
from math import *
import shutil

# the following functions are used in step1()
def retrieve_files(dir, end_str, files):
     """ 
     given a dir, append all pathes of the files end with end_str
     passing files as a reference
     """
     # print dir
     items = os.listdir(dir)
     
     for item in items:
         if item.endswith(end_str):
             files.append(dir+'/'+item) 
             
     #for file in files:
     #    print file
     
     
def check_folder(dir, folder_name): 
    """
    given a dir, if there is a folder_name under dir
    return true, otherwise return false 
    """  
    
    items = os.listdir(dir)
    if folder_name in items: 
        print(' has a folder named: ' + folder_name)
        return True
    else:
        print(' does not have a folder named: ' + folder_name)
        return False  
  

def retrieve_folders(dir): 
    """
    given a dir, return all the pathes of all the folders under dir 
    """
    print(dir)
    items = os.listdir(dir)
    folders = []
    for item in items:
        print(item )
        if os.path.isdir(dir+'/'+item):			#what means of preconditions in if 
            folders.append(dir+'/'+item) 
    return folders
     
     
def retrieve_all_files(dir, folder_name, end_str, files):
    """
    given a dir, if a folder under dir has a child folder named folder_name, 
    retrieve all the files end with end_str under that child folder 
    """
    
    folders = retrieve_folders(dir)				#obtain all folders in dir
    print(folders)
    valid_folders = []
    for folder in folders:
        if check_folder(folder, folder_name):	#obtain sub-folders of folder,if folder_name in sub-folders
            valid_folders.append(folder)
            
    print ('*' * 30)								#print 30 *
    print ('users with ' + folder_name)
    for valid_folder in valid_folders:
        print (valid_folder)
    print ('No of valid users: ')
    print (len(valid_folders))
    
    for valid_folder in valid_folders: 
        retrieve_files(valid_folder+'/'+folder_name, end_str, files)
    
    # print '*' * 30 
    # for file in files: 
    #    print file     
    #
    print ('No of files in total: ' )
    print (len(files)    )


def write_directories_to_file(output_file, directories):
    """
    write a set of directories to an output_file
    passing directories as a reference
    """
    # print len(directories)
    file = open(output_file, 'w')
    print ('open file correctly' )
    for directory in directories:
        file.write(directory + '\n')
    print ('No of lines has been written')
    print (len(directories) )
    file.close()   
            

# the following functions are used in step2()
def read_enron_email(file_name, start_line_number): 	#start_line_number from which line start
    """
    Given an email named file_name in enron dataset, read stencence from this email without metadata 
    i.e., the content of an email only, append content to data 
    """   
    
    # file = open(file_name, 'r')
    # print file.read()
    with open(file_name, 'r') as file: 
        lines = file.readlines()
    data = []
    count = 1								#count the number of line, once beyond start_line_number, start append to data 
    for line in lines:                      #each line represent one line in an singal email
        line.strip()						#delete 'line breaks'
        if count > start_line_number:
            # print count, line
            data.append(line)
        count = count + 1
    file.close()
    return data


def read_enron_emails(input_file, start_line_number,data):
    """
    Given an input file with a list of all the email file names obtained from step1()
    read the content of each email without metadata, and append all the content to data 
    """

    with open(input_file, 'r') as file:
        lines = file.readlines()

    count = 1
    for line in lines:
        print (count, line.strip())
        read_enron_email(line.strip(), start_line_number,data)
        count = count + 1

    file.close()



def read_write_enron_emails(input_file, start_line_number):
    """
    Given an input file with a list of all the email file names obtained from step1()
    read the content of each email without metadata, and append all the content to data 
    """
    dir = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))
    items = os.listdir(dir)
    with open(input_file, 'r') as file:
         lines = file.readlines()
    if 'legitimateUserMails' not in items:                                                                                       #check if email4test folder is in the parent folder
        os.mkdir(dir + '/legitimateUserMails')
    items0 = os.listdir(dir + '/legitimateUserMails')
    if items0 is not None:
        shutil.rmtree(dir + '/legitimateUserMails')
        os.mkdir(dir + '/legitimateUserMails')
    count = 1
    for line in lines:                                                                                                  #each line represent one mail
        print (count, line.strip())
        msj = read_enron_email(line.strip(), start_line_number)

        # outputName = line.strip()[100:-1].replace('/_sent_mail/','_')
        # with open(dir + '/legitimateUserMails/' + outputName + '.txt', 'w') as file:

        with open(dir + '/legitimateUserMails/' + 'Mail_' + str(count) + '.txt', 'w') as file:
            for x in msj:
                file.write(x)
        count = count + 1
    file.close()
    print ('number of outputfiles ', count)
    print ('Corpus has been writted in file')
   
      
def write_corpus_to_file(output_file, corpus):
    """
    Given corpus, i.e., a list of data (lines), write them to an outputfile
    """    
    
    file = open(output_file, 'w')
    for line in corpus: 
        file.write(line)
    print ('Corpus has been writted in file')
    file.close() 
    

# the following functions are step3 
def read_corpus_from_file(input_file): 
    """
    Given input_file, read/load corpus 
    """       
    
    print ('reading corpus')
    file = open(input_file, 'r')
    corpus = file.read()
    return corpus 
    
    
def preprocess_corpus(corpus):  
    """
    Given corpus, remove space, keep words only, stem and remove common words
    """ 
    
    # print 'preprocessing words'
    # remove space
    # text = re.findall(r'\w+', corpus) # for [a-zA-Z0-9_]
    text = re.findall(r'[a-zA-Z]+', corpus)   # for [a-zA-Z] keep words only no numbers and '_'   
    words = [w.lower() for w in text]
    # print words 
    
    # stemmer based on existing ones in the current list
    lemma = nltk.WordNetLemmatizer()			#extract the original word pattern
    lemmed_words = [lemma.lemmatize(w) for w in words]
    
    # tag lemmed_words
    tagged_words = nltk.pos_tag(lemmed_words)
    # print tagged_words 
    
    processed_words = []
    tag_list = ['CC', 'DT', 'EX', 'IN', 'MD', 
                'PDT', 'POS', 'PRP', 'PRP$', 'TO', 
                'WDT', 'WP', 'WRB']
    for word, tag in tagged_words:
       if tag in tag_list:
           pass    
       else: 
          processed_words.append(word)
    
    return processed_words
    
    
def find_frequent_words(words, most_frequent):
    """
    Given a list of words, calculate the frequency of each unique word, 
    and keep the most_frequent one
    """   
     
    # common_words = Counter(sorted(words))
    # print common_words
    common_words = Counter(sorted(words)).most_common(most_frequent)
    print (common_words )
    most_common_words = [w for w, w_count in common_words]
    return most_common_words
    
    
# the following are functions in step4  
def read_keywords_from_file(file_name):
    '''
    Given a file_name, which has a list of keywords, read all the keywords
    '''
    
    print ('reading keywords...')
    file = open(file_name, 'r')
    corpus = file.read()
    
    text = re.findall(r'[a-zA-Z]+', corpus)   # for [a-zA-Z] keep words only no numbers and '_'
    print ('keyword pool as: ', text)
    
    return text

  
def list_keywords_infile(file_name, pool):
    '''
    Given a pool of keywords and a file_name, return keywords that are in this file
    '''    
    
    corpus = read_corpus_from_file(file_name)
    words = preprocess_corpus(corpus)
    unique_words = set(words)
    # print unique_words
    # print type(unique_words)
    
    keywords = []
    for u in unique_words:
        if u in pool: 
           keywords.append(u)
    return keywords
    
    
def list_keywords_infiles(input_file, pool):     
    '''
    Given a pool of keywords and an input_file, which includes multiple file names
    list the keywords of each file, return a pair list, where each pair is (id, keywords) 
    '''
    
    with open(input_file, 'r') as file: 
        lines = file.readlines()
    
    file.close()
    
    file_set = []
    keywords_set = []
    
    count = 1
    for line in lines: 
        file_name = line.strip()
        keywords = list_keywords_infile(file_name, pool)
        # print count, file_name, keywords
        file_set.append(line.strip())
        keywords_set.append(keywords)
        count = count + 1
    
    file_keywords_pairs = zip(file_set, keywords_set)
    return file_keywords_pairs


def set_unique_element_count(elements):
    '''
    given a set of elements, where each element is the number of keywords in a file
    first build a set of pairs, where pair.a is a unique element, and pair.b is its 
    counter in the set of elements, sorted based on counter before the end of this 
    function
    '''    
    
    unique_elements = []
    unique_counts = []
    for element in elements: 
        if element in unique_elements: 
            ind = unique_elements.index(element) 
            unique_counts[ind] = unique_counts[ind] + 1
        else: 
            unique_elements.append(element)
            unique_counts.append(1)
    
    total = len(elements)
    pmfs = []
    cdfs = []
    for count in unique_counts:
        pmf = count/float(total) 
        pmfs.append(pmf)         
    
    pairs = zip(unique_elements, unique_counts, pmfs)
    # sort based on itemgetter(1) first, then if equal based on itemtter(0)
    # in this case, itemgetter(1) is counter, 
    # itemgetter(0) is element
    pairs.sort(key=operator.itemgetter(0), reverse=True)        #according to the 1st element of subset to order
    # for pair in pairs:
    #    print pair
    
    cdfs = []
    for i, pair in enumerate(pairs): 
        if i == 0: 
           cdf = pair[2]
        else:
           cdf = pair[2] + cdfs[i-1]
        cdfs.append(cdf)
    
    tuples = zip(pairs, cdfs)
    for tuple in tuples:
        print (tuple)
            
    return tuples
    

def calculate_set_mean(elements):    
    '''
    given a set of elements, output mean of this set
    '''
    
    sum = 0.0
    count = len(elements)
    for element in elements: 
        sum = sum + element
    
    mean = sum/count
    print (sum, mean, count)
    return mean


def calculate_set_variance(elements, mean):   
    '''
    given a set of elements and its mean, output variance
    '''
 
    sum = 0.0 
    count = len(elements)
    for element in elements: 
        sum = sum + (element - mean)**2 
    var = sum/count
    
    print (sum, var, count)
    return var
        
             
def generate_inverted_index(keyword_pool, pairs):

    id_set = []
    count = 0
    for w in keyword_pool: 
        ids = []
        for pair in pairs:
            if w in pair[1]:  
                # print pair[0]
                ids.append(pair[0])
        count = count + 1 
        # print count, w, ids
        id_set.append(ids) 
    
    items = zip(keyword_pool, id_set)
    # sort it based on keyword
    items.sort(key=operator.itemgetter(0), reverse=False)
    # for item in items:
    #   print item
        
    return items         

def write_index_to_file(output_file, items):
    """
    Given an inverted index, write all the items to a file
    """    
    
    file = open(output_file, 'w')
    for item in items: 
        str0 = str(item[0])
        str1 = '  '.join(str(x) for x in item[1])
        file.write( str0 + '  ' + str1 + '\n') 
        # file.write(item)
    print ('An inverted index has been writted in file')
    file.close() 
   

# the following functions will be used for step7

def group_keywords(keyword_pool, threshold): 
    '''
    Given a set of keywords, divide it into subgroups based on threshold T 
    '''             
    
    k_size = len(keyword_pool)  # by default, it should be 5000
    # reshuffle keyword_pool, since it might be sorted
        # in practice, we should shuffle, here we sorted for easy debug and observation
    random.shuffle(keyword_pool)
    # print 'After shufflering the order, the keyword pool as ', keyword_pool
    # keyword_pool.sort()
    print (keyword_pool)
    groups = []
    group = []
    count = 1  
    for i, w in enumerate(keyword_pool): 
        if i < (count * threshold): 
           group.append(w)
        # print i, count, w   
        if ((i+1)%threshold == 0) or (i == k_size - 1):
           count = count + 1
           groups.append(group)
           group = []     
     
    # for g in groups:
    #     print g
    # print k_size, threshold
       
    return groups 
 
    
def pair_groups(groups): 
    '''
    Given a group of keyword_sets, pair every two of them to form a new group 
    and return a new group of pairs
    '''           
    g_size = len(groups)
    
    group_pairs = []
    pair = []

    for i, group in enumerate(groups):
       # print i, group
       for g in group: 
          pair.append(g)
       if (i%2 == 1) or (i == g_size - 1):
          group_pairs.append(pair) 
          pair = []          
    
    #for p in group_pairs:
    #   print p 
       
    return group_pairs
 
 
def get_inject_keywords(keywords, inject_groups): 
    '''
    Given a group of keywords, with a size of 2*T (or less, e.g., the last one), 
    where T is the threshold, output a logT sets for file-injection attack, 
    where each set has exactly (or less) T keywords 
    ''' 
    
    k_size = len(keywords)
    degree = int(ceil(log(k_size, 2)))
    # print k_size, degree
    
    bins = get_bins(k_size, degree)
    v_bins = get_vertical_bins(bins, degree)
    
    # inject_groups = []
    for x in range(0, degree):
        inject_words = []
        for y in range(0, k_size):
            # print v_bins[x][y]
            if (v_bins[x][y] == 1):
                # print keywords[y]
                inject_words.append(keywords[y])   
        inject_groups.append(inject_words)
    
    # for g in inject_groups:
    #    print g
    

def int_to_bin(a, degree): 
    '''
    Given an integer a, present it as a binary sequence bin with a number of degree bits
    '''
    
    bin = [0] * degree  
    temp = a 
    for i, b in enumerate(bin):
       if (temp == 0):
           break  
       reminder = temp%2
       bin[(degree-1)-i] = reminder
       # print reminder, temp
       temp = temp/2   
    # print bin
    return bin 
    
    
def get_bins(size, degree): 
    '''
    Given an integer size and a degree, output the first $size$ bins among 
    all the 2^degree bins, where the length of each bin is degree
    e.g, given degree 3 and size = 4
    will output { (0, 0, 0)
                  (0, 0, 1)
                  (0, 1, 0)
                  (0, 1, 1)}   
    '''
    
    bins = [] 
    for i in range(0, size):
       bin = int_to_bin(i, degree)
       bins.append(bin)
    # for i, b in enumerate(bins):
       # print i, b
    
    return bins


def get_vertical_bins(bins, degree):
    '''
    Given a set of bins and degree, where each bin has a length of degree, 
    output a set of vertical bins, where each bin has a length of size(bins)
    e.g given degree = 3 and size = 4
    and  { (0, 0, 0)
           (0, 0, 1)
           (0, 1, 0)
           (0, 1, 1)}      
    will output 
         {(0,0,0,0)
          (0,0,1,1)
          (0,1,0,1)}    
    '''
    
    size = len(bins)
    v_bins = []
    for i in range(0, degree):
       v_bin = [bin[i] for bin in bins]
       v_bins.append(v_bin)
    # for v in v_bins:
    #   print v     
        
    return v_bins        