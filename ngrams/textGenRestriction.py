from collections import Counter
#import matplotlib.pyplot as plt
import numpy as np
import os,time,random,copy,shutil,sys,nltk,math
import n_gramAuto


def randomChooseKeywords(dir,n):
    """

    :param dir:
    :param n:
    :return:
    """

    inputFile = 'enron_top_keywords_meaningful.txt'
    outputFile = 'injected_keyword_' + str(n+1) + '.txt'
    with open(dir + '/data/' + inputFile,'r') as f:
        keywordPool = f.readlines()
    # print 'length of input keyword pool is: ',len(keywordPool)
    injectedKeywords = random.sample(keywordPool,20)
    text = ' '.join(w.strip() for w in injectedKeywords)
    with open(dir + '/test/' + outputFile,'w') as f:
        f.write(text)
    return text


def readInjectedKeyword(dir, inputPath):
    """

    :param dir:
    :param iuputPath:
    :return: injectedKeywordsGroup: each subset in Group refer to the keywords parsed from one file, there may a lot of files
    """
    injectedKeywordsGroup = []
    allInjectedKeywords = []
    items = os.listdir(dir + '/' + inputPath)
    for item in items:
        if not item.endswith('.txt'):
            os.remove(dir + '/' + inputPath + '/' + item)
    items.sort(key = lambda x: int(x[17:-4]))
    count = 0
    for item in items:
        if item.endswith('txt'):
            count += 1
            with open(dir + '/' + inputPath + '/' + item,'r') as f:
                injectedfile = f.read()
            injectedKeywords = injectedfile.strip().split(' ')
            injectedKeywordsGroup.append(injectedKeywords)                                                              #append each input file as a words sub-list in a list group
            "append all injected keywords in one list"
            for word in injectedKeywords:
                allInjectedKeywords.append(word)
    allInjectedKeywords = list(set(allInjectedKeywords))                                                                #remove repeated words
    print ('num of input files: ', count)
    return injectedKeywordsGroup, allInjectedKeywords



def buildMarchovModel(dir, corpusName):
    """
    train a Marchov model MM for input corpus
    :param corpusName:
    :return:
    """
    # inputPath = dir + corpusName
    MarchovChain = n_gramAuto.MarkovChain(2)
    MM = MarchovChain.learn(MarchovChain.preprocess_corpus(corpusName))
    print ('Marchov Model is built successfully.')
    print ('\n')
    return MM


def excludeInjectedKeywords(allInjectedKeywords,injectedKeywords):
    """
    remove the injectedKeywords from allInjectedKeywords
    :param allInjectedKeywords: all input file which present within one words list
    :param injectedKeywords: each one input file whicn present as words list
    :return: excludeInjectedOne
    """
    allInjectedKeywordsCopy = copy.deepcopy(allInjectedKeywords)
    for i in range(len(allInjectedKeywordsCopy)-1,-1,-1):                                                                   #should be recycle in reverse order, since the index of the list would be changed dynamicly in the positive order
        if allInjectedKeywordsCopy[i] in injectedKeywords:
            allInjectedKeywordsCopy.pop(i)
    excludeInjectedOne = allInjectedKeywordsCopy
    return excludeInjectedOne


def preprocessWords(words):
    """
    output words that have removed the common words that in that tag_list
    :param words: words should be list type
    :return:
    """
    removeComWords =[]
    taggedWords = nltk.pos_tag(words)
    tag_list = ['CC','EX', 'IN', 'MD',
                'PDT', 'POS', 'PRP$', 'TO', 'DT'
                'WDT', 'WP', 'WRB', 'VBP', 'VBZ', '.',',' ':']    #remove  PRP:I you us;  ','
    for word, tag in taggedWords:
        if tag in tag_list:
            continue
        else:
            removeComWords.append(word)
    return removeComWords


def genExpRandomList(numInputFile, text_len_para):
    """
    generate inters which based on exponential distribution using for length of generated mails
    :param numInputFile: must greater than 10; the one decide the size of the number of exponential distribution
    :return: lenThreshold and lenTerminal
    lenThreshold: ideal length for each generated text should be
    lenTerminal: the final length threshold for each generated text
    """
    lenThreshold0 = []
    lenThreshold = []
    #lambd = 0.43
    # lambd = 0.1                                                        # control the length of output text
    step = 0.02
    numOutput = numInputFile * step
    x = np.arange(0, numOutput, step)                                   # step=0.1, range(0,numOutput), so total number is numOutput*10
    y = text_len_para * np.exp(-text_len_para * x)
    y = y.tolist()
    for v in y:
        v = int(math.ceil(v * 1000))                                    # amplify 1000, so the maxmum is 430(lambd*1000)
        lenThreshold0.append(v)                                         # range(0,430)
    addOn = [50] * numInputFile                                         # the minimal value of lenThreshold should be greater than the length of injected keywords(20)
    for i,j in zip(lenThreshold0,addOn):
        summ = i + j
        lenThreshold.append(summ)                                       # range(21,430+21)
    lenTerminal0 = [30] * numInputFile
    lenTerminal = []
    for p,q in zip(lenTerminal0,lenThreshold):
        summ1 = p + q
        lenTerminal.append(summ1)
    print ('length threshold ', lenThreshold, '\n', 'length terminal ', lenTerminal)
    print ('the length of the lenThreshold is:{} '.format(len(lenThreshold)))
    # plt.figure(1)
    # plt.plot(x,lenThreshold)
    # plt.show()

    return lenThreshold, lenTerminal


def getNextword(inputKey, MM, injectedKeywords, excludeInjectedOne, reducedInjectedKeywords, usedWdsInKeywds, usedWdsNotKeywds):
    """
    find the next word for inputKey based on the MM (Marchov chain)
    :param inputKey: initial input for n-gram
    :param MM: Marchov chain
    :param injectedKeywords: inputfile, which includes keydwords need to be injected, parsed as words in list
    :param excludeInjectedOne: exclude 'injectedKeywords' from all injected keywords
    :param reducedInjectedKeywords: exclude injected Keyword that has no next words based on current 'injectedKeywords'
    :param usedWdsNotKeywds: store words that have been used and all those words are not in injected keywords
    :return: next word for inputKey
    """
    possibleNextWords = MM.get(inputKey)
    "remove inputkey that has been used"
    if type(inputKey) is str:
        if inputKey in reducedInjectedKeywords:
            reducedInjectedKeywords.remove(inputKey)    #later use for storing keywords that have no nextwords
    else:   #type is tuple ('input',),inputKey[0]='input'. based on 2-gram, just consider one word that if it is in the 'reducedInjectedKeywords'
        if inputKey[0] in reducedInjectedKeywords:
            reducedInjectedKeywords.remove(inputKey[0])
    "store inputkey that has been used"
    # usedKeywords = []
    if type(inputKey) is str:
        if inputKey in injectedKeywords:
            usedWdsInKeywds.append(inputKey)  # later use for storing keywords that have been used
    else:  # type is tuple ('input',),inputKey[0]='input'. based on 2-gram, just consider one word that if it is in the 'injectedKeywords'
        if inputKey[0] in injectedKeywords:
            usedWdsInKeywds.append(inputKey[0])

    randomStart = 0
    randomTimesMMkey = 0
    "when possibleNextWords unempty"
    if possibleNextWords is not None:
        nextwordsInKeywords = []
        nextwordsNotKeywords = []
        next_inputKey = inputKey    #e.g.:inputKey= 'you are', then next_inputKey ='you are', when go to next inputkey process, next_inputkey=next_inputkey[1:] + nextword.
        for possibleNextWord in possibleNextWords:
            if possibleNextWord not in excludeInjectedOne:
                "should not use any keywords from other keywords subsets"
                if possibleNextWord in injectedKeywords:
                    nextwordsInKeywords.append(possibleNextWord)
                else:
                    nextwordsNotKeywords.append(possibleNextWord)

        temp_store = [] #store words that have been used before, all those words should be in injected keywords
        for ww in nextwordsInKeywords[::-1]:        #in a reverse loop, cause the positive order is changed once perform remove step,
            if ww in usedWdsInKeywds:
                temp_store.append(ww)
                nextwordsInKeywords.remove(ww)
                continue
        "choose one(high frequency) from the rest of nextwords that in injected keywords and not being used"
        if len(nextwordsInKeywords) != 0:
            nextword = Counter(nextwordsInKeywords).most_common(1)                                                      # return the one with highest frequency, when there are several words with same frequency, return the one first appear in the list
            nextword = nextword[0][0]
        # else:   #all nextwords are not in injected keywords
        #     nextwordsInKeywords = temp_store
        #     "choose one(randomly) from ones that in injectedkeywords but have been used"
        #     if len(nextwordsInKeywords) != 0:
        #         nextword = random.sample(nextwordsInKeywords,1)[0]                                                      #with [0], output original type, here output string
        else:
            "choose one from ones that not in injected keywords"
            temp_store1 = []    #store words that have been used in nextwords which remove the common words (all those words are not in injected keywords)
            remvUsedWdsNotKeywds = []   #store the words after remove words used before (exclude common words), the baseline is those words are not in injected keywords
            "remove the common words in the used words that not in injected keywords"
            processedUsedWdsNotKeywds = preprocessWords(usedWdsNotKeywds)
            for yy in nextwordsNotKeywords:
                if yy in processedUsedWdsNotKeywds:
                    temp_store1.append(yy)
                else:
                    remvUsedWdsNotKeywds.append(yy)
            if len(remvUsedWdsNotKeywds) != 0:
                "choose one(high frequency) from ones that have not been used(common words as candidates every time,and all words not in injected keywords)"
                """
                ************************** add on 7/11/2018 **********************************
                return 3 elements with higher frequency,then randomly choose one from them
                """
                nextword = Counter(remvUsedWdsNotKeywds).most_common(3)
                nextword = random.sample(nextword,1)[0][0]
            else:
                "choose one(randomly) from ones that have been used(all words not in injected keywords)"
                # nextword = random.sample(temp_store1,1)[0]
                nextword = random.sample(possibleNextWords,1)[0]
                # nextword = random.choice(temp_store1)
            usedWdsNotKeywds.append(nextword)

    else:
        while possibleNextWords is None:
            randomStart += 1
            "possibleNextWords is None, choose next word from the rest of injectedKeyWords"
            if len(reducedInjectedKeywords) != 0:
                next_inputKey = inputKey
                possibleNextWords = reducedInjectedKeywords
            else:
                ++randomTimesMMkey
                print('randomTimesMMkey+1')
                next_inputKey = random.sample(MM.keys(),1)[0]                                                       #output list
                possibleNextWords = MM.get(next_inputKey)
        nextword = random.sample(possibleNextWords,1)[0]
    return nextword, next_inputKey, reducedInjectedKeywords, randomStart, randomTimesMMkey, usedWdsInKeywds, usedWdsNotKeywds



def textGen(MM,injectedKeywords,allInjectedKeywords,lenThreshold,lenTerminal,N):
    """
    ************generation step*********
    generate text based on the only injected keywords and all should be included in the genText, at mean time genText should not allow
    include keywords from other keywords subset
    ************text requirements*******
    firstly, check if all injected keywords included in the genText
    then, check if genText endwiths '.', '?'..., and end generation step until endwiths the '.','?'...
    :param MM: Marchov model, it is in dictionary structure
    :param injectedKeywords: a keywords list refer to one file
    :param allInjectedKeywords: exclude 'injectedKeywords' from all injected keywords
    :param lenThreshold: ideal length for each generated text should be
    :param lenTerminal: the final length threshold for each generated text
    :param N: the number of n-gram
    :return: generated text
    """
    "check if text length therehold is greater than the number of injected keywords"
    if lenThreshold <= len(injectedKeywords):
        print('please check lenThreshold again, make sure Length threshold is greater than the number of injected keywords.')
        sys.exit(0)
    genWords = []                                                                                                       #store the chosen words
    randomStartSum = 0                                                                                                  #store the num of random start from the rest of injectedKeywords or MM[key]
    randomTimesMMkeySum = 0                                                                                             #store the num of random start from MM[key]
    usedWdsInKeywds = []                                                                                                #store words that have been used, and all those words are in injected keywprds
    usedWdsNotKeywds = []                                                                                               #store words that have been used, and all those words are not injected keywords
    "initial input"
    initialInput = random.sample(injectedKeywords,1)[0]                                                                 #without '[0]' output list, otherwise output string here (the original type,e.g. int str...)
    "append initial input to genWords as the fisrt word of this text"
    initialInput1 = initialInput.split(' ')
    for w in initialInput1[:N-1]:
        genWords.append(w)
    reducedInjectedKeywords = copy.deepcopy(injectedKeywords)                                                           #storing keywords that have no nextwords under currrent keyword subset
    excludeInjectedOne = excludeInjectedKeywords(allInjectedKeywords, injectedKeywords)                                 #storing keywords that exclude the current keyword subset
    inputKey_j = initialInput                                                                                           #inputKey_j: the initial input for n-gram each time
    for j in range(lenThreshold-(N-1)):
        nextword,inputKey_j,reducedInjectedKeywords,randomStart,randomTimesMMkey,usedWdsInKeywds,usedWdsNotKeywds \
            = getNextword(inputKey_j,MM,injectedKeywords,excludeInjectedOne,reducedInjectedKeywords,usedWdsInKeywds,usedWdsNotKeywds)
        genWords.append(nextword)
        randomStartSum += randomStart
        randomTimesMMkeySum += randomTimesMMkey
        "process the inputKey_j for next nextword searching"
        inputKey_j = str(inputKey_j).split(' ')
        inputKey_j = inputKey_j[1:]
        inputKey_j.append(nextword)
        inputKey_j = tuple(inputKey_j)
    genText = ' '.join(genWords)

    "*********************************************************"
    "check if all injected keywords from the file all are included in genText"
    keywordsExcluded = []                                                                                               #store the injected keywords that are not included in genText
    for word in injectedKeywords:
        while word not in genWords:
            nextword,inputKey_j,reducedInjectedKeywords,randomStart,randomTimesMMkey,usedWdsInKeywds,usedWdsNotKeywds \
                = getNextword(inputKey_j, MM, injectedKeywords, excludeInjectedOne,reducedInjectedKeywords,usedWdsInKeywds,usedWdsNotKeywds)
            genWords.append(nextword)
            randomStartSum += randomStart
            randomTimesMMkeySum += randomTimesMMkey
            "process the inputKey_j for next nextword searching"
            inputKey_j = str(inputKey_j).split(' ')
            inputKey_j = list(inputKey_j[1:])
            inputKey_j.append(nextword)
            inputKey_j = tuple(inputKey_j)
            genText = genText + ' ' + nextword
            if len(genWords) < lenTerminal:
                continue
            else:
                for xx in injectedKeywords:
                    if xx in genWords:
                        continue
                    else:
                        keywordsExcluded.append(xx)
                for yy in keywordsExcluded:
                    genWords.append(yy)
                    genText = genText + ' ' + yy
            break   #terminate while loop
    """
    *******************textEndingCheck********************
    check if genText endwiths '.', '?'...
    while loop times threshold be reached,then simply put '.' at the end of genText
    """
    loop_count = 0
    loopThreshold = 10
    while not genText.endswith(('.','!','?')):
        loop_count += 1
        if loop_count > loopThreshold:
            genText = genText + '.'
            genWords.append('.')
            break
        nextword,inputKey_j,reducedInjectedKeywords,randomStart,randomTimesMMkey,usedWdsInKeywds,usedWdsNotKeywds = \
            getNextword(inputKey_j, MM, injectedKeywords, excludeInjectedOne,reducedInjectedKeywords,usedWdsInKeywds,usedWdsNotKeywds)
        genWords.append(nextword)
        randomStartSum += randomStart
        randomTimesMMkeySum += randomTimesMMkey
        "process the inputKey_j for next nextword searching"
        inputKey_j = str(inputKey_j).split(' ')
        inputKey_j = list(inputKey_j[1:])
        inputKey_j.append(nextword)
        inputKey_j = tuple(inputKey_j)
        genText = genText + ' ' + nextword

    print ('num of random start(both from rest of injected words and MM[key]):' \
        , randomStartSum, '; num of random start from MM[key]:', randomTimesMMkeySum)
    print ('length of generated text ',len(genText.split()))
    return genText


def writeToFile(genText,outputPath,outputFile):
    """

    :param genText:
    :param outputPath:
    :param outputFile:
    :return:
    """
    outputPath = outputPath + '/' + outputFile
    with open(outputPath, 'w') as f:
        f.write(genText)


if __name__ == '__main__':
    print('textGenRestriction.py is running...' + '\n')
    startTime = time.time()
    dir = os.path.abspath(os.path.join(os.path.dirname('__file__'), (os.path.pardir)))

    # "test step0"
    # outputPath_test = dir + '/test'
    # try:
    #     items = os.listdir(outputPath_test)
    #     if items is not None:
    #         shutil.rmtree(outputPath_test)
    #         os.mkdir(outputPath_test)
    # except IOError:
    #     print 'no file in test folder yet'
    # for i in range(5):
    #     c = randomChooseKeywords(dir,i)
    #     # print c

    "test step1"
    inputPath = 'result/enron_results/injected_keywords'
    # inputPath = 'test'
    injectedKeywordsGroup, allInjectedKeywords= readInjectedKeyword(dir, inputPath)
    print (injectedKeywordsGroup)

    "test step2"
    corpusName = 'amazon_review_part1.txt'
    # corpusName = 'test_corpus.txt'
    MM = buildMarchovModel(dir, corpusName)

    "test step3"
    """
    1.lengthTerminal > lenThreshold > the num of injected keywords of each file
    2.generate numInputFile length list which contain numInputFile random integers among [1,500], numInputFile is equal to the number of input injected files
    """
    "uniform distribution"
    # lenThreshold = np.random.randint(21, 450, [1, numInputFile])[0]     #range[1,500], array: 1 row,1000 column;[0] covert array[[t]] to [t]
    # lenTerminal = [30] * numInputFile
    # lenTerminal = lenTerminal + lenThreshold
    "exponentila distribution"
    numInputFile = len(injectedKeywordsGroup)                                           #must greater than 10; len(injectedKeywordsGroup) is equal to the number of input injected files
    lenThreshold,lenTerminal = genExpRandomList(numInputFile)
    N = 2                                                                                                               #the number of n-gram
    outputPath = dir + '/genMessageWrestriction'
    try:
        items = os.listdir(outputPath)
        if items is not None:
            shutil.rmtree(outputPath)
            os.mkdir(outputPath)
    except IOError:
        print ('no file in ../genMessageWrestriction folder yet')
    for i in range(len(injectedKeywordsGroup)):
        print (str(i+1))
        genText = textGen(MM, injectedKeywordsGroup[i], allInjectedKeywords, lenThreshold[i], lenTerminal[i], N)
        outputFile = 'fakeMail_' + str(i+1) + '.txt'
        writeToFile(genText, outputPath, outputFile)
        print(genText)
    print('\n')
    print('fake mails have been writen in ../genMessageWrestriction folder successfully.')

    endTime = time.time()
    runningTime = float('%0.4f' % (endTime - startTime))
    print('Running time: ', runningTime, ' seconds')
