#coding = utf-8
import random,re,os,shutil,time


class MarkovChain:

    def __init__(self,n):                                                                                               #n refer to n-gram
        self.n = n
        self.dir = os.path.abspath(os.path.join(os.path.dirname('__file__'),os.path.pardir))


    def _learn_key(self, memory, key, value):                                                                           #key-refers to "current state"; value-refers to next_possible following current state
        if key not in memory:                                                                                           #if this key is the first time showing in the memory[key]
            memory[key] = []                                                                                            #to assign a new array for this key;
        memory[key].append(value)                                                                                       #append the value for the array assingned at the first time


    def learn(self, text):                                                                                              #text-refers to the text need to training
        tokens = text.split(" ")                                                                                        #split word according to blank space
        N = self.n
        n_grams = [tuple(tokens[i:i+N]) for i in range(0, len(tokens) - (N-1))]                                         #each group with 3 words; the range should be range(0,len(tokens)-(n-1)),n is the number of grams
        memory = {}
        for n_gram in n_grams:
            self._learn_key(memory, n_gram[0:(N-1)], n_gram[N-1])                                                       #[0:2]:0,1; for (bigram[i],bigram[i+1]), call function _learn_ley()
        # print memory
        return memory



    def _next(self, current_state, memory):
        next_possible = memory.get(current_state)                                                                       #obtain next possible results that following the current state
        count = 0                                                                                                       #record the times of random choose
        if next_possible is not None:
            next_state = current_state
        else:
            while next_possible is None:                                                                                #if this next possible result is none
                count += 1
                next_state_possible = list(memory.keys())                                                               #choose key from the memory we built in the first place randomly, the key as the input for next state
                next_state = random.sample(next_state_possible, 1)[0]                                                   #without [0] output orginal type (it is string here), otherwise output list
                next_possible = memory.get(next_state)
        next_word = ''.join(random.sample(next_possible, 1)[0])                                                         #''.join() convert list to string
        return next_word,next_state,count


    def babble(self, sentence_length, state, memory):                                                                   #sentence_length-the length of the sentence that want to generate
        babble_words = []
        state_0 = state                                                                                                 #store the initial input word for later use
        N = self.n
        count_randomStart = 0
        for i in range(0,sentence_length-(N-1)):                                                                        #for(0,s_length-(n-1))
            state_i = state                                                                                             #save the last state in state_i for the if_function
            next_result = self._next(state, memory)                                                                     #call _next function
            next_word = next_result[0]
            state = next_result[1]
            count_randomStart += next_result[2]
            "next_possible is not none, so the next_state won't change, that's to say state(next_state)=state(last state)"
            if state != state_i:
                babble_words += state                                                                                   #when random choose happens in the _next step, then the next_state should be append in the sentence
            babble_words.append(next_word)                                                                              #append each next_word in a list, for the sentence that needed to be generated, only miss the initialted word to insert in the head of this sentence
            "generate state for _next function"
            state = list(state[1:])                                                                                     #***n-gram***for n-gram,state should include (n-1) elements;state is input for self._next function; e.g. state=[s,g],next_word=j, then next state=[g,j]
            state.append(next_word)                                                                                     #can't make anychange in tuple, convert to list first and then append
            state = tuple(state)
        for i in range(0,N-1):
            babble_words.insert(i,state_0[i])                                                                           #insert initial input word in the head of the sentence
        babble_sentence = ' '.join(babble_words)
        while not babble_sentence.endswith(('.','!','?')):
            state_j = state
            next_result = self._next(state, memory)
            next_word = next_result[0]
            state = next_result[1]
            count_randomStart += next_result[2]
            "if next_possible is not none...append next_state(which is choosed randomly) to babble_words"
            if state != state_j:
                babble_words += state
            babble_words.append(next_word)
            "generate next state"
            state = list(state[1:])
            state.append(next_word)
            state = tuple(state)
            babble_sentence = babble_sentence + ' ' + next_word
        # print('The number of random start is:',count_randomStart)
        return babble_sentence,count_randomStart


    def preprocess_corpus(self,filename):
        with open(self.dir + '/data/' + filename, 'r') as f:
            s = f.read()
        s = re.sub('[()]', r'', s)                                                                                      # remove certain punctuation chars
        s = re.sub('([.-])+', r'\1', s)                                                                                 # collapse multiples of certain chars
        s = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', s)                                                          # pad sentence punctuation chars with whitespace
        s = ' '.join(s.split()).lower()                                                                                 # remove extra whitespace (incl. newlines)
        return s


    def postprocess_output(self,s):
        s = re.sub('\\s+([.,!?])\\s*', r'\1 ', s)                                                                       # correct whitespace padding around punctuation
        s = s.capitalize();                                                                                             # capitalize first letter
        s = re.sub('([.!?]\\s+[a-z])', lambda c: c.group(1).upper(),s)                                                  # capitalize letters following terminated sentences
        return s


    def write_all_messages_to_file(self,numbers_test,sentence_length,memory):
        """
        given n messages, write them in an output_file 
        :return: 
        """
        try:
            items = os.listdir(self.dir + '/genMessage')
            if items is not None:
                shutil.rmtree(self.dir + '/genMessage')
                os.mkdir(self.dir + '/genMessage')
        except IOError:
            print('creating genMessage folder')
        finally:
            count = 0                                                                                                   #the order of the generate message
            try:
                with open(self.dir + '/result/randomStart.txt', 'r') as file:
                    f = file.read()
                if f is not None:
                    os.remove(self.dir + '/result/randomStart.txt')
            except IOError:
                print('creating randomStart.txt...')
            finally:
                for i in range(0,numbers_test):
                    "random input"
                    initial_input = random.sample(memory.keys(),1)[0]                                                   #choose input randomly each time; with [0], output a original type (string) instead of list
                    "fixed input"
                    # initial_input = '"I'
                    # initial_input = initial_input.lower()
                    # initial_input = re.sub('([^0-9])([.,!?])([^0-9])', r'\1 \2 \3', initial_input)
                    # initial_input = ' '.join(initial_input.split()).lower()
                    # initial_input = tuple(initial_input.split(" "))                                                   #after convert to tuple from list e.g.('meeting',), there is a ',' in the ending, that wont consider a efficient char, just for resentation
                    # print('Initial input is:',initial_input)
                    message_result = self.babble(sentence_length, initial_input, memory)
                    message = message_result[0]                                                                         #babble function return 2 values, [0] refer to the first one
                    msj_randomStart = message_result[1]
                    message = self.postprocess_output(message)
                    # print(message)
                    count += 1
                    filename = 'gen_msj_' + str(count) + '.txt'                                                         # creat a txt file in the specific path
                    with open(self.dir + '/genMessage/' + filename, 'w') as file:
                        file.write(message)
                    with open(self.dir + '/result/randomStart.txt', 'a') as file:
                        file.write(filename + ': ' + str(msj_randomStart) + '\n')


if __name__ == '__main__':
    start_time = time.time()
    print('n_gramAuto program is running...')
    m = MarkovChain(3)
    corpus = m.preprocess_corpus('corpus4FilterLegitUserDivide1.txt')
    memory = m.learn(corpus)
    gen_sentence = m.write_all_messages_to_file(1,3,memory)                                                           #write_all_message_to_file should return only one result, so in the final filter version should read generated sentence from file
    print('Message has been written in file.')
    elapse_time = time.time()
    print('running time is: ' + str(float('%0.4f' %(elapse_time - start_time))) + ' seconds')
