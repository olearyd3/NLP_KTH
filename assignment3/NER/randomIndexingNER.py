import os
import argparse
import time
import string
import numpy as np
from halo import Halo
from sklearn.neighbors import NearestNeighbors
import re
import random

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2018 by Dmytro Kalpakchi and Johan Boye.
"""


##
## @brief      Class for creating word vectors using Random Indexing technique.
## @author     Dmytro Kalpakchi <dmytroka@kth.se>
## @date       November 2018
##
class RandomIndexing(object):

    ##
    ## @brief      Object initializer Initializes the Random Indexing algorithm
    ##             with the necessary hyperparameters and the textfiles that
    ##             will serve as corpora for generating word vectors
    ##             
    ## The `self.__vocab` instance variable is initialized as a Python's set. If you're unfamiliar with sets, please
    ## follow this link to find out more: https://docs.python.org/3/tutorial/datastructures.html#sets.
    ##
    ## @param      self               The RI object itself (is omitted in the descriptions of other functions)
    ## @param      filenames          The filenames of the text files (7 Harry
    ##                                Potter books) that will serve as corpora
    ##                                for generating word vectors. Stored in an
    ##                                instance variable self.__sources.
    ## @param      dimension          The dimension of the word vectors (both
    ##                                context and random). Stored in an
    ##                                instance variable self.__dim.
    ## @param      non_zero           The number of non zero elements in a
    ##                                random word vector. Stored in an
    ##                                instance variable self.__non_zero.
    ## @param      non_zero_values    The possible values of non zero elements
    ##                                used when initializing a random word. Stored in an
    ##                                instance variable self.__non_zero_values.
    ##                                vector
    ## @param      left_window_size   The left window size. Stored in an
    ##                                instance variable self__lws.
    ## @param      right_window_size  The right window size. Stored in an
    ##                                instance variable self__rws.
    ##
    def __init__(self, filenames, dimension=2000, non_zero=100, non_zero_values=list([-1, 1]), left_window_size=3, right_window_size=3):
        self.__sources = filenames
        self.__vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        # there is a list call in a non_zero_values just for Doxygen documentation purposes
        # otherwise, it gets documented as "[-1,"
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None

        # added two more values for nearest neighbour model and index to word
        self.__nearest_neighbour_model = None
        self.__index_to_word = {}
        

    ##
    ## @brief      A function cleaning the line from punctuation and digits
    ##
    ##             The function takes a line from the text file as a string,
    ##             removes all the punctuation and digits from it and returns
    ##             all words in the cleaned line.
    ##
    ## @param      line  The line of the text file to be cleaned
    ##
    ## @return     A list of words in a cleaned line
    ##
    def clean_line(self, line):
        # replace and digits and characters that aren't word characters or whitespace with blank spaces and split the words (regex) 
        return re.sub(r'\d|[^\w\s]', '', line).split()

    ##
    ## @brief      A generator function providing one cleaned line at a time
    ##
    ##             This function reads every file from the source files line by
    ##             line and returns a special kind of iterator, called
    ##             generator, returning one cleaned line a time.
    ##
    ##             If you are unfamiliar with Python's generators, please read
    ##             more following these links:
    ## - https://docs.python.org/3/howto/functional.html#generators
    ## - https://wiki.python.org/moin/Generators
    ##
    ## @return     A generator yielding one cleaned line at a time
    ##
    def text_gen(self):
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                for line in f:
                    yield self.clean_line(line)


    ##
    ## @brief      Build vocabulary of words from the provided text files.
    ##
    ##             Goes through all the cleaned lines and adds each word of the
    ##             line to a vocabulary stored in a variable `self.__vocab`. The
    ##             words, stored in the vocabulary, should be unique.
    ##             
    ##             **Note**: this function is where the first pass through all files is made
    ##             (using the `text_gen` function)
    ##
    def build_vocabulary(self):
        # YOUR CODE HERE
        line_gen = self.text_gen()
        # for each line in the text
        for line in line_gen:
            self.add_words_to_vocab(line)

        self.write_vocabulary()

    # if the word is new, add it to the __vocab
    def add_words_to_vocab(self, line):
        for word in line:
            if word not in self.__vocab:
                self.__vocab.add(word)

    ##
    ## @brief      Get the size of the vocabulary
    ##
    ## @return     The size of the vocabulary
    ##
    @property
    def vocabulary_size(self):
        return len(self.__vocab)


    ##
    ## @brief      Creates word embeddings using Random Indexing.
    ## 
    ## The function stores the created word embeddings (or so called context vectors) in `self.__cv`.
    ## Random vectors used to create word embeddings are stored in `self.__rv`.
    ## 
    ## Context vectors are created by looping through each cleaned line and updating the context
    ## vectors following the Random Indexing approach, i.e. using the words in the sliding window.
    ## The size of the sliding window is governed by two instance variables `self.__lws` (left window size)
    ## and `self.__rws` (right window size).
    ## 
    ## For instance, let's consider a sentence:
    ##      I really like programming assignments.
    ## Let's assume that the left part of the sliding window has size 1 (`self.__lws` = 1) and the right
    ## part has size 2 (`self.__rws` = 2). Then, the sliding windows will be constructed as follows:
    ## \verbatim
    ##      I really like programming assignments.
    ##      ^   r      r
    ##      I really like programming assignments.
    ##      l   ^      r       r
    ##      I really like programming assignments.
    ##          l      ^       r           r
    ##      I really like programming assignments.
    ##                 l       ^           r
    ##      I really like programming assignments.
    ##                         l           ^
    ## \endverbatim
    ## where "^" denotes the word we're currently at, "l" denotes the words in the left part of the
    ## sliding window and "r" denotes the words in the right part of the sliding window.
    ## 
    ## Implementation tips:
    ## - make sure to understand how generators work! Refer to the documentation of a `text_gen` function
    ##   for more description.
    ## - the easiest way is to make `self.__cv` and `self.__rv` dictionaries with keys being words (as strings)
    ##   and values being the context vectors.
    ## 
    ## **Note**: this function is where the second pass through all files is made (using the `text_gen` function).
    ##         The first one was done when calling `build_vocabulary` function. This might not the most
    ##         efficient solution from the time perspective, but it's quite efficient from the memory
    ##         perspective, given that we are using generators, which are lazily evaluated, instead of
    ##         keeping all the cleaned lines in memory as a gigantic list.
    ##
    def create_word_vectors(self):
        # YOUR CODE HERE
        line_gen = self.text_gen()

        # initialise the random vectors and context vectors
        self.__rv = {}
        self.__cv = {}
        # for each line 
        for line in line_gen:
            # for each word index and the word associated with it
            for word_index, focus_word in enumerate(line):
                # get the words to the left and right of the current word
                left_words = self.get_left_words(word_index, line)
                right_words = self.get_right_words(word_index, line)
                # let these words be the context words
                context_words = left_words + right_words
                # update the word vectors with the current word and context words
                self.update_word_vectors(focus_word, context_words)

        # testing a word in text and a word not in text
        #print('"Harry" vector:', self.get_word_vector('needed'))
        #print('"míthábhachtach" vector:', self.get_word_vector('míthábhachtach'))

    # function to get the words to the left of the current word
    def get_left_words(self, word_index, line):
        # let the words before the current word be the index
        words_before = word_index
        # if the number of words before is leq the size of the left window, return the words before the current one
        if words_before <= self.__lws:
            return line[:word_index]
        # otherwise return the left window size number of words before the current one
        return line[word_index - self.__lws:word_index]

    # function to get the words to the right of the current word
    def get_right_words(self, word_index, line):
        # the words after the current one are the length of the line minus the current index minus 1 
        words_after = len(line) - word_index - 1
        # if the number of words is leq the size of the right window then return the words after the current word
        if words_after <= self.__rws:
            return line[word_index + 1:]
        # otherwise return the right window size number of words after the current one
        return line[word_index + 1:word_index + 1 + self.__rws]

    # function to update the word vectors
    def update_word_vectors(self, focus_word, context_words):
        # if the word is new then set its context vectors to zero and its random vectors to random nums
        if not focus_word in self.__rv:
            self.__cv[focus_word] = np.zeros((self.__dim,), dtype=int)
            self.__rv[focus_word] = self.generate_random_vec()

        # for each context word, if it's new then initialise CV and RV as before.
        for context_word in context_words:
            if not context_word in self.__rv:
                self.__cv[context_word] = np.zeros((self.__dim,), dtype=int)
                self.__rv[context_word] = self.generate_random_vec()
            # otherwise, concatenate the context vector of the focus word with the random vector of the context word
            self.__cv[focus_word] += self.__rv[context_word]

    # function the generate random vectors for initialising __rv 
    def generate_random_vec(self):
        # choose first and secong numbers and probabilities for the first
        first_num = self.__non_zero_values[0]
        second_num = self.__non_zero_values[1]
        prob_first_num = random.uniform(0, 1)
        # calculate the length of the first and second numbers
        len_first_nums = round(prob_first_num*self.__non_zero)
        len_second_nums = self.__non_zero - len_first_nums

        # gets the first and second nums and zeros
        first_nums = np.full(len_first_nums, first_num)
        second_nums = np.full(len_second_nums, second_num)
        zeros = np.zeros((self.__dim - self.__non_zero,), dtype=int)
        # concatenates the above into a random vector and shuffles it and returns it
        random_vec = np.concatenate([first_nums, second_nums, zeros])
        np.random.shuffle(random_vec)
        return random_vec


    ##
    ## @brief      Function returning k nearest neighbors with distances for each word in `words`
    ## 
    ## We suggest using nearest neighbors implementation from scikit-learn 
    ## (https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html). Check
    ## carefully their documentation regarding the parameters passed to the algorithm.
    ## 
    ## To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
    ## "Harry" and "Potter" using cosine distance (which can be computed as 1 - cosine similarity). 
    ## For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='cosine')`.
    ## The output of the function would then be the following list of lists of tuples (LLT)
    ## (all words and distances are just example values):
    ## \verbatim
    ## [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
    ##  [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
    ## \endverbatim
    ## The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
    ## list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
    ## The tuples are sorted either by descending similarity or by ascending distance.
    ##
    ## @param      words   A list of words, for which the nearest neighbors should be returned
    ## @param      k       A number of nearest neighbors to be returned

    ## @param      metric  A similarity/distance metric to be used (defaults to cosine distance)
    ##
    ## @return     A list of list of tuples in the format specified in the function description
    ##
    def find_nearest(self, words, k=5, metric='cosine'):
        # YOUR CODE HERE
        # if there is not already a nearest neighbour model
        if self.__nearest_neighbour_model is None:
            # call the NearestNeighbours method using cosine distance
            self.__nearest_neighbour_model = NearestNeighbors(metric = metric)
            # get all the context vectors
            all_cvs = self.get_all_cvs()
            # fit the nearest neighbour model with all the context vectors
            self.__nearest_neighbour_model.fit(all_cvs)

        # get the query vectors
        # let query_vecs be a list of zeros
        query_vecs = np.zeros([len(words), self.__dim], dtype=int)
        # for each index and word in the words
        for index, word in enumerate(words):
            # if the word is in the vocab
            if word in self.__cv:
                # set the index of query_vecs equal to the word'th slot of the context vector and return the list of query vectors
                query_vecs[index] = self.__cv[word]
            # if not in the vocab print error 
            else:
                print("ERROR: Word is not in the vocabulary")
                return None
        # calculate the distances and word indices using the kneighbours method 
        distances, word_indices = self.__nearest_neighbour_model.kneighbors(query_vecs)
        # let the formatted nearest neighbour
        nearest_neighbour = self.format_nearest_neighbours(distances, word_indices)

        return nearest_neighbour

    # function to format the nearest neighbours
    def format_nearest_neighbours(self, distances, word_indices):
        # create an empty list
        nearest_neighbour = []
        # for each distance and list of indices 
        for distance_list, word_index_list in zip(distances, word_indices):
            nearest_neighbour_data = []
            # same again
            for distance, word_index in zip(distance_list, word_index_list):
                # let the nearest word be the index of the current word
                nearest_neighbour_word = self.__index_to_word[word_index]
                # append this word rounded to two decimal places to the list of nearest neighbours
                nearest_neighbour_data.append((nearest_neighbour_word, round(distance, 2)))
            # sort the list in ascending order, from lowest values to highest
            nearest_neighbour_data.sort(key=lambda x: x[1])
            nearest_neighbour.append(nearest_neighbour_data)

        return nearest_neighbour
    
    # function to get all the context vectors
    def get_all_cvs(self):
        # let the number of word vectors be the length of the context vectors
        number_word_vecs = len(self.__cv)
        # set all_cvs to zero
        all_cvs = np.zeros([number_word_vecs, self.__dim], dtype=int)
        # for each index and each word and its vector in the list of context vectors
        for index, (word, vector) in enumerate(self.__cv.items()):
            # let the index to the word be the word
            self.__index_to_word[index] = word
            # set that slot of all_cvs to the vector
            all_cvs[index] = vector
        return all_cvs

    # function to get all the word vectors
    def get_words_for_wvs(self):
        # initialise an empty words array
        words = []
        # for each index and word vector in the list of all context vectors, append the word to the words list
        for word_index, word_vec in enumerate(self.get_all_cvs()):
            words.append(self.__index_to_word[word_index])
        return words

    # function to get all the query vectors
    def get_query_vecs(self, words):
        # let query_vecs be a list of zeros
        query_vecs = np.zeros([len(words), self.__dim], dtype=int)
        # for each index and word in the words
        for index, word in enumerate(words):
            if word in self.__cv:
                # set the index of query_vecs equal to the word'th slot of the context vector and return the list of query vectors
                query_vecs[index] = self.__cv[word]
            else:
                print("ERROR: Word is not in the vocabulary")
                break

        return query_vecs


    ##
    ## @brief      Returns a vector for the word obtained after Random Indexing is finished
    ##
    ## @param      word  The word as a string
    ##
    ## @return     The word vector if the word exists in the vocabulary and None otherwise.
    ##
    def get_word_vector(self, word):
        if word not in self.__cv:
            return None
        return self.__cv[word]


    ##
    ## @brief      Checks if the vocabulary is written as a text file
    ##
    ## @return     True if the vocabulary file is written and False otherwise
    ##
    def vocab_exists(self):
        return os.path.exists('vocab.txt')


    ##
    ## @brief      Reads a vocabulary from a text file having one word per line.
    ##
    ## @return     True if the vocabulary exists was read from the file and False otherwise
    ##             (note that exception handling in case the reading failes is not implemented)
    ##
    def read_vocabulary(self):
        vocab_exists = self.vocab_exists()
        if vocab_exists:
            with open('vocab.txt') as f:
                for line in f:
                    self.__vocab.add(line.strip())
        self.__i2w = list(self.__vocab)
        return vocab_exists


    ##
    ## @brief      Writes a vocabulary as a text file containing one word from the vocabulary per row. 
    ##
    def write_vocabulary(self):
        with open('vocab.txt', 'w') as f:
            for w in self.__vocab:
                f.write('{}\n'.format(w))


    ##
    ## @brief      Main function call to train word embeddings
    ## 
    ## If vocabulary file exists, it reads the vocabulary from the file (to speed up the program),
    ## otherwise, it builds a vocabulary by reading and cleaning all the Harry Potter books and
    ## storing unique words.
    ## 
    ## After the vocabulary is created/read, the word embeddings are created using Random Indexing.
    ##
    def train(self):
        spinner = Halo(spinner='arrow3')
        if self.vocab_exists():
            spinner.start(text="Reading vocabulary...")
            start = time.time()
            self.read_vocabulary()
            spinner.succeed(text="Read vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        else:
            spinner.start(text="Building vocabulary...")
            start = time.time()
            self.build_vocabulary()
            spinner.succeed(text="Built vocabulary in {}s. Size: {} words".format(round(time.time() - start, 2), ri.vocabulary_size))
        
        spinner.start(text="Creating vectors using random indexing...")
        start = time.time()
        self.create_word_vectors()
        spinner.succeed("Created random indexing vectors in {}s.".format(round(time.time() - start, 2)))

        spinner.succeed(text="Execution is finished! Please enter words of interest (separated by space):")


    ##
    ## @brief      Trains word embeddings and enters the interactive loop, where you can 
    ##             enter a word and get a list of k nearest neighours.
    ##
    def train_and_persist(self):

        self.train()
        self.write_vecs_to_file()
        print("PRESS q FOR EXIT")
        text = input('> ')
        while text != 'q':
            text = text.split()
            neighbors = self.find_nearest(text)
            # added an if loop to handle words not in the vocab
            if neighbors != None:
                for w, n in zip(text, neighbors):
                    print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

    def write_vecs_to_file(self):
        """
        Write the model to a file `ri.txt`
        """
        try:
            with open("ri.txt", 'w') as f:
                W = self.get_all_cvs()
                nr_vecs = W.shape[0]
                vec_dim = W.shape[1]

                f.write("{} {}\n".format(nr_vecs, vec_dim))
                for i, w in self.__index_to_word.items():
                    f.write(str(w) + " " + " ".join(map(lambda x: "{0:.6f}".format(x), W[i,:])) + "\n")
        except:
            print("Error: failing to write model to the file")

    @classmethod
    def load(cls, fname):
        """
        Load the ri model from a file `fname`
        """
        ri = None
        try:
            with open(fname, 'r') as f:
                V, H = (int(a) for a in next(f).split())
                ri = cls([], dimension=H)

                temp_vec, i2w = np.zeros(H), []
                W = {}
                for i, line in enumerate(f):
                    parts = line.split()
                    word = parts[0].strip()
                    temp_vec = list(map(float, parts[1:]))
                    W[word] = temp_vec
                    i2w.append(word)

                ri.init_params(W, i2w, H)
        except:
            print("Error: failing to load the model to the file")
        return ri

    def init_params(self, W, i2w, dim):
        self.__cv = W
        self.__index_to_word = i2w
        self.__dim = dim


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('-fv', '--force-vocabulary', action='store_true', help='regenerate vocabulary')
    parser.add_argument('-c', '--cleaning', action='store_true', default=False)
    parser.add_argument('-co', '--cleaned_output', default='cleaned_example.txt', help='Output file name for the cleaned text')
    args = parser.parse_args()

    if args.force_vocabulary:
        os.remove('vocab.txt')

    if args.cleaning:
        ri = RandomIndexing(['example.txt'])
        with open(args.cleaned_output, 'w') as f:
            for part in ri.text_gen():
                f.write("{}\n".format(" ".join(part)))
    else:
        dir_name = "dataRI"
        filenames = [os.path.join(dir_name, fn) for fn in os.listdir(dir_name)]

        ri = RandomIndexing(filenames)
        ri.train_and_persist()