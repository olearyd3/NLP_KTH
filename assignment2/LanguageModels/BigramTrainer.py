#  -*- coding: utf-8 -*-
from __future__ import unicode_literals
import math
import argparse
import nltk
import os
from collections import defaultdict
import codecs

"""
This file is part of the computer assignments for the course DD2417 Language Engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTrainer(object):
    """
    This class constructs a bigram language model from a corpus.
    """

    def process_files(self, f):
        """
        Processes the file @code{f}.
        """
        with codecs.open(f, 'r', 'utf-8') as text_file:
            text = reader = str(text_file.read()).lower()
        try :
            self.tokens = nltk.word_tokenize(text) # Important that it is named self.tokens for the --check flag to work
        except LookupError :
            nltk.download('punkt')
            self.tokens = nltk.word_tokenize(text)
        for token in self.tokens:
            self.process_token(token)

    # function to process the current word and adjust the counts of for unigrams and bigrams
    def process_token(self, token):
        """
        Processes one word in the training corpus, and adjusts the unigram and
        bigram counts.

        :param token: The current word to be processed.
        """
        # YOUR CODE HERE

        # if the token has not yet appeared in the unique tokens list (added below in __init__)
        if token not in self.unique_tokens:
            # append it to the unique_tokens list
            self.unique_tokens.append(token)
            # allocate its ID to index (0, 1, 2, 3 etc...)
            self.index[token] = self.unique_tokens.index(token)
            # map the token to the appropriate slot of word
            self.word[self.unique_tokens.index(token)] = token

        # YOUR CODE HERE
        # calculate the total number of words in the corpus
        self.total_words += 1
        self.unigram_count[token] += 1

        # assign the index of the current token to the variable curr_index
        curr_index = self.index[token]

        # if not the first word
        if self.last_index > -1:
            # add one to the bigram count at the indices for last word and current word
            self.bigram_count[self.last_index][curr_index] += 1

        # set the previous word to the current token and set the last index to the current one
        self.prev_word = token
        self.last_index = curr_index

        #print(token, self.unigram_count[token])
        


    def stats(self):
        """
        Creates a list of rows to print of the language model.

        """
        rows_to_print = []

        # YOUR CODE HERE
        # set the size of unique_words to the number of unique tokens
        self.unique_words = len(self.unique_tokens)
        # to append to the list, need to specify there are going to be two terms in the row and then pass the values for V and N
        # first row is the vocab size (V) then the size of the corpus (N)
        rows_to_print.append("{} {}".format(self.unique_words, self.total_words))
        #print(self.unique_words, self.total_words)

        # for each unique word:
        for i in range(self.unique_words):
            # print the index of the word, the word and its unigram count -- again need to append 3 terms, so {} {} {}
            rows_to_print.append("{} {} {}".format(i, self.unique_tokens[i], self.unigram_count[self.unique_tokens[i]]))
            #print(i, self.unique_tokens[i], self.unigram_count[self.unique_tokens[i]])
        
        # for each first index in bigram_count (e.g. like)
        for i in self.bigram_count:
            # for each second index that follows that first index (e.g. like honey)
            for j in self.bigram_count[i]:
                # calculate the bigram probability by getting the natural log of the number of times the word pair appeared
                # and subtracting the natural log of the number of times the first word appeared 
                # (number of times word 2 follows word 1)/(number of times word 1 appears) --- also: log(a/b) = log(a) - log(b)
                bigram_probability = math.log(self.bigram_count[i][j]) - math.log(self.unigram_count[self.unique_tokens[i]])
                # print i, j and bigram probability, and specify probability to 15 decimal places
                rows_to_print.append("{} {} {:.15f}".format(i, j, bigram_probability))
                #print(i, j, '{0:.15f}'.format(bigram_probability))
        # also append the -1 to indicate the end of the file
        rows_to_print.append("{}".format(-1))
        #print(rows_to_print)
        
        return rows_to_print

    def __init__(self):
        """
        <p>Constructor. Processes the file <code>f</code> and builds a language model
        from it.</p>

        :param f: The training file.
        """

        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = defaultdict(int)

        """
        The bigram counts. Since most of these are zero (why?), we store these
        in a hashmap rather than an array to save space (and since it is impossible
        to create such a big array anyway).
        """
        self.bigram_count = defaultdict(lambda: defaultdict(int))

        # The identifier of the previous word processed.
        self.last_index = -1

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        self.laplace_smoothing = False

        # added list of unique tokens
        self.unique_tokens = []

        # added previous word  
        self.prev_word = ""


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTrainer')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file from which to build the language model')
    parser.add_argument('--destination', '-d', type=str, help='file in which to store the language model')

    arguments = parser.parse_args()

    bigram_trainer = BigramTrainer()

    bigram_trainer.process_files(arguments.file)

    stats = bigram_trainer.stats()
    if arguments.destination:
        with codecs.open(arguments.destination, 'w', 'utf-8' ) as f:
            for row in stats: f.write(row + '\n')
    else:
        for row in stats: print(row)


if __name__ == "__main__":
    main()
