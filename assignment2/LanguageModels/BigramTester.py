#  -*- coding: utf-8 -*-
import math
import argparse
import nltk
import codecs
from collections import defaultdict

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""

class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter. 
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0


    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, 'r', 'utf-8') as f:
                # reads the first line of file to get the number of unique words and number of total words -- this line was given
                self.unique_words, self.total_words = map(int, f.readline().strip().split(' '))
                
                # YOUR CODE HERE
                # SAME CODE AS GENERATOR.PY
                # for each unique word (same logic as BigramTrainer.py)
                for i in range(self.unique_words):
                    # pass the id, token and number of appearances of each unique word from the file to the appropriate variables
                    id, token, numAppearances = f.readline().strip().split(' ')
                    # pass the id to self.index, the token to self.word and the number of appearances to unigram_count
                    self.index[token] = int(id)
                    self.word[int(id)] = token
                    self.unigram_count[int(id)] = int(numAppearances)

                # while there are still lines to read in
                while True:
                    # remove any leading or ending whitespace 
                    line = f.readline().strip()
                    # if the line reads -1, the end of the file has been reached
                    if line == "-1":
                        break
                    # store the two IDs and the probability of bigram occurring in the correct variables
                    firstID, secondID, probability = line.strip().split(' ')
                    # store the probability in the correct slot of bigram_prob
                    self.bigram_prob[int(firstID)][int(secondID)] = float(probability)

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    # the lower the entropy, the better language model learned from the training corpus
    # assumption that the test corpus is representative of the actual data
    def compute_entropy_cumulatively(self, word):
        # YOUR CODE HERE

        # coefficient to multiply the log probability by
        multiplier = (-1/self.total_words)
        #print(multiplier)

        # EQN: lambda1*P(wi|wi-1) + lambda2*P(wi) + lambda3

        # initialise the first and second terms to zero so they can be used outside the loops
        firstTerm = 0
        secondTerm = 0
        # if the current word is in index list
        if word in self.index:
            self.test_words_processed += 1
            # let id be the index of that word
            id = self.index[word]
            # if last word was not the first word or invalid and if the ID of the current word is a possibility to pair with the last word in a bigram from the text
            if self.last_index > -1 and id in self.bigram_prob[self.last_index]:
                # let the first term be equal to lambda1 times e^log probability of that bigram occurring (P(wi|wi-1))
                firstTerm = self.lambda1 * math.exp(self.bigram_prob[self.last_index][id])
            
            # if the ID is a valid unigram
            if id in self.unigram_count:
                # let the second term equal lambda times the probability of that word occurring
                secondTerm = self.lambda2 * self.unigram_count[id] / self.total_words
            # set the last_index to the ID of the current word
            self.last_index = id
        # if the word is not in the index list then set the previous index to -1, 
        else:
            self.last_index = -1
        # calculate the interpolated probability as the first two terms and lambda three added together as per the eqn
        wordProbability = firstTerm + secondTerm + self.lambda3
        #print("WORD PROBABILITY", wordProbability)

        logProb = -math.log(wordProbability)/len(self.tokens)
        self.logProb += logProb

    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, 'r', 'utf-8') as f:
                self.tokens = nltk.word_tokenize(f.read().lower()) 
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print('Error reading testfile')
            return False


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--test_corpus', '-t', type=str, required=True, help='test corpus')

    arguments = parser.parse_args()

    bigram_tester = BigramTester()
    bigram_tester.read_model(arguments.file)
    bigram_tester.process_test_file(arguments.test_corpus)
    print('Read {0:d} words. Estimated entropy: {1:.2f}'.format(bigram_tester.test_words_processed, bigram_tester.logProb))

if __name__ == "__main__":
    main()
