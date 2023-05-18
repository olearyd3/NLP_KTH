import math
import argparse
import codecs
from collections import defaultdict
import random

"""
This file is part of the computer assignments for the course DD2417 Language engineering at KTH.
Created 2018 by Johan Boye and Patrik Jonell.
"""

class Generator(object) :
    """
    This class generates words from a language model.
    """
    def __init__(self):
    
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


    def read_model(self,filename):
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
        
    def generate(self, w, n):
        """
        Generates and prints n words, starting with the word w, and sampling from the distribution
        of the language model.
        """ 
        # YOUR CODE HERE
        # convert all words to lowercase so that they all match -- program does not run without this
        w = w.lower()
        # if the word is not in the text, print an error
        if w not in self.index:
            print("\n" + "ERROR: This word does not occur in the text. Please try again." + "\n")
        # if the word is in the text:
        else:
            # set the output to be the first word, with a space after it
            output = w + " "
            # set the the index of the word to the id
            id = self.index[w]
            # for the range up to n-1
            for i in range(n-1):
                # create two empty lists - one for choices and one for the likelihood of each choice occurring
                choices = []
                likelihood = []
                # if the ID is in bigram probabilities
                if id in self.bigram_prob:
                    #print(self.bigram_prob[id].items()) --- debugging
                    # each slot of bigram_prob[id].items contains all possible words that follow it and their respective bigram probabilities
                    # e.g. if the ID is 0, self.bigram_prob[id].items() is [(1, -1.09861228866811), (5, -0.405465108108164)]
                    # these correspond to                                   i live (logprob)              i like (logprob)
                    # assign the word following w to k and the log prob to v for each possible word that follows the current word
                    for k, v in self.bigram_prob[id].items():
                        # add the word, k, to the choices list
                        choices.append(k)
                        # calculate e^(log probability) to calculate likelihood of each word occurring after the current word
                        likelihood.append(math.exp(v))
                    # set id to one of the options in the choices list at random, based on the likelihoods in the likelihood list
                    id = random.choices(population=choices, weights=likelihood)[0]
                    # set the current word to the word at that ID
                    w = self.word[id]
                    # concatenate that word to the output with a space after it
                    output = output + w + " "
                else:
                    # if the current word has no bigrams that start with it, randomly assign a word from the list of unique words 
                    # (starting at index 0 until index unique_words-1)
                    id = random.randint(0, self.unique_words-1)
                    # add the new word to the output 
                    output = output + self.word[id] + " "
            # print the output
            print(output)


def main():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BigramTester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='file with language model')
    parser.add_argument('--start', '-s', type=str, required=True, help='starting word')
    parser.add_argument('--number_of_words', '-n', type=int, default=100)

    arguments = parser.parse_args()

    generator = Generator()
    generator.read_model(arguments.file)
    generator.generate(arguments.start,arguments.number_of_words)

if __name__ == "__main__":
    main()
