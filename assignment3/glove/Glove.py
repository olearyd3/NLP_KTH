import os
import math
import random
import nltk
import numpy as np
import numpy.random as rand
import os.path
import argparse
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

"""
Python implementation of the Glove training algorithm from the article by Pennington, Socher and Manning (2014).

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2021, 2022 by Johan Boye.
"""
class Glove:
    def __init__( self, continue_training, left_window_size, right_window_size ) :
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Mapping from words to IDs.
        self.word2id = defaultdict(lambda: None)

        # Mapping from IDs to words.
        self.id2word = defaultdict(lambda: None)

        # Mapping from focus words to neighbours to counts (called X 
        # to be consistent with the notation in the Glove paper).
        self.X = defaultdict(lambda: defaultdict(int))

        # Mapping from word IDs to (focus) word vectors. (called w_vector 
        # to be consistent with the notation in the Glove paper).
        self.w_vector = defaultdict(lambda: None)

        # Mapping from word IDs to (context) word vectors (called w_tilde_vector
        # to be consistent with the notation in the Glove paper)
        self.w_tilde_vector = defaultdict(lambda: None)

        # The ID of the latest encountered new word.
        self.latest_new_word = -1

        # Total number of tokens processed
        self.tokens_processed = 0

        # Dimension of word vectors.
        self.dimension = 50

        # The ID of the current focus word.
        self.focus_word_id = -1

        # The current token number.
        self.tokens_processed = 0

        # Cutoff for gradient descent.
        self.epsilon = 0.01

        # Initial learning rate.
        self.learning_rate = 0.05

        # The number of times we can tolerate that loss increases
        self.patience = 5
        
        # Temporary file used for storing the model
        self.temp_file = "temp__.txt"

        # added a window variable that is initialised to -1 for the total size of the left and right windows
        self.window = [-1 for i in range(left_window_size + right_window_size)]

        # Possibly continue training from pretrained vectors
        if continue_training and os.path.exists(self.temp_file):
            self.read_temp_file( self.temp_file )
        


    #--------------------------------------------------------------------------
    #
    #  Methods for processing all files and computing all counts
    #


    # Initializes the necessary information for a word.

    def init_word( self, word ) :

        self.latest_new_word += 1

        # This word has never been encountered before. Init all necessary
        # data structures.
        self.id2word[self.latest_new_word] = word
        self.word2id[word] = self.latest_new_word

        # Initialize arrays with random numbers in [-0.5,0.5].
        w = rand.rand(self.dimension)-0.5
        self.w_vector[self.latest_new_word] = w
        w_tilde = rand.rand(self.dimension)-0.5
        self.w_tilde_vector[self.latest_new_word] = w_tilde
        return self.latest_new_word



    # Slides in a new word in the local context window
    #
    # The local context is a list of length left_window_size+right_window_size.
    # Suppose the left window size and the right window size are both 2.
    # Consider a sequence
    #
    # ... this  is  a  piece  of  text ...
    #               ^
    #           Focus word
    #
    # Then the local context is a list [id(this),id(is),id(piece),id(of)],
    # where id(this) is the wordId for 'this', etc.
    #
    # Now if we slide the window one step, we get
    #
    # ... is  a  piece  of  text ...
    #              ^
    #         New focus word
    #
    # and the new context window is [id(is),id(a),id(of),id(text)].

    def get_word_id( self, word ) :
        """
        Returns the word ID for a given word. If the word has not
        been encountered before, the necessary data structures for
        that word are initialized.
        """
        word = word.lower()
        if word in self.word2id :
            return self.word2id[word]
        
        else :
            # This word has never been encountered before. Init all necessary
            # data structures
            self.latest_new_word += 1
            self.id2word[self.latest_new_word] = word
            self.word2id[word] = self.latest_new_word

            # Initialize arrays with random numbers in [-0.5,0.5].
            w = rand.rand(self.dimension)-0.5
            self.w_vector[self.latest_new_word] = w
            w_tilde = rand.rand(self.dimension)-0.5
            self.w_tilde_vector[self.latest_new_word] = w_tilde
            return self.latest_new_word

    def update_counts( self, focus_word, context ) :
        """
        Updates counts based on the local context window.
        """
        focus_word_id = self.get_word_id( focus_word )
        all_context_words = self.X[focus_word_id]
        if all_context_words == None:
            all_context_words = defaultdict(int)
            self.X[focus_word_id] = all_context_words
        for idx in context :
            count = all_context_words[idx]
            if count == None :
                count = 0
            all_context_words[idx] = count+1


    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.

        :param      i:      Index of the focus word in the list of tokens
        :type       i:      int
        """

        # REPLACE WITH YOUR CODE

        # let the context size be the length of the window
        context_size = len(self.window) 
        # let the start point be the maximum of 0 to the index minus the context size
        start = max(0, i - context_size)
        # let the end point be the min of the length of the tokens and the index plus the context size plus one
        end = min(len(self.tokens), i + context_size + 1)
        # let the context be the word at that id for each word in the start:end range if there is a valid word
        context = [self.word2id[word] for word in self.tokens[start:end] if word in self.word2id]
        return context

    def process_files( self, file_or_dir ) :
        """
        This function recursively processes all files in a directory.

        Each file is tokenized and the tokens are put in the list
        self.tokens. Then each token is processed through the methods
        'get_context' and 'update_counts' above.
        """
        if os.path.isdir( file_or_dir ) :
            for root,dirs,files in os.walk( file_or_dir ) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            print ( file_or_dir )
            stream = open( file_or_dir, mode='r', encoding='utf-8', errors='ignore' )
            text = stream.read()
            try :
                self.tokens = nltk.word_tokenize(text) 
            except LookupError :
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(text)
            for i, token in enumerate(self.tokens):
                self.tokens_processed += 1

                context = self.get_context(i)
                self.update_counts(token, context)
                
                if self.tokens_processed % 10000 == 0 :
                    print( 'Processed ' + str(self.tokens_processed) + ' tokens' )

        
    #
    #  Methods for processing all files and computing all counts
    #
    #--------------------------------------------------------------------------

    #--------------------------------------------------------------------------
    #
    #   Loss function, gradient descent, etc.
    #

    def f( self, count ) :
        """
        The "f" function from the Glove article
        """
        if count<100 :
            ratio = count/100.0
            return math.pow( ratio, 0.75 )
        return 1.0
    
    def loss( self ) :
        """
        Returns the total loss, computed from all the vectors.
        """

        # REPLACE WITH YOUR CODE
        # set the initial loss to 0
        loss = 0
        # let N be the length of the index 2 word
        N = len(self.id2word)
        # for i
        for i in range(N):
            # for j
            for j in range(N):
                # if the value at the i,j slot of X is 0
                if self.X[i][j] == 0:
                    # let the loss equal the f function of X times the dot product of the w vector and w tilde vector squared
                    loss += self.f(self.X[i][j]) * np.dot(self.w_vector[i], self.w_tilde_vector[j]) ** 2
                # otherwise, let the loss equal the same thing but minus the log of X inside the bracket
                else:
                    loss += self.f(self.X[i][j]) * (np.dot(self.w_vector[i], self.w_tilde_vector[j]) - np.log(self.X[i][j])) ** 2
        # half of this is the loss
        return loss / 2
    
    def compute_gradient(self, i, j) :
        """
        Computes the gradient of the loss function w.r.t. w_vector[i] and
        w.r.t. w_tilde_vector[j]

        Returns wi_vector_grad, wj_tilde_vector_grad
        """

        # REPLACE WITH YOUR CODE
        # let the wi_vector grad equal the appropriate formula from the article on Glove
        wi_vector_grad = self.f(self.X[i][j]) * (np.dot(self.w_vector[i], self.w_tilde_vector[j]) - math.log(self.X[i][j])) * self.w_tilde_vector[j]
        # similarly for wj_tilde_vector_grad
        wj_tilde_vector_grad = self.f(self.X[i][j]) * (np.dot(self.w_vector[i], self.w_tilde_vector[j]) - math.log(self.X[i][j])) * self.w_vector[i]

        return wi_vector_grad, wj_tilde_vector_grad

    def train( self ) :
        """
        Trains the vectors using stochastic gradient descent
        """
        iterations = 0

        # YOUR CODE HERE
        # let the max iterations be 2000000
        max_iteration = 2000000
        # calculate the previous loss
        prev_loss = self.loss()
        count = 0
        # initialise the probabilities to zero
        probs = np.zeros(len(self.id2word))
        # for each i and j in the id lists and the X lists respectively
        for i in range(len(self.id2word)):
            for j in range(len(self.X[i])):
                # set the probability equal to that slot of X (from Glove stuff)
                probs[i] += self.X[i][j]
        # print(prev_loss)
        # generate a list of i values randomly
        i_list = random.choices(range(len(self.id2word)), weights=probs, k=max_iteration)
        # for the total number of iterations
        for iterations in tqdm(range(max_iteration)):
            # let i be that slot of the list of i values
            i = i_list[iterations]
            # generate random j based on X
            j = random.choices(list(self.X[i].keys()), weights=list(self.X[i].values()))[0]
            # compute the gradient using the i and j values
            wi_vector_grad, wj_tilde_vector_grad = self.compute_gradient(i, j)
            # the w vector is the learning rate times the vector gradient
            self.w_vector[i] -= self.learning_rate * wi_vector_grad
            # the w tilde vector is the same but with the w tilde vector gradient
            self.w_tilde_vector[j] -= self.learning_rate * wj_tilde_vector_grad
            # this bit was provided
            if iterations%1000000 == 0 :
                self.write_word_vectors_to_file( self.outputfile )
                self.write_temp_file( self.temp_file )
                self.learning_rate *= 0.99           

    #
    #  End of loss function, gradient descent, etc.
    #
    #-------------------------------------------------------

    #-------------------------------------------------------
    #
    #  I/O
    #

    def write_word_vectors_to_file( self, filename ) :
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w') as f:
            for idx in self.id2word.keys() :
                f.write('{} '.format( self.id2word[idx] ))
                for i in self.w_vector[idx] :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()

    def write_temp_file( self, filename ) :
        """
        Saves the state of the computation to file, so that
        training can be resumed later.
        """
        with open(filename, 'w') as f:
            f.write('{} '.format( self.learning_rate ))
            f.write( '\n' )
            for idx in self.id2word.keys() :
                f.write('{} '.format( self.id2word[idx] ))
                for i in list(self.w_vector[idx]) :
                    f.write('{} '.format( i ))
                for i in list(self.w_tilde_vector[idx]) :
                    f.write('{} '.format( i ))
                f.write('\n')
        f.close()

    def read_temp_file(self, fname):
        """
        Reads the partially trained model from file, so
        that training can be resumed.
        """
        i = 0
        with open(fname) as f:
            self.learning_rate = float(f.readline())
            for line in f:
                data = line.split()
                print(len(data))
                w = data[0]
                vec = np.array([float(x) for x in data[1:self.dimension+1]])
                self.id2word[i] = w
                self.word2id[w] = i
                self.w_vector[i] = vec
                vec = np.array([float(x) for x in data[self.dimension+1:]])
                self.w_tilde_vector[i] = vec
                i += 1
        f.close()
        self.dimension = len( self.w_vector[0] )

    #
    #  End of I/O
    #
    #----------------------------------------------------------
def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Glove trainer')
    parser.add_argument('--file', '-f', type=str,  default='../glove/data', help='The files used in the training.')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')

    arguments = parser.parse_args()  
    
    glove = Glove(False, arguments.left_window_size, arguments.right_window_size)
    glove.outputfile = arguments.output
    glove.process_files( arguments.file )
    print( 'Processed', "{:,}".format(glove.tokens_processed), 'tokens' )
    print( 'Found', len(glove.word2id), 'unique words' )
    glove.train()

        
if __name__ == '__main__' :
    main()    
