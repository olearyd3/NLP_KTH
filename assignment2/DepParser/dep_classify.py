import os
import pickle
from parse_dataset import Dataset
from dep_parser import Parser
from logreg import LogisticRegression


class TreeConstructor:
    """
    This class builds dependency trees and evaluates using unlabeled arc score (UAS) and sentence-level accuracy
    """
    def __init__(self, parser):
        self.__parser = parser

    def build(self, model, words, tags, ds):
        """
        Build the dependency tree using the logistic regression model `model` for the sentence containing
        `words` pos-tagged by `tags`
        
        :param      model:  The logistic regression model
        :param      words:  The words of the sentence
        :param      tags:   The POS-tags for the words of the sentence
        :param      ds:     Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE
        # create empty lists for moves and stack as well as variables for i and pred_tree -- same as dep_parser
        moves = []
        i, stack, pred_tree = 0, [], [0] * len(words)
        # convert the datapoints to an array -- words are the words, tags are the pos tags of the sentence, i is always 0 and stack empty
        arr = ds.dp2array(words, tags, i, stack)
        # get the log probabilities for the words in this array
        probabilities = model.get_log_probs(arr)

        # get ranking of the probabilities
        # start the best probability at -1 and the best move as None
        bestProbability = -1
        bestMove = None
        # for each move and probability in the probabilities list
        for move, prob in enumerate(probabilities):
            # print("move", move, "prob", prob)
            # if the move is a valid move
            if move in self.__parser.valid_moves(i, stack, pred_tree):
                # if the probability is higher than the current best probability
                if prob > bestProbability:
                    # set the bestProbability equal to probability and bestMove equal to move
                    bestProbability = prob
                    bestMove = move
        # append the bestMove to the list of moves
        moves.append(bestMove)
        # set the values for i, stack and pred_tree to the result of the move function from dep_parser
        i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, moves[-1])
        
        # while there are still moves
        while True:
            # add a datapoint
            ds.add_datapoint(words, tags, i, stack, False)
            # do the same as before the While loop
            arr = ds.dp2array(words, tags, i, stack)
            # get the log probabilities for the words in this array
            probabilities = model.get_log_probs(arr)
            # all of this is the same as before the while loop
            bestProbability = -1
            bestMove = None
            for move, prob in enumerate(probabilities):
                if move in self.__parser.valid_moves(i, stack, pred_tree):
                    if prob > bestProbability:
                        bestProbability = prob
                        bestMove = move

            moves.append(bestMove)
            
            # pop the last term
            ds.datapoints.pop(0)
            # update the variables as before
            i, stack, pred_tree = self.__parser.move(i, stack, pred_tree, moves[-1])
            # if the stack is only 1 word, break
            if len(stack) == 1:
                break

        return moves

    def evaluate(self, model, test_file, ds):
        """
        Evaluate the model on the test file `test_file` using the feature representation given by the dataset `ds`
        
        :param      model:      The model to be evaluated
        :param      test_file:  The CONLL-U test file
        :param      ds:         Training dataset instance having the feature maps
        """
        #
        # YOUR CODE HERE

        # set p equal to the parsed data
        p = self.__parser
        # define all necessary variables as 0
        arc = 0
        correctArc = 0
        sentenceCount = 0
        totalSentences = 0
        # with the file open and using UTF-8 encoding
        with open(test_file, encoding="utf-8") as source:
            # UNDERSTAND THIS
            for words, tags, tree, relations in p.trees(source):
                # determine the sequence of correct moves required to form the correct tree
                correct_moves = p.compute_correct_moves(tree)
                # add 1 to the total number of sentences
                totalSentences += 1
                
                moves = self.build(model, words, tags, ds)
                flag = 1
                for i in range(len(correct_moves)):
                    if correct_moves[i] != 0:
                        arc = arc + 1
                    if moves[i] == correct_moves[i]:
                        if moves[i] != 0:
                            correctArc += 1
                    else:
                        flag = 0
                if flag == 1:
                    sentenceCount += 1
        # sentence level accuracy as a percentage is 100 times the sentence count divided by the sentence total -- gives ratio of correctly parsed sentences
        print("Sentence-level accuracy:", (sentenceCount / totalSentences)*100, "%")
        # UAS is the ratio of correctly assigned arcs -- correct arcs divided by total arcs times 100
        print("UAS:", (correctArc/arc)*100, "%")
        # move accuracy is the number of correct moves divided by the total number of moves

if __name__ == '__main__':

    # Create parser
    p = Parser()

    # Create training dataset
    ds = p.create_dataset("en-ud-train-projective.conllu", train=True)

    # Train LR model
    if os.path.exists('model.pkl'):
        # if model exists, load from file
        print("Loading existing model...")
        lr = pickle.load(open('model.pkl', 'rb'))
    else:
        # train model using minibatch GD
        lr = LogisticRegression()
        lr.fit(*ds.to_arrays())
        pickle.dump(lr, open('model.pkl', 'wb'))
    
    # Create test dataset
    test_ds = p.create_dataset("en-ud-dev-projective.conllu")
    # Copy feature maps to ensure that test datapoints are encoded in the same way
    test_ds.copy_feature_maps(ds)
    # Compute move-level accuracy
    lr.classify_datapoints(*test_ds.to_arrays())
    
    # Compute UAS and sentence-level accuracy
    t = TreeConstructor(p)
    t.evaluate(lr, 'en-ud-dev-projective.conllu', ds)
