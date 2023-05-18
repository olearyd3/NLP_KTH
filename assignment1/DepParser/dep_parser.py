from pathlib import Path
from parse_dataset import Dataset
import argparse

class Parser: 
    SH, LA, RA = 0,1,2

    def conllu(self, source):
        buffer = []
        for line in source:
            line = line.rstrip()    # strip off the trailing newline
            if not line.startswith("#"):
                if not line:
                    yield buffer
                    buffer = []
                else:
                    columns = line.split("\t")
                    if columns[0].isdigit():    # skip range tokens
                        buffer.append(columns)

    def trees(self, source):
        """
        Reads trees from an input source.

        Args: source: An iterable, such as a file pointer.

        Yields: Triples of the form `words`, `tags`, heads where: `words`
        is the list of words of the tree (including the pseudo-word
        <ROOT> at position 0), `tags` is the list of corresponding
        part-of-speech tags, and `heads` is the list of head indices
        (one head index per word in the tree).
        """
        for rows in self.conllu(source):
            words = ["<ROOT>"] + [row[1] for row in rows]
            tags = ["<ROOT>"] + [row[3] for row in rows]
            tree = [0] + [int(row[6]) for row in rows]
            relations = ["root"] + [row[7] for row in rows]
            yield words, tags, tree, relations


    def step_by_step(self,string) :
        """
        Parses a string and builds a dependency tree. In each step,
        the user needs to input the move to be made.
        """
        w = ("<ROOT> " + string).split()
        i, stack, pred_tree = 0, [], [0]*len(w) # Input configuration
        while True :
            print( "----------------" )
            print( "Buffer: ", w[i:] )
            print( "Stack: ", [w[s] for s in stack] )
            print( "Predicted tree: ", pred_tree )
            try :
                ms = input( "Move: (Shift,Left,Right): " ).lower()[0]
                m = Parser.SH if ms=='s' else Parser.LA if ms=='l' else Parser.RA if ms=='r' else -1
                if m not in self.valid_moves(i,stack,pred_tree) :
                    print( "Illegal move" )
                    continue
            except :
                print( "Illegal move" )
                continue
            i, stack, pred_tree = self.move(i,stack,pred_tree,m)
            if i == len(w) and stack == [0] :
                # Terminal configuration
                print( "----------------" )
                print( "Final predicted tree: ", pred_tree )
                return

    def create_dataset(self, source, train=False) :
        """
        Creates a dataset from all parser configurations encountered
        during parsing of the training dataset.
        (Not used in assignment 1).
        """
        ds = Dataset()
        with open(source) as f:
            for w,tags,tree,relations in self.trees(f): 
                i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
                m = self.compute_correct_move(i,stack,pred_tree,tree)
                while m != None :
                    ds.add_datapoint(w, tags, i, stack, m, train)
                    i,stack,pred_tree = self.move(i,stack,pred_tree,m)
                    m = self.compute_correct_move(i,stack,pred_tree,tree)
        return ds
   


    def valid_moves(self, i, stack, pred_tree):
        """Returns the valid moves for the specified parser
        configuration.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
        
        Returns:
            The list of valid moves for the specified parser
                configuration.
        """
        moves = []

        # YOUR CODE HERE

        # pred_tree contains all words but no arcs initially
        # if there is a word in the buffer, shift can be performed.
        # i.e. if the index is less than the length of the number of words, SH can be done
        if i < len(pred_tree):
            moves.append(0)
        # If root and two or more other words are in the stack, Left Arc can be performed
        if len(stack) > 2:
            moves.append(1)
        # If there are two or more words in the stack (incl. root), Right Arc can be performed
        if len(stack) > 1:
            moves.append(2)
        
        return moves

        
    def move(self, i, stack, pred_tree, move):
        """
        Executes a single move.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            move: The move that the parser should make.
        
        Returns:
            The new parser configuration, represented as a triple
            containing the index of the new first unprocessed word,
            stack, and partial dependency tree.
        """

        # YOUR CODE HERE
        # if next move is SH
        if move == 0:
            # add the word to the stack
            stack.append(i)
            # go to the next word
            i += 1
        # if next move is LA
        elif move == 1:
            # set the target equal to the second most recent word
            targetPos = len(stack)-2
            # set the head equal to the most recent word
            headPos = len(stack)-1
            # assign head to the position that the target was at
            # set the targetWord to the targetPos'th index of stack
            targetWord = stack[targetPos]
            # set the head to the headPos'th index of stack
            headWord = stack[headPos]
            # set the targetWord'th slot of pred_tree to headWord
            pred_tree[targetWord] = headWord
            # remove the word the arrow points to from the stack
            stack.pop(targetPos)
        # if next move is RA
        else:
            targetPos = len(stack)-1
            headPos = len(stack)-2
            # set the targetWord to the targetPos'th index of stack
            targetWord = stack[targetPos]
            # set the head to the headPos'th index of stack
            headWord = stack[headPos]
            # set the targetWord'th slot of pred_tree to headWord
            pred_tree[targetWord] = headWord

            stack.pop(targetPos)

        return i, stack, pred_tree

    # function to update the pred_tree - i.e. 
    def update_pred_tree(self, pred_tree, stack, targetPos, headPos):
        # set the targetWord to the targetPos'th index of stack
        targetWord = stack[targetPos]
        # set the head to the headPos'th index of stack
        headWord = stack[headPos]
        # set the targetWord'th slot of pred_tree to headWord
        pred_tree[targetWord] = headWord
        return pred_tree


    def compute_correct_moves(self, tree):
        """
        Computes the sequence of moves (transformations) the parser 
        must perform in order to produce the input tree.
        """
        i, stack, pred_tree = 0, [], [0]*len(tree) # Input configuration
        moves = []
        m = self.compute_correct_move(i,stack,pred_tree,tree)
        while m != None :
            moves.append(m)
            i,stack,pred_tree = self.move(i,stack,pred_tree,m)
            m = self.compute_correct_move(i,stack,pred_tree,tree)
        return moves


    def compute_correct_move(self, i, stack, pred_tree, correct_tree) :
        """
        Given a parser configuration (i,stack,pred_tree), and 
        the correct final tree, this method computes the  correct 
        move to do in that configuration.
    
        See the textbook, chapter 18.2.1.
        
        Args:
            i: The index of the first unprocessed word.
            stack: The stack of words (represented by their indices)
                that are currently being processed.
            pred_tree: The partial dependency tree.
            correct_tree: The correct dependency tree.
        
        Returns:
            The correct move for the specified parser
            configuration, or `None` if no move is possible.
        """
        assert len(pred_tree) == len(correct_tree)

        # YOUR CODE HERE

        # call the valid_moves function from before to get the list of valid moves that can be performed
        validMoves = self.valid_moves(i, stack, pred_tree)
        # if only 1 valid Move can be performed, return that
        if len(validMoves) == 1:
            return validMoves[0]

        # if the length of the stack is greater than 1, 
        if len(stack) > 1:
            # let s1 be the last item in the stack and let s2 be the second last number in the stack
            s1 = stack[-1]
            s2 = stack[-2]

            # if the left arc is a valid move and the second last member of the stack's slot in correct_tree is the last 
            # member in the stack, return left arc
            if self.LA in validMoves and correct_tree[s2] == s1:
                return self.LA
            # if the right arc is a valid move and the last member of the stack's slot in correct_tree is equal to the 
            # second last member of the stack, AND the number in the stack at the slot in correct_tree and pred_tree is same
            # then return right arc
            if self.RA in validMoves and correct_tree[s1] == s2 and correct_tree.count(s1) == pred_tree.count(s1):
                return self.RA

        # if LA or RA were valid, they'd have already been returned, so if SH is in validMoves, return it.
        if self.SH in validMoves:
            return self.SH
        # return None if no valid moves
        return None
  
filename = Path("en-ud-dev-projective.conllu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transition-based dependency parser')
    parser.add_argument('-s', '--step_by_step', type=str, help='step-by-step parsing of a string')
    parser.add_argument('-m', '--compute_correct_moves', type=str, default=filename, help='compute the correct moves given a correct tree')
    args = parser.parse_args()

    p = Parser()
    if args.step_by_step:
        p.step_by_step( args.step_by_step )

    elif args.compute_correct_moves:
        with open(args.compute_correct_moves, encoding='utf-8') as source:
            for w,tags,tree,relations in p.trees(source) :
                print( p.compute_correct_moves(tree) )





