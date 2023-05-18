from terminaltables import AsciiTable
import argparse

"""
The CKY parsing algorithm.

This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Johan Boye.
"""

class CKY :

    # The unary rules as a dictionary from words to non-terminals,
    # e.g. { cuts : [Noun, Verb] }
    unary_rules = {}

    # The binary rules as a dictionary of dictionaries. A rule
    # S->NP,VP would result in the structure:
    # { NP : {VP : [S]}} 
    binary_rules = {}

    # The parsing table
    table = []

    # The backpointers in the parsing table
    backptr = []

    # The words of the input sentence
    words = []


    # Reads the grammar file and initializes the 'unary_rules' and
    # 'binary_rules' dictionaries
    def __init__(self, grammar_file) :
        stream = open( grammar_file, mode='r', encoding='utf8' )
        for line in stream :
            rule = line.split("->")
            left = rule[0].strip()
            right = rule[1].split(',')
            if len(right) == 2 :
                # A binary rule
                first = right[0].strip()
                second = right[1].strip()
                if first in self.binary_rules :
                    first_rules = self.binary_rules[first]
                else :
                    first_rules = {}
                    self.binary_rules[first] = first_rules
                if second in first_rules :
                    second_rules = first_rules[second]
                    if left not in second_rules :
                        second_rules.append( left )
                else :
                    second_rules = [left]
                    first_rules[second] = second_rules
            if len(right) == 1 :
                # A unary rule
                word = right[0].strip()
                if word in self.unary_rules :
                    word_rules = self.unary_rules[word]
                    if left not in word_rules :
                        word_rules.append( left )
                else :
                    word_rules = [left]
                    self.unary_rules[word] = word_rules


    # Parses the sentence a and computes all the cells in the
    # parse table, and all the backpointers in the table
    def parse(self, s) :
        self.words = s.split()        
        #  YOUR CODE HERE
        # Create a 2D list for table with the number of words as its number of rows and also as its number of columns
        # Initiate each cell to an empty list [].
        self.table = [[[] for _ in range(len(self.words))]for _ in range(len(self.words))]
        # Same for backpointers
        self.backptr = [[[] for _ in range(len(self.words))]for _ in range(len(self.words))]

        # print(self.binary_rules) -- debugging
        
        # for each column (0 upwards) (set number of columns to the number of words)
        for j in range(len(self.words)):
            #print("j", j)
            # set the diagonal of the table equal to the unary rules for the word corresponding to that column
            self.table[j][j] = self.unary_rules[self.words[j]]
            # for i starting one less than j and decreasing until it reaches 0 (start, endpt, stepsize)
            for i in range(j-1, -1, -1):
                #print("i:", i)
                # for k from i up to (but not incl.) j (for handling leftmost with one below, second left with two below etc)
                for k in range(i, j):
                    #print("k:", k)
                    # flet a be the first term in the cell to the left of the current cell, then second term etc.
                    for a in self.table[i][k]:
                        #print("a:", a)
                        # if a is one of the keys in the binary rules (NP, JJ, Verb, Prep for 'giant cuts in welfare')
                        if a in self.binary_rules.keys():
                            # then, for the cell below the current cell, let b be its first value, then second value etc.
                            for b in self.table[k+1][j]:
                                #print("b:", b)
                                # if c is in the set of binary_rules at the slot of b
                                if b in self.binary_rules[a]:
                                    # set c to the term that branches to give a and b
                                    for c in self.binary_rules[a][b]:
                                        #print("c:", c)
                                        # add the term of c to the end of the list for that table slot
                                        self.table[i][j].append(c)
                                        # set the backpointer equal to the corresponding a and b terms and cells
                                        self.backptr[i][j].append(([a, i, k], [b, k+1, j]))

    # Prints the parse table
    def print_table( self ) :
        t = AsciiTable(self.table)
        t.inner_heading_row_border = False
        print( t.table ) 


    # Prints all parse trees derivable from cell in row 'row' and
    # column 'column', rooted with the symbol 'symbol'
    # changed the order of column and row to fix it
    def print_trees(self, row, column, symbol):
        #
        #  YOUR CODE HERE
        results = []
        # set the output to an empty string
        output = ""

        # if the row value is equal to the column (on the main diagonal) - this means we have reached the word to be printed
        if row == column:
            # concatenate the output with the current symbol and the word at that column position
            output = symbol + "(" + self.words[column] + ")"

        # otherwise, if the symbol is in the current table cell
        elif symbol in self.table[row][column]:
            # 
            # indices is a list of all the indices where the value in self.table[row][column] is equal to symbol
            # - i.e. where there will be valid trees to parse
            indices = [i for i, values in enumerate(self.table[row][column]) if values == symbol]
            #print(indices) -- debugging
            for values in indices:
                # allocate the symbol for the left and its row and column indices to the variables leftSym, rowLeft and colLeft
                symbolLeft, rowLeft, colLeft = self.backptr[row][column][values][0]
                symbolDown, rowDown, colDown = self.backptr[row][column][values][1]
                # if the output is not empty this means that it reached the end of a tree so print it
                if(output != ""):
                    print(output)
                    output = ""
                # concatenate the output with the current symbol
                output = output + symbol
                # concatenate the output with the result of recursively calling the tree going to the left 
                output = output + "(" + self.print_trees(rowLeft, colLeft, symbolLeft)
                # concatenate the output with the result of recursively calling the cells going down
                output = output +", " + self.print_trees(rowDown, colDown, symbolDown) + ")"
        
        # if the symbol entered is invalid print error message
        else: 
            print("Error: No trees can be formed starting with this symbol")

        # if this point reached, we are at the end of the last tree and so it should be printed
        # the absolute value between the row and column should always be the length of the number of words minus 1 if at end of tree
        if(abs(row-column) == len(self.words)-1):
            print(output)

        return output

def main() :

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CKY parser')
    parser.add_argument('--grammar', '-g', type=str,  required=True, help='The grammar describing legal sentences.')
    parser.add_argument('--input_sentence', '-i', type=str, required=True, help='The sentence to be parsed.')
    parser.add_argument('--print_parsetable', '-pp', action='store_true', help='Print parsetable')
    parser.add_argument('--print_trees', '-pt', action='store_true', help='Print trees')
    parser.add_argument('--symbol', '-s', type=str, default='S', help='Root symbol')

    arguments = parser.parse_args()

    cky = CKY( arguments.grammar )
    cky.parse( arguments.input_sentence )
    if arguments.print_parsetable :
        cky.print_table()
    if arguments.print_trees :
        # changed this to match with row and columns
        cky.print_trees(0, len(cky.words)-1, arguments.symbol)
    

if __name__ == '__main__' :
    main()    
