# Still tryong to fix some of the imports 
import copy 
from flask import Flask, request, render_template
from board4x4 import constraint4x4
from board9x9 import constraint9x9

class SetBoard:
    def __init__(self, board, constraints):
         values = {}

         # Setting up the values and the constraints
         for row in range(len(board)):
           for col in range(len(board)):
               if board[row][col] != None:
                   values["C"+str(row+1)+str(col+1)] = board[row][col]
               else:
                   values["C"+str(row+1)+str(col+1)] = None
         # Setting attributes
         self.values = values
         self.constraints = constraints

# Part #1

fourbyfour = [[1, None, None, None],
             [None, 2, None, None],
             [None, None, 3, None],
             [None, None, None, 4]]

puzzle1 = [[7,None,None,4,None,None,None,8,6],
            [None,5,1,None,8,None,4,None,None],
            [None,4,None,3,None,7,None,9,None],
            [3,None,9,None,None,6,1,None,None],
            [None,None,None,None,2,None,None,None,None],
            [None,None,4,9,None,None,7,None,8],
            [None,8,None,1,None,2,None,6,None],
            [None,None,6,None,5,None,9,1,None]]

puzzle2 = [[1,None,None,2,None,3,8,None,None],
            [None,8,2,None,6,None,1,None,None],
            [7,None,None,None,None,1,6,4,None],
            [3,None,None,None,9,5,None,2,None],
            [None,7,None,None,None,None,None,1,None],
            [None,9,None,3,1,None,None,None,6],
            [None,5,3,6,None,None,None,None,1],
            [None,None,7,None,2,None,3,9,None],
            [None,None,4,1,None,9,None,None,5]]

puzzle3 = [[1,None,None,8,4,None,None,5,None],
            [5,None,None,9,None,None,8,None,3],
            [7,None,None,None,6,None,1,None,None],
            [None,1,None,5,None,2,None,3,None],
            [None,7,5,None,None,None,2,6,None],
            [None,3,None,6,None,9,None,4,None],
            [None,None,7,None,5,None,None,None,6],
            [4,None,1,None,None,6,None,None,7],
            [None,6,None,None,9,4,None,None,2]]

puzzle4 = [[None,None,None,None,9,None,None,7,5],
            [None,None,1,2,None,None,None,None,None],
            [None,7,None,None,None,None,1,8,None],
            [3,None,None,6,None,None,9,None,None],
            [1,None,None,None,5,None,None,None,4],
            [None,None,6,None,None,2,None,None,3],
            [None,3,2,None,None,None,None,4,None],
            [None,None,None,None,None,6,5,None,None],
            [7,9,None,None,1,None,None,None,None]]

puzzle5 = [[None,None,None,None,None,6,None,8,None],
            [3,None,None,None,None,2,7,None,None],
            [7,None,5,1,None,None,6,None,None],
            [None,None,9,4,None,None,None,None,None],
            [None,8,None,None,9,None,None,2,None],
            [None,None,None,None,None,8,3,None,None],
            [None,None,4,None,None,7,8,None,5],
            [None,None,2,8,None,None,None,None,6],
            [None,5,None,9,None,None,None,None,None]]

# Board/Puzzle Combinations
Board4x4 = SetBoard(fourbyfour,constraint4x4)
Board9x9_1 = SetBoard(puzzle1,constraint9x9)
Board9x9_2 = SetBoard(puzzle2,constraint9x9)
Board9x9_3 = SetBoard(puzzle3,constraint9x9)
Board9x9_4 = SetBoard(puzzle4,constraint9x9)
Board9x9_5 = SetBoard(puzzle5,constraint9x9)

def printPuzzle(board):
    puzzle = [[], [], [], [], [], [], [], [], []]
    for key in board:
        row = int(key[1]) - 1
        puzzle[row].append(board[key])
    for row in range(len(puzzle)):
        print(puzzle[row])
    return
# Part 2
'''
Function revise:
    Changes the first cells based on the domain of a cell and it's neightbor, 
    if the combination of the two cells is outside the constraints then we 
    reverse order to check it
'''
def revise(C: SetBoard, A1: str, A2: str) -> bool:
    if A1 < A2:
        constraint = C.constraints[A1,A2]
    else:
        constraint = C.constraints[A2,A1]
    # Boolean showing if A1 domain is changed
    revised = False

    for i in C.values[A1]:
        cell = False  # used to check if it needs to be removed from domain or not
        for j in C.values[A2]:
            # Check if the combination is in the domain 
            if [i,j] in constraint:
                cell = True
                break 
            if cell == False:
                C.values[A1].remove(i)
                revised = True
        return revised

# Part 3
'''
Function AC3:
    Looks at all the cell domains within the board and removes any values
    that are inconsistent and don't match up with the rest. 
'''
def AC3(C: SetBoard) -> bool:
    # queue of all the constraints
    queue = []
    for i in C.constraints:
        queue.append(i)
    while queue != []:
        constraint = queue.pop()
        if revise(C,constraint[0],constraint[1]):
            if C.values[constraint[0]] == []:
                return False
            for i in C.constraints:
                if constraint[0] in i:
                    queue.append(i)
        if revise(C,constraint[1],constraint[0]):
            if C.values[constraint[1]] == []:
                return False
            for i in C.constraints:
                if constraint[1] in i:
                    queue.append(i)
    return True

# Part 4
'''
Function MRV:
    Goes throught all the values looking for the smallest
    value in the doamin and returns it
'''
def MRV(C: SetBoard, assign: dict) -> str:
    # making empty cell for now
    minCell = ''
    for atr, value in C.values.items():
        if atr not in assign and C.constraints[atr] == None:
            if minCell == '':
                # if we found nothing and minCell still empty, we set it to the art
                minCell = atr
            elif len(value) < len(C.values[minCell]):
                # if not empty, we compare the two and if atr is smaller then we change 
                # minCell to be atr since it's the smallest value found so far
                minCell = atr
    return minCell
# Part 5
'''
Function assign:
    Makes a dictionary so we can store a values or no value for each cell
'''
def assign(C: SetBoard):
    assigned = {}
    for key in C.values:
        if len(C.values[key]) == 1:
            assigned[key] = C.values[key]
        else:
            assigned[key] = None
    return assigned
'''
Function complete:
    Checks to see if a assigned dictionary has a complete set
'''
def complete(C: SetBoard, assigned):
    for key in C.values:
        if key not in assigned:
            return False
    return True

'''
Function backtracking:
    creates an initial board, runs backTrackingSearch and returns the final solution
'''
def backTracking(C: SetBoard):
    AC3(C)
    assign = assign(C)
    return backtrackingSearch(C, assign)
'''
Function backTrackingSearch:
    Finds the smalled cell that is unsolved in the board and tests different values in the 
    cells domain and picks the best one
'''
def backtrackingSearch(C: SetBoard, assign):
    if complete(C, assign):
        return assign
    key = MRV(C, assign)
    DC = copy.deepcopy(C.values)
    for val in C.values[key]:
        assign[key] = val
        C.values[key] = [val]
        testing = AC3(C)
        if testing:
            testing2 = backTracking(C, assign)
            if testing2:
                return testing2
        C.values = DC
# Part 6 - Web stuff
app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        puzzle = [[None for x in range(9)] for y in range(9)]
        for i in range(9):
            for j in range(9):
                puzzle[i][j] = request.form["C"+str(i+1)+str(j+1)]
        #turn string board into int board
        for row in range(len(puzzle)):
            for col in range(len(puzzle)):
                if puzzle[row][col] == '':
                    puzzle[row][col] = None
                elif 0 < int(puzzle[row][col]) < 10:
                    puzzle[row][col] = int(puzzle[row][col])
                else:
                    puzzle[row][col] = None
        board = backTracking(C(puzzle, constraint9x9))
        return render_template("solve.html", solution=board)
    return render_template("sudoku.html")    
if __name__=='__main__':
    app.run()



