#---------------------------------------------------------------------------------------------------------------
# Part 1
class CSP:
    def __init__(self, board, c, size):
        #making a new dictionary to hold all the vars and constraints
        newDict = {}
        
        for i in range(len(board)):
            for j in range(len(board)):

                #creates a new key depending on the position of the board
                key = 'C' + str(i+1) + str(j+1)

                # check for a nil ssquare and sets it to all options, else give it the og value
                if(board[i][j] == 'nil' or board[i][j] == None):
                    newDict[key] = list(range(1, size + 1))
                else:
                    newDict[key] = [board[i][j]]
    
        self.constraints = c
        self.variables = newDict
#---------------------------------------------------------------------------------------------------------------
# Part 2
def revise(CSP, var1, var2):
    check = False

    if(var1 < var2):
        constraints = CSP.constraints[var1, var2]
    else:
        constraints = CSP.constraints[var2, var1]
    
    for value in CSP.variables[var1]:
        rm = True 
        for value2 in CSP.variables[var2]:
            pair = [value, value2]
            if(pair in constraints):
                rm = False

        if rm:
            check = True
            CSP.variables[var1].remove(value)

    return check
