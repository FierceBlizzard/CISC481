from memory_profiler import profile
import cProfile
from operator import attrgetter


# Node class in order to keep track of the state, its parents, and its move
class node():

    def __init__(self, par, move, puzzle, pathcost, depth):
        self.parent = par
        self.move = move
        self.puzzle = puzzle
        self.cost = pathcost
        self.depth = depth


# finds the nil in the puzzle and records it's coordinates
def find_nil(puzzle):
    for i in range(len(puzzle)):
        for j in range(len(puzzle)):
            if puzzle[i][j] == "nil":
                return (i, j)


# get all the possible actions we can do with the location
# of the nil
def possible_action(puzzle):
    possAct = []
    row, col = find_nil(puzzle)
    if col == 0:
        possAct.append("Left " + str(row) + " " + str(col + 1))
        if (row != 2):
            possAct.append("Up " + str(row + 1) + " " + str(col))
        if (row != 0):
            possAct.append("Down " + str(row - 1) + " " + str(col))
    elif col == 1:
        possAct.append("Right " + str(row) + " " + str(col - 1))
        possAct.append("Left " + str(row) + " " + str(col + 1))
        if (row != 2):
            possAct.append("Up " + str(row + 1) + " " + str(col))
        if (row != 0):
            possAct.append("Down " + str(row - 1) + " " + str(col))
    elif (col == 2):
        possAct.append("Right " + str(row) + " " + str(col - 1))
        if (row != 2):
            possAct.append("Up " + str(row + 1) + " " + str(col))
        if (row != 0):
            possAct.append("Down " + str(row - 1) + " " + str(col))
    return possAct


# does the action and give's us the resulting puzzle
def result(action, puzzle):
    newPuzzle = [list(place) for place in puzzle]
    newList = action.split()
    i, j = find_nil(newPuzzle)

    value = newPuzzle[int(newList[1])][int(newList[2])]
    newPuzzle[i][j] = value
    newPuzzle[int(newList[1])][int(newList[2])] = "nil"

    return newPuzzle


# get's us all the possible nodes from the current position
def expand(puzzle):
    actions = possible_action(puzzle)
    puzzleNodes = []
    for move in actions:
        puzzleNodes.append(result(move, puzzle))
    return puzzleNodes


# depth first search algo
# calls iterative deepening but we just run it
# until we find the solution
def depthFirst(puzzle, goal):
    maxDepth = 1
    actions = []
    while True:
        result, actions = iterative_deepening(puzzle, goal, 0, maxDepth, actions)
        if result is not None:
            print(actions)
            print(goal)
            return actions
        maxDepth += 1
    return None


# iterative deepening algo
# goes through and looks at all the branches
# and sees if there is a solution at the allowed depth
def iterative_deepening(puzzle, goal, currentDepth, maxDepth, actions):
    possAct = possible_action(puzzle)
    children = expand(puzzle)

    if (puzzle == goal):
        return puzzle, actions

    if (currentDepth == maxDepth):
        return None, actions

    for i in range(len(children)):
        result, actions = iterative_deepening(children[i], goal, currentDepth + 1, maxDepth, actions)

        actions.append(possAct[i])

        if result is not None:
            return result, actions

        actions.pop()

    return None, []


# this is a helper function which makes our lives a lot easier
# helps us hash the node
def createHash(node):
    hashString = ""
    for i in range(len(node.puzzle)):
        for j in range(len(node.puzzle)):
            if (node.puzzle[i][j] == "nil"):
                hashString += "0"
            hashString += str(node.puzzle[i][j])
    return hashString


# find node path just recursives called to find the path it took
def findNodePath(node):
    finalMoveList = []
    currentNode = node

    while currentNode.parent is not None:
        finalMoveList.append(currentNode.move)
        currentNode = currentNode.parent

    finalMoveList.reverse()

    print(finalMoveList)
    return finalMoveList


# bfs algo
def breadthFirstSearch(puzzle, goal):
    firstNode = node(None, None, puzzle, None, None)

    reached = {}
    queue = [firstNode]

    # While loop for bfs
    while queue:
        state = queue.pop(0)
        hashmark = createHash(state)

        if (state.puzzle == goal):
            finalPath = findNodePath(state)
            return finalPath
        elif hashmark in reached:
            continue
        else:
            reached[createHash(state)] = 1

        newChildren = expand(state.puzzle)
        newMove = possible_action(state.puzzle)

        for i in range(len(newChildren)):
            queue.append(node(state, newMove[i], newChildren[i], None, None))

    return None


# heuristic when it's 0
def heuristic0(puzzle, goal):
    return 0


# helper function to compare the puzzle to the goal
def checker(puzzle, goal):
    totalDiff = 0
    for i in range(len(puzzle)):
        for j in range(len(puzzle)):
            if (puzzle[i][j] != goal[i][j]):
                totalDiff += 1
    return totalDiff


# manhattant distance algo
def manhattanDist(puzzle, goal):
    manhatDiff = 0
    for i in range(len(puzzle)):
        for j in range(len(puzzle)):
            if (puzzle[i][j] != goal[i][j]):
                if (puzzle[i][j] == 'nil'):
                    row = 0 // len(puzzle)
                    col = 0 % len(puzzle)
                else:
                    row = int(puzzle[i][j]) // len(puzzle)
                    col = int(puzzle[i][j]) % len(puzzle)
                manhatDiff += abs((row - i)) + abs((col - j))
    return manhatDiff


# A star search algo with the huerisitic
def A_Star_Search(Puzzle, goal, hFunc):
    rootNode = node(None, None, Puzzle, 0, 0)
    reached = {}
    nodeList = [rootNode]

    while nodeList:
        state = nodeList.pop(0)
        hashmark = createHash(state)

        if (state.puzzle == goal):
            findNodePath(state)
            return 1
        elif hashmark in reached:
            continue
        else:
            reached[createHash(state)] = 1

        newChildren = expand(state.puzzle)
        newMove = possible_action(state.puzzle)
        newDepth = state.depth + 1

        for i in range(len(newChildren)):
            nodeList.append(node(state, newMove[i], newChildren[i], hFunc(newChildren[i], goal) + newDepth, newDepth))

        nodeList = sorted(nodeList, key=attrgetter("cost"))


def main():
    # @profile(precision=4)

    puzzle0 = [[3, 1, 2], [7, "nil", 5], [4, 6, 8]]
    puzzle1 = [[7, 2, 4], [5, "nil", 6], [8, 3, 1]]
    puzzle2 = [[6, 7, 3], [1, 5, 2], [4, "nil", 8]]
    puzzle3 = [["nil", 8, 6], [4, 1, 3], [7, 2, 5]]
    puzzle4 = [[7, 3, 4], [2, 5, 1], [6, 8, "nil"]]
    puzzle5 = [[1, 3, 8], [4, 7, 5], [6, "nil", 2]]
    puzzle6 = [[8, 7, 6], [5, 4, 3], [2, 1, "nil"]]
    goal = [["nil", 1, 2], [3, 4, 5], [6, 7, 8]]
    biggerGoal = [['nil', 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    biggerPuzzle = [[13, 10, 11, 6], [5, 7, 4, 8], [1, 12, 14, 9], [3, 15, 2, 'nil']]
    biggerOneMove = [[1, 'nil', 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    depthFirst(puzzle0, goal)

    # breadthFirstSearch(puzzle0, goal)
    # breadthFirstSearch(puzzle1, goal)

    # A_Star_Search(puzzle0, goal, comparePuzzle)
    # A_Star_Search(puzzle1, goal, comparePuzzle)

    # A_Star_Search(puzzle0, goal, manhattanDist)
    # A_Star_Search(puzzle6, goal, manhattanDist)


if __name__ == "__main__":
    cProfile.run('main()')
