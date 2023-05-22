import math
import numpy as np
StartBoard = [
    0, 
    1, 1, 1,
    0, 0, 0,
    -1, -1, -1
]

# Part 1
'''
Move: returns 1 if it's white's turn and -1 if it's black's turn

turn - Whose ever turn it is
'''
def to_move(board):
    return board[0]

'''
changeturn: support function that changes whose turn it is

turn - Whose ever turn it is
'''
def changeTurn(board):
    return -board[0]

'''
actions: gets all the possible actions

turn - Whose ever turn it is
board - the current turn of the board
'''  
def actions(turn, board):
    possible_actions = []
    location = []
    if to_move(turn) == 1:
        for i in range(1, len(board)):  # Start from index 1
            if board[i] == 1:
                location.append(i)
        for i in location:
            if i + 3 <= 9 and i + 4 <= 9:
                if i != 3 and i != 6:
                    if board[i + 3] == 0:
                        possible_actions.append(("Forward", i))
                    if board[i + 4] == -1:
                        possible_actions.append(("TakePiece", i))
                if i == 3:
                    if board[i + 2] == -1:
                        possible_actions.append(("TakePiece", i))
                    if board[i + 3] == 0:
                        possible_actions.append(("Forward", i))
                if i == 6:
                    if board[i + 2] == -1:
                        possible_actions.append(("TakePiece", i))
                    if board[i + 3] == 0:
                        possible_actions.append(("Forward", i))
        return possible_actions

    if to_move(turn) == -1:
        for i in range(1, len(board)):  # Start from index 1
            if board[i] == -1:
                location.append(i)
        for i in location:
            if i - 3 >= 1 and i - 4 >= 1:
                if i != 6 and i != 9:
                    if board[i - 3] == 0:
                        possible_actions.append(("Forward", i))
                    if board[i - 2] == 1:
                        possible_actions.append(("TakePiece", i))
                if i == 6:
                    if board[i - 2] == 1:
                        possible_actions.append(("TakePiece", i))
                    if board[i - 3] == 0:
                        possible_actions.append(("Forward", i))
                if i == 9:
                    if board[i - 2] == 1:
                        possible_actions.append(("TakePiece", i))
                    if board[i - 3] == 0:
                        possible_actions.append(("Forward", i))
        return possible_actions

'''
result - returns the updated board after the selected action of a pawn

turn - Whose ever turn it is
pawnIndex - index of the pawn that is being moved
action - The action that we have to happen
board - the current board of the game
'''
def result(turn, pawnIndex, action, board):
    next_turn = changeTurn(turn)  # Get the next turn
    if board[0] == 1:
        if action == "Forward":
            board[pawnIndex] = 0
            board[pawnIndex + 3] = 1
        if action == "TakePiece":
            board[pawnIndex] = 0
            board[pawnIndex + 4] = 1
    if board[0] == -1:
        if action == "Forward":
            board[pawnIndex] = 0
            board[pawnIndex - 3] = -1
        if action == "TakePiece":
            board[pawnIndex] = 0
            board[pawnIndex - 4] = -1
    board[0] = next_turn  # Update the turn
    return board

'''
terminal - checks if the game has reached a terminal state

board - the current state of the board
turn - Whose ever turn it is
'''
def terminal(board, turn):
    if board[0] == 1:
        if board[1] == -1 or board[2] == -1 or board[3] == -1 or not actions(turn, board):
            return True
    if board[0] == -1:
        if board[7] == 1 or board[8] == 1 or board[9] == 1 or not actions(turn, board):
            return True
    return False

'''
utility - returns the utility value for the given terminal state

board - the current state of the board
turn - Whose ever turn it is
'''
def utility(board, turn):
    if terminal(board, turn):
        if board[0] == 1:
            if board[1] == -1 or board[2] == -1 or board[3] == -1:
                return -1
            if not actions(-1, board):
                return -1
        if board[0] == -1:
            if board[7] == 1 or board[8] == 1 or board[9] == 1:
                return 1
            if not actions(-1, board):
                return 1
    return 0

# Part 2 Min-Max
'''
Min_Value - returns the minimum utility value and corresponding action for the current state

board - the current state of the board
turn - Whose ever turn it is
'''
def Min_Value(board, turn):
    if terminal(board, turn):
        return utility(board, turn), None
    v = math.inf
    for action, pawnIndex in actions(turn, board):
        new_board = result(board, pawnIndex, action, turn)
        v2, a2 = Max_Value(new_board, changeTurn(board, turn))
        if v2 < v:
            v, move = v2, (action, pawnIndex)
    return v, move

'''
Max_Value - returns the maximum utility value and corresponding action for the current state

board - the current state of the board
turn - Whose ever turn it is
'''
def Max_Value(board, turn):
    if terminal(board, turn):
        return utility(board, turn), None
    v = -math.inf
    for action, pawnIndex in actions(turn, board):
        new_board = result(board, pawnIndex, action, turn)
        v2, a2 = Min_Value(new_board, changeTurn(board, turn))
        if v2 > v:
            v, move = v2, (action, pawnIndex)
    return v, move

'''
MinMax_Search - performs the Min-Max search to find the best move for the current state

board - the current state of the board
turn - Whose ever turn it is
'''
def MinMax_Search(board, turn):
    values = {}
    value, _ = Max_Value(board, turn)

    for action, pawnIndex in actions(turn, board):
        new_board = result(board, pawnIndex, action, turn)
        _, new_value = Min_Value(new_board, changeTurn(board, turn))
        if new_value not in values:
            values[new_value] = []
        values[new_value].append(action)

    return value, values

# Part 3 -> 5
# class for each nueron incoming and outcoming

# a neuron in the nueral network
class Neuron: 
    def __init__(self, activation_function, weight):
        self.activation_function = activation_function
        self.value = 0
        self.bias = 0
        self.incoming_edges = []
        self.outgoing_edges = []
        self.weight = weight

# the edges the connects the neurons in the nueral network 
class Edge:
    def __init__(self, source_neuron, target_neuron, weight):
        self.source_neuron = source_neuron
        self.target_neuron = target_neuron
        self.weight = weight

# the whole nueral network
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.neurons = []

    # adds a neuron to the network
    def add_neuron(self, neuron):
        self.neurons.append(neuron)
    
    '''
    add_edge - Adds an edge connecting two neurons
    
    source_neuron - The source neuron of the edge
    target_neuron -The target neuron of the edge
    weight - The weight of the edge
    value - The value of the edge
    error - The error of the edge
    '''
    def add_edge(self, source_neuron, target_neuron, weight, bias, activation_function):
        edge = Edge(source_neuron, target_neuron, weight)
        source_neuron.outgoing_edges.append(edge)
        target_neuron.incoming_edges.append(edge)
        target_neuron.bias = bias
        target_neuron.activation_function = activation_function
    
    '''
    preprocessed_board - The preprocessed board state
    
    board - The current state of the board
    '''
    def preprocess_board(self, board):
        preprocessed_board = []
        turn = board[0]
        for i in range(1, len(board)):
            if board[i] == 0:
                preprocessed_board.append(0)
            elif board[i] == 1:
                preprocessed_board.extend([1, 1, 1])
            elif board[i] == -1:
                preprocessed_board.extend([-1, -1, -1])
        return preprocessed_board
    
    '''
    sigmoid - output value after applying the sigmoid activation function. 

    x - the input
    '''
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    '''
    relu - output value after applying the relU activation function. 

    x - the input
    '''
    def relu(self, x):
        return np.maximum(x)
    
    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)
    
    """
    feed_forward - Performs feed-forward propagation on the neural network
    
    board - The current state of the board
    """
    def feed_forward(self, board, neurons):
        outputs = []
        layer_inputs = board

        for neuron in neurons:
            weighted_sum = neuron.bias
            if isinstance(layer_inputs, np.ndarray):
                weighted_sum += np.dot(layer_inputs, neuron.weight)
            else:
                if neuron.weight:
                    weighted_sum += layer_inputs * neuron.weight

            if neuron.activation_function == 'sigmoid':
                output = self.sigmoid(weighted_sum)
            if neuron.activation_function == 'relu':
                output = self.relu(weighted_sum)

            outputs.append(output)
            layer_inputs = output

        return outputs


    '''
    update_weights - adjusts the weights accordingly to get the best weights

    expected_outputs - what the outputs are supposed to be
    '''
    def update_weights(self, expected_outputs, neurons):
        for i in range(len(neurons) - 1):
            neuron = neurons[i]
            for edge in neuron.outgoing_edges:
                input_neuron = edge.target_neuron
                input_value = input_neuron.value
                error_delta = (expected_outputs[i] - input_value) * self.derivative_activation(input_value)
                if i < len(neuron.weights):
                    # Update weight
                    edge.weight += self.learning_rate * error_delta * input_neuron.weight[i]

                # Update bias
                input_neuron.bias += self.learning_rate * error_delta

        # Update weights and biases of the last (output) neuron
        output_neuron = neurons[-1]
        for edge in output_neuron.outgoing_edges:
            input_neuron = edge.target_neuron
            input_value = input_neuron.value
            error_delta = (expected_outputs[-1] - input_value) * self.derivative_activation(input_value)
            if len(input_neuron.weights) == 1:
                # Update weight
                edge.weight += self.learning_rate * error_delta * input_neuron.weight[0]
            else:
                # Update weight
                edge.weight += self.learning_rate * error_delta * input_neuron.weight[i]

            # Update bias
            input_neuron.bias += self.learning_rate * error_delta
    '''
    train - trains the neural network based on the provided target output
    
    board - The current state of the board
    target_output - The target output for the training
    '''
    def train(self, board, target_output, neurons):
        preprocessed_inputs = self.preprocess_board(board)
        self.feed_forward(preprocessed_inputs, neurons)
        self.update_weights(target_output, neurons)


    def calculate_neuron_value(self, neuron):
        weighted_sum = 0
        for edge in neuron.incoming_edges:
            source_value = edge.source_neuron.value
            weight = edge.weight
            weighted_sum += source_value * weight

        if neuron.activation_function == 'sigmoid':
            neuron.value = self.sigmoid(weighted_sum)
        elif neuron.activation_function == 'relu':
            neuron.value = self.relu(weighted_sum)       
    '''
    classify - the output values after running the network

    board - the game board
    '''
    def classify(self, board, neurons):
        preprocessed_inputs = self.preprocess_board(board)
        outputs = self.feed_forward(preprocessed_inputs, neurons)
        value_neuron = neurons[-1]
        action_neurons = neurons[1:-1]  # Exclude input and value neurons

        value = self.calculate_neuron_value(value_neuron, outputs[-1])

        actions = [self.calculate_neuron_value(neuron, output) for neuron, output in zip(action_neurons, outputs[:-1])]

        # Update the policy table
        state = str(board)
        self.policy_table[state] = {
            'value': value,
            'actions': actions
        }

# TODO: fix up maing accordingly
def main():
    # Create the neurons
    input_neuron = Neuron(activation_function='relu', weight=np.random.uniform(-1, 1))
    neuron_value = Neuron(activation_function='sigmoid', weight=np.random.uniform(-1, 1))
    action_neurons = []
    for i in range(9):
        action_neurons.append(Neuron(activation_function='sigmoid', weight=np.random.uniform(-1, 1)))

    # Create the neural network
    neural_network = NeuralNetwork(learning_rate=0.01)

    # Add neurons to the network and their edges
    neurons = [input_neuron, neuron_value] + action_neurons
    neural_network.add_neuron(input_neuron)
    neural_network.add_neuron(neuron_value)
    for action_neuron in action_neurons:
        neural_network.add_neuron(action_neuron)
        weight = 0
        while weight == 0:
            weight = np.random.uniform(-1, 1)
        neural_network.add_edge(input_neuron, action_neuron, weight, 0, 'sigmoid')
        neural_network.add_edge(action_neuron, neuron_value, weight, 0, 'sigmoid')

    # Train the network on the Hexapawn data
    training_data = [
        {
            'board': [
                0,
                1, 1, 1,
                0, 0, 0,
                -1, -1, -1
            ],
            'target_output': [
                0, 0, 0, 0, 1, 0, 0, 0, 0
            ]
        },
        # Add more training examples here
    ]

    for epoch in range(10):
        for data in training_data:
            board = data['board']
            target_output = data['target_output']
            neural_network.train(board, target_output, neurons)

    # Call the classify function
    neural_network.classify(StartBoard, neurons)

    # Print the policy table of the best network
    policy_table = neural_network.policy_table
    for state, policy in policy_table.items():
        value = policy['value']
        actions = policy['actions']
        print("State:", state)
        print("Value:", value)
        print("Actions:", actions)
        print()


if __name__ == "__main__":
    main()