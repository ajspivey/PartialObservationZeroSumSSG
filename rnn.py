import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]



# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize neural network weights
inputToHiddenWeights = 2*np.random.random((input_dim,hidden_dim)) - 1
hiddenToOutputWeights = 2*np.random.random((hidden_dim,output_dim)) - 1
hiddenToHiddenWeights = 2*np.random.random((hidden_dim,hidden_dim)) - 1

inputToHiddenWeights_update = np.zeros_like(inputToHiddenWeights)
hiddenToOutputWeights_update = np.zeros_like(hiddenToOutputWeights)
hiddenToHiddenWeights_update = np.zeros_like(hiddenToHiddenWeights)

# training logic
for j in range(100000):
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]

    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0

    output_layer_deltas = list()
    hidden_layer_values = list()
    hidden_layer_values.append(np.zeros(hidden_dim))

    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        previousLayer = hidden_layer_values[-1]
        hidden_layer = sigmoid(np.dot(X,inputToHiddenWeights) + np.dot(previousLayer,hiddenToHiddenWeights))

        # output layer (new binary representation)
        output_layer = sigmoid(np.dot(hidden_layer,hiddenToOutputWeights))

        # did we miss?... if so, by how much?
        output_layer_error = y - output_layer
        output_layer_deltas.append((output_layer_error)*sigmoid_output_to_derivative(output_layer))
        overallError += np.abs(output_layer_error[0])

        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(output_layer[0][0])

        # store hidden layer so we can use it in the next timestep
        hidden_layer_values.append(copy.deepcopy(hidden_layer))

    future_hidden_layer_delta = np.zeros(hidden_dim)

    for position in range(binary_dim):
        X = np.array([[a[position],b[position]]])
        hidden_layer = hidden_layer_values[-position-1]
        prev_hidden_layer = hidden_layer_values[-position-2]

        # error at output layer
        output_layer_delta = output_layer_deltas[-position-1]
        # error at hidden layer
        hidden_layer_delta = (future_hidden_layer_delta.dot(hiddenToHiddenWeights.T) + output_layer_delta.dot(hiddenToOutputWeights.T)) * sigmoid_output_to_derivative(hidden_layer)

        # let's update all our weights so we can try again
        hiddenToOutputWeights_update += np.atleast_2d(hidden_layer).T.dot(output_layer_delta)
        hiddenToHiddenWeights_update += np.atleast_2d(prev_hidden_layer).T.dot(hidden_layer_delta)
        inputToHiddenWeights_update += X.T.dot(hidden_layer_delta)

        future_hidden_layer_delta = hidden_layer_delta


    inputToHiddenWeights += inputToHiddenWeights_update * alpha
    hiddenToOutputWeights += hiddenToOutputWeights_update * alpha
    hiddenToHiddenWeights += hiddenToHiddenWeights_update * alpha

    inputToHiddenWeights_update *= 0
    hiddenToOutputWeights_update *= 0
    hiddenToHiddenWeights_update *= 0

    # print out progress
    if(j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
