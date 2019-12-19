import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# Network definition
class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNetwork, self).__init__()
        # Initialize pytorch components used (LSTM, Linear, and Sigmoid)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    # Define a forward pass of the network
    # TODO: find out when tensors need to be reshaped
    def forward(self, twoNums):
        # Get the output of the lstm layer
        lstmOut, _ = self.lstm(twoNums)
        # Get the dimensions of the output
        intASize, intBSize, featureDimensionSize = lstmOut.size(0), lstmOut.size(1), lstmOut.size(2)
        # We need to reshape the data before feeding to the linear layer
        lstmOut = lstmOut.view(intBSize*intASize,featureDimensionSize)
        # Get the output of the linear layer
        outputLayerActivations = self.linear(lstmOut)
        # Reshape before feeding to the sigmoid (activation) layer
        print("LINEAR OUTPUT")
        print(outputLayerActivations)
        print("LINEAR OUTPUT RESHAPED")
        outputLayerActivations = outputLayerActivations.view(intASize,intBSize,-1).squeeze(1)
        print(outputLayerActivations)
        # Return the sigmoid results
        outputSigmoid = self.sigmoid(outputLayerActivations)
        print("SIGMOID")
        print(outputSigmoid)
        return outputSigmoid

def getBinaryDict(length):
    """ Returns a dict that maps an integer to its binary representation of a
    certain length """
    dict = {}
    largest_number = pow(2, length)
    binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        dict[i] = binary[i]
    return dict

def getSample(length, binaryDict, printOutput=False):
    """ Generates a binary addition problem of a certain length, and returns the
    input/label representation -- a tuple containing a list of bits for each number,
    and the list of bits for the solution to the problem """
    largest_number = pow(2,length)
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = binaryDict[a_int] # binary encoding
    b_int = np.random.randint(largest_number/2) # int version
    b = binaryDict[b_int] # binary encoding
    # true answer
    c_int = a_int + b_int
    c = binaryDict[c_int]
    lenC = len(c)

    # Package the input as a tuple of two lists of bits:
    # EX: x = ([0,1,1,0,1],[0,0,0,1,0]) for a length of 5
    # NOTE: bits are in order from least-significant to most-significant -- the
    # order they will be added in.
    x = np.zeros((lenC,2), dtype=np.float32)
    for i in range(0, lenC):
        x[i,0] = a[lenC-1-i]
        x[i,1] = b[lenC-1-i]
    # package the label as a list of bits
    y = np.array([int(_) for _ in c[::-1]])

    return x, y




def main():
    # network variables
    alpha = 0.1     # The learning rate
    input_dim = 2   # two bits -- one from each binary number being added
    hidden_dim = 16 # inputDim * length (we're gonna train on length 8, so 2 * 8 = 16) Why? Not sure yet.
    output_dim = 1  # one bit -- A one or a zero

    # Create the model
    model = LSTMNetwork(input_dim, hidden_dim, output_dim)
    lossFunction = nn.MSELoss() # Mean-squared error loss function
    optimizer = optim.SGD(model.parameters(), lr=alpha) # Adam optimizer

    # Train the model
    length = 8
    binaryDict = getBinaryDict(length)
    epochs=500
    print("model initialized")
    totalLoss = float("inf")
    # Train until an arbitrary accuracy
    while totalLoss > 1e-5:
        print(f"Avg loss for last 500 samples = {totalLoss}")
        totalLoss = 0

        for _ in range(0,epochs):
            x,y = getSample(length, binaryDict)

            # Unsqueeze is a dimensionality thing -- need to find out why it's necessary
            # Only floats can use requires_grad_, and we want that for our back pass
            xVar = torch.from_numpy(x).unsqueeze(1).float().requires_grad_(True)
            yVar = torch.from_numpy(y).unsqueeze(1).float().requires_grad_(True)

            guesses = model(xVar) # Perform a forward pass (calls the forward function of the model)
            loss = lossFunction(guesses,yVar) # calculate loss using MSE, the guesses, and the labels
            totalLoss += loss.item() # Calculate cumulative loss over epoch

            # optimizer gradients need to be cleared out from the last step,
            # otherwise all backward pass gradients will be accumulated
            optimizer.zero_grad()
            loss.backward() # compute the gradient of the loss with respect to the parameters of the model
            optimizer.step() # Perform a step of the optimizer based on the gradient just calculated

        totalLoss = totalLoss/epochs


    # Test the model with a different length of binary numbers
    # and print out the results
    print("Testing Model")
    newLength = 5
    printOutput = True
    newBinaryDict = getBinaryDict(newLength)
    # test the network on 10 random binary number addition cases
    for i in range (0,10):
        x,y=getSample(newLength, newBinaryDict, printOutput)
        x_var = torch.from_numpy(x).unsqueeze(1).float().requires_grad_(True)
        y_var = torch.from_numpy(y).unsqueeze(1).float().requires_grad_(True)
        #x_var= x_var.contiguous()
        finalScores = model(x_var).data.t()
        # Round up guesses
        x_bits = finalScores.gt(0.5)
        x_bits = x_bits[0].detach().numpy()
        x_bits = [int(_) for _ in x_bits]
        y_bits = y_var.detach().numpy()
        y_bits = [int(_) for _ in y_bits]

        print(f'sum predicted by RNN is {x_bits[::-1]}')
        print(f"actual sum is:          {y_bits[::-1]}")
        print('##################################################')
        print()

if __name__ == "__main__":
    main()
