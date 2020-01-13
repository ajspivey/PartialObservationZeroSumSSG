# ==============================================================================
# IMPORTS
# ==============================================================================
# External imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

# Internal Imports
import ssg

# Set the random seed
torch.manual_seed(1)
np.random.seed(1)

# ==============================================================================
# CLASSES
# ==============================================================================
class AttackerOracle(nn.Module):
    def __init__(self, targetNum, featureCount):
        super(AttackerOracle, self).__init__()

        # Initizlize class variables
        # Example attacker observation
        #o_a = [0,0,1, 0,0,1, 0,0,1, 1.9,4.1,3.4]
        #      [action, past attacks, past attack status, payoffs]
        self.observation_dim = targetNum * featureCount
        self.featureCount = featureCount

        # Initialize pytorch components used (LSTM, Linear, softmax, and concatReLU)
        # LINEAR LAYER
        self.linearLayerLinear = nn.Linear(self.observation_dim, featureCount)
        self.linearLayerReLU = nn.ReLU()

        # PI LAYER
        self.piLayerLSTM = nn.LSTM(2*featureCount, 2*targetNum*featureCount)
        self.piLayerLinear = nn.Linear(2*targetNum*featureCount, targetNum)
        self.piLayerSoftmax = nn.Sigmoid()

    # Define a forward pass of the network
    # TODO: find out when tensors need to be reshaped
    def forward(self, observation):
        # LINEAR LAYER OUTPUT (ll)
        llLinearOut = self.linearLayerLinear(observation)
        # Simulates a CReLU
        llReLUOutPast, llReLUOutNew = self.linearLayerReLU(llLinearOut)
        llCReLUOutPast = torch.cat((llReLUOutPast, -llReLUOutPast),0)
        llCReLUOutNew = torch.cat((llReLUOutNew, -llReLUOutNew),0)
        llCReLUOut = torch.cat((llCReLUOutPast,-llCReLUOutNew),0).view(2,8).unsqueeze(1)
        # LSTM LAYER OUTPUT
        piLayerLSTMOut, _ = self.piLayerLSTM(llCReLUOut)
        sequenceSize, batchSize, numberOfOutputFeatures = piLayerLSTMOut.size(0), piLayerLSTMOut.size(1), piLayerLSTMOut.size(2)
        piLayerLSTMOut = piLayerLSTMOut.view(sequenceSize*batchSize, numberOfOutputFeatures)
        piLayerLinearOut = self.piLayerLinear(piLayerLSTMOut)
        piLayersoftMaxOut = self.piLayerSoftmax(piLayerLinearOut)
        return piLayersoftMaxOut

# ==============================================================================
# FUNCTIONS
# ==============================================================================
def generateRewards(numTargets, lowBound=1, highBound = 10):
    return np.random.uniform(low=lowBound, high=highBound, size=numTargets)

def getMixedDefenderPolicy(game, payoffs):
    """ Generates a random defender policy for testing """
    # [action, pastattacks, pastattackstatus, payoffs]
    defenderPolicy = {
    # Null round
    tuple(np.concatenate(([0,0,0, 0,0,0, 0,0,0], payoffs))): [1,0,0],

    # First round
    tuple(np.concatenate(([1,0,0, 1,0,0, 0,0,0], payoffs))): [0,0,0],   #1
    tuple(np.concatenate(([1,0,0, 0,1,0, 0,1,0], payoffs))): [1,0,0],   #2
    tuple(np.concatenate(([1,0,0, 0,0,1, 0,0,1], payoffs))): [1,0,0],   #3

    # Second round
    tuple(np.concatenate(([0,0,0, 1,2,0, 0,1,0], payoffs))): [0,0,0],   #1
    tuple(np.concatenate(([0,0,0, 1,0,2, 0,0,1], payoffs))): [0,0,0],

    tuple(np.concatenate(([1,0,0, 2,1,0, 0,1,0], payoffs))): [0,0,0],   #2
    tuple(np.concatenate(([1,0,0, 0,1,2, 0,1,1], payoffs))): [1,0,0],

    tuple(np.concatenate(([1,0,0, 2,0,1, 0,0,1], payoffs))): [0,0,0],   #3
    tuple(np.concatenate(([1,0,0, 0,2,1, 0,1,1], payoffs))): [1,0,0],
    }
    return defenderPolicy


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    # ==============
    # Create Network
    # ==============
    model = AttackerOracle(3,4) # Create an attacker oracle to train on a game with 3 targets and 4 features
    lossFunction = nn.SmoothL1Loss() # Mean-squared error loss function
    optimizer = optim.RMSprop(model.parameters()) # Adam optimizer
    print("Model initialized")
    #      [action, past attacks, past attack status, payoffs]
    previousVector = torch.from_numpy(np.array([1,0,0, 0,0,1, 0,0,1, 1.9,4.1,3.4])).float().requires_grad_(True)
    featureVector = torch.from_numpy(np.array([0,1,0, 0,2,1, 0,0,1, 1.9,4.1,3.4])).float().requires_grad_(True)
    pastAndNew = torch.cat((previousVector.unsqueeze(0),featureVector.unsqueeze(0)))
    guessBeforeTraining = model(pastAndNew)

    # =============
    # Train network
    # =============
    epochs = 10
    totalLoss = float("inf")

    # Define game type
    targets = 3
    resources = 1
    timesteps = 2

    rewards = generateRewards(targets)
    game = ssg.SequentialZeroSumSSG(targets, resources, rewards, timesteps)
    # Generate the mixed defender policy (randomly generated for testing)
    mixedDefenderPolicy = getMixedDefenderPolicy(game, rewards)

    print("Training framework initialized: training...")
    while totalLoss > 1e-8:
        # Create a new game to train on
        print(f"Avg loss for last {epochs} samples = {totalLoss}")
        totalLoss = 0
        for _ in range(0,epochs):
            aAction = [0]*targets
            dAction = [0]*targets

            # Play a full game
            for timestep in range(game.timesteps):
                # Get observations
                dObservation, aObservation = game.getPlayerObservations(dAction, aAction)

                # Create model input
                dAction = mixedDefenderPolicy[tuple(dObservation)]  # Defender action
                oldObservation = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
                newObservation = torch.from_numpy(aObservation).float().requires_grad_(True)
                modelInput = torch.cat((oldObservation.unsqueeze(0),newObservation.unsqueeze(0)))

                # Get the guess and label
                x = model(modelInput)   # Attacker action guess
                y, yScore = game.getBestActionAndScore(game.ATTACKER, dAction, rewards) # Attacker action true label
                yVarBit = np.concatenate((game.previousAttackerAction,y))
                xVar = x.view(2,3).squeeze(1).float().requires_grad_(True)
                yVar = torch.from_numpy(yVarBit).view(2,3).float().requires_grad_(True)


                # Calculate loss
                loss = lossFunction(xVar, yVar) # calculate loss using MSE, the guesses, and the labels
                totalLoss += loss.item() # Calculate cumulative loss over epoch

                # optimizer gradients need to be cleared out from the last step,
                # otherwise all backward pass gradients will be accumulated
                optimizer.zero_grad()
                loss.backward() # compute the gradient of the loss with respect to the parameters of the model
                optimizer.step() # Perform a step of the optimizer based on the gradient just calculated

                game.performActions(dAction, y)

            # Reset the game for another round of learning (generate new game?)
            rewards = generateRewards(targets)
            game = ssg.SequentialZeroSumSSG(targets, resources, rewards, timesteps)
            # Generate the mixed defender policy (randomly generated for testing)
            mixedDefenderPolicy = getMixedDefenderPolicy(game, rewards)

        totalLoss = totalLoss/epochs
    print("Done with Training")

    # ================
    # Guess some stuff
    # ================
    print("Testing Model")
    # test the network on 3 different games
    targets = 3
    resources = 1
    timesteps = 2

    for i in range (0,15):
        print(f"Game {i}")
        rewards = generateRewards(targets)
        game = ssg.SequentialZeroSumSSG(targets, resources, rewards, timesteps)
        print(f"Rewards: {rewards}")
        # Generate the mixed defender policy (randomly generated for testing)
        mixedDefenderPolicy = getMixedDefenderPolicy(game, rewards)

        aAction = [0]*targets
        dAction = [0]*targets

        # Play a full game
        for timestep in range(game.timesteps):
            # Get observations
            dObservation, aObservation = game.getPlayerObservations(dAction, aAction)

            # Create model input
            dAction = mixedDefenderPolicy[tuple(dObservation)]  # Defender action
            oldObservation = torch.from_numpy(game.previousAttackerObservation).float().requires_grad_(True)
            newObservation = torch.from_numpy(aObservation).float().requires_grad_(True)
            modelInput = torch.cat((oldObservation.unsqueeze(0),newObservation.unsqueeze(0)))

            # Get the guess and label
            x = model(modelInput).view(2,3)[1]   # Attacker action guess
            print(f"xValues: {x}")
            x = x.gt(0.5).int().detach().numpy()
            y, yScore = game.getBestActionAndScore(game.ATTACKER, dAction, rewards) # Attacker action true label

            print(f"x guess: {x}")
            print(f"y:       {y}")

            game.performActions(dAction, x)

        print()
        print()


if __name__ == "__main__":
    main()
