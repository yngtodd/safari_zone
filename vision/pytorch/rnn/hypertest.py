import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import train_loader, test_loader
from model import RNN

import numpy as np
from hyperspace.space_division import HyperSpace
from skopt.callbacks import DeadlineStopper
from skopt import gp_minimize
from skopt import dump

def objective(space):
    """
    Objective function to be minimized by hyperspace.

    Parameters
    ----------
    * `space` [list]:
      List of Scikit-Optimize Space instances.

      - For more Documentation:
        https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/space/space.py#L462

    Returns
    -------
    * `test_accuracy` [float]:
      Simple accuracy on 10000 test images.
    """
    hidden_size, num_layers, learning_rate = space

    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    learning_rate = float(learning_rate)
    input_size = hidden_size
    sequence_length = hidden_size

    #hidden_size = torch.from_numpy(hidden_size)
    #num_layers = torch.from_numpy(num_layers)
    #learning_rate = torch.from_numpy(learning_rate)

    print(hidden_size)
    print(type(hidden_size))
    print(input_size)

    # Hyper Parameters
    num_epochs = 2

    rnn = RNN(num_layers)
    #rnn.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = rnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, 60000//100, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        outputs = rnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    test_accuracy = 100 * correct / total
    return test_accuracy


def main():
    hyperparameters = {'hiddenSize': np.arange(25, 100),
                       'numLayers': np.arange(1, 6),
                       'learningRate': np.linspace(0.001, 0.1)}

    hyperspace = HyperSpace(hyperparameters)
    all_intervals = hyperspace.fold_space()
    hyperspaces = hyperspace.hyper_permute(all_intervals)
    subspace_keys, subspace_boundaries = hyperspace.format_hyperspace(hyperspaces)


    space = subspace_boundaries[0]
    # Gaussian process (see scikit-optimize skopt module for other optimizers)
    res_gp = gp_minimize(objective, space, n_calls=20, random_state=0, verbose=True)
    gathered_evaluations = comm.gather(res_gp, root=0)


if __name__ == '__main__':
    main()
