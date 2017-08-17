import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import train_loader, test_loader
from model import cnn

from hyperspace import HyperSpace
from skopt.callbacks import DeadlineStopper
from skopt import gp_minimize
from skopt import dump
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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
    kernel_size1, padding1, pool_kernel1, kernel_size2, padding2, pool_kernel2, learning_rate = space

    # Hyper Parameters
    num_epochs = 2

    cnn = CNN(kernel_size1, padding1, pool_kernel1, kernel_size2, padding2, pool_kernel2)
    cnn.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.view(-1, sequence_length, input_size)).cuda()
            labels = Variable(labels).cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    test_accuracy = 100 * correct / total
    return test_accuracy


def main():
    if rank == 0:
        hyperparameters = {'kernelSize1': np.arange(2,10),
                           'padding1': np.arange(1, 10),
                           'poolKernel1': np.arange(2, 100),
                           'kernelSize2': np.arange(2,10),
                           'padding2': np.arange(1, 10),
                           'poolKernel2': np.arange(2, 100),
                           'learningRate': np.linspace(0.001, 0.1)}

        hyperspace = HyperSpace(hyperparameters)
        all_intervals = hyperspace.fold_space()
        hyperspaces = hyperspace.hyper_permute(all_intervals)
        subspace_keys, subspace_boundaries = hyperspace.format_hyperspace(hyperspaces)
    else:
        subspace_keys, subspace_boundaries = None, None

    space = comm.scatter(subspace_boundaries, root=0)
    keys = comm.scatter(subspace_keys, root=0)

    # Gaussian process (see scikit-optimize skopt module for other optimizers)
    res_gp = gp_minimize(objective, space, n_calls=20, random_state=0, verbose=True)
    gathered_evaluations = comm.gather(res_gp, root=0)

    for i in range(len(gathered_evaluations)):
        dump(gathered_evaluations[i], 'hyper_results/gp_subspace_' + str(i))


if __name__ == '__main__':
    main()
