import pdb
import unittest

import numpy as np

from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from pydca.learning_models.my_models.celm import CELM
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
import sklearn.datasets
#from pycallgraph3 import PyCallGraph
#from pycallgraph3.output import GraphvizOutput


def test_train_mlp_cascade():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.random_sample(x.shape)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
    #cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (100, 50, 100)}))

    cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (1, )}))
    cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (1,)}))
    cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (100,)}))
    target = y_train
    cascade.train_networks(x_train,target, True)
    print("APPLY DATA")
    cascade.apply_data(x_test)
    y_hat = cascade.get_last_layer_output()
    if len(y.shape) > 1:
        shape = (y.shape[0], y.shape[1])
    else:
        shape = (y.shape[0],1)
    y_hat = cascade.predict(x, 1)
    plt.plot(y, c="green")
    plt.plot(y_hat, c="red")
    plt.show()
    print(f"test error {mean_squared_error(y.flatten(), y_hat.flatten(), squared=False)}")

def test_train_celm_cascade():
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.random_sample(x.shape)
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
    cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size":100, "constr": {}}))

    target = y_train
    cascade.train_networks(x_train,target, True)
    print("APPLY DATA")
    cascade.apply_data(x_test)
    y_hat = cascade.get_last_layer_output()
    if len(y.shape) > 1:
        shape = (y.shape[0], y.shape[1])
    else:
        shape = (y.shape[0],1)
    y_hat = cascade.predict(x, 1)
    plt.plot(y, c="green")
    plt.plot(y_hat, c="red")
    plt.show()
    print(f"test error {mean_squared_error(y.flatten(), y_hat.flatten(), squared=False)}")


def test_sample_based_incremental_cascade():
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x) + np.cos(x)**2 + np.sin(x)**3 + (np.random.random_sample(x.shape)-0.5)
    plt.plot(x, y, c="green")
    plt.plot(x, np.sin(x) + np.cos(x)**2 + np.sin(x)**3, c="blue")
    plt.show()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
    #tmp_x_train = x_train
    #tmp_y_train = y_train
    for i in range(3):
        print(f"number of samples{x_train.shape}")
        cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (1, )}))
        cascade.train_networks(x_train,y_train, True)
        y_hat = cascade.predict(x_train, 1)
        print(f"cascade::train error {mean_squared_error(y_train.flatten(), y_hat.flatten(), squared=False)}")
        score = ((y_train - y_hat)**2).flatten()
        mean_error = np.mean(score)
        indexes = np.where(score > mean_error)
        #pdb.set_trace()
        x_train = np.vstack([x_train, x_train[indexes]])
        y_train = np.vstack([y_train, y_train[indexes]])

        y_hat = cascade.predict(x, 1)
        plt.plot(x, y, c="green")
        plt.plot(x, y_hat, c="red")
        plt.plot(x, np.sin(x) + np.cos(x) ** 2 + np.sin(x) ** 3, c="blue")
        plt.show()

    y_hat = cascade.predict(x_test, 1)
    print(f"test error {mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)}")

def test_celm_sample_based_incremental_cascade():
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x) + np.cos(x)**2 + np.sin(x)**3 + np.tanh(x)*np.cos(x) #+ 10*(np.random.random_sample(x.shape)-0.5)
    plt.plot(x, y, c="green")
    #plt.plot(x, np.sin(x) + np.cos(x)**2 + np.sin(x)**3, c="blue")
    plt.show()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)
    cascade = CascadeNetwork("Test_Supervised")

    for i in range(4):
        print(f"number of samples{x_train.shape}")
        cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": 15, "constr": {}}))
        cascade.train_networks(x_train,y_train, True)
        y_hat = cascade.predict(x_train, 1)
        print(f"cascade::train error {mean_squared_error(y_train.flatten(), y_hat.flatten(), squared=False)}")
        score = ((y_train - y_hat)**2).flatten()
        mean_error = np.mean(score)
        indexes = np.where(score > mean_error)
        #pdb.set_trace()
        x_train = np.vstack([x_train, x_train[indexes]])
        y_train = np.vstack([y_train, y_train[indexes]])

        y_hat = cascade.predict(x_train, 1)
        plt.plot(x, y, c="green")
        plt.scatter(x_train, y_hat, c="red")
        plt.scatter(x_train, y_train, c="magenta")
        y_hat = cascade.predict(x, 1)
        plt.plot(x, y_hat, c="cyan")
        #plt.plot(x, np.sin(x) + np.cos(x) ** 2 + np.sin(x) ** 3, c="blue")
        plt.show()

    y_hat = cascade.predict(x_test, 1)
    print(f"test error {mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)}")

def swipe_test_celm_sample_based_incremental_cascade():
    x = np.linspace(0, 2*np.pi, 500)
    y = np.sin(x) + np.cos(x)**2 + np.sin(x)**3 + np.tanh(x)*np.cos(x) #+ 10*(np.random.random_sample(x.shape)-0.5)
    plt.plot(x, y, c="green")
    #plt.plot(x, np.sin(x) + np.cos(x)**2 + np.sin(x)**3, c="blue")
    plt.show()
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)

    cascade = CascadeNetwork("Test_Supervised")

    blocks = x.shape[0] // 4
    last_block = x.shape[0] % 4

    #tmp_x_train = x[0*blocks:blocks+0*blocks,:]
    #tmp_y_train = y[0*blocks:blocks+0*blocks, :]
    errors = []
    errors_celm = []
    print(f"blocks {blocks}")
    hidden_layers = []
    for i in np.linspace(0.05,0.7,4):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i)
        print(f"number of samples{x_train.shape}")
        cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": x_train.shape[0], "constr": {}}))
        celm = CascadeNetwork("celm_test")
        hidden_layers.append( x_train.shape[0])
        celm.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": hidden_layers, "constr": {}}))
        cascade.train_networks(x_train,y_train, True)
        celm.train_networks(x_train, y_train, True)
        y_hat = cascade.predict(x_test, 1)
        y_hat_celm = celm.predict(x_test, 1)
        error = mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)
        error_celm = mean_squared_error(y_test.flatten(), y_hat_celm.flatten(), squared=False)
        print(f"cascade::test error {error}")
        errors.append(error)
        print(f"cascade::test error {error_celm}")
        errors_celm.append(error_celm)
        y_hat = cascade.predict(x_train, 1)
        plt.plot(x, y, c="green")
        plt.scatter(x_train, y_hat, c="red")
        plt.scatter(x_train, y_train, c="magenta")
        y_hat = cascade.predict(x, 1)
        plt.plot(x, y_hat, c="cyan")
        #plt.plot(x, np.sin(x) + np.cos(x) ** 2 + np.sin(x) ** 3, c="blue")
        print(f"HIDDEN SIZE {x_train.shape[0]}")
        plt.show()
    y_hat = cascade.predict(x_test, 1)
    print(f"test error {mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)}")
    plt.plot(np.linspace(0.05,0.7,10),errors)
    plt.plot(np.linspace(0.05, 0.7, 10), errors_celm)
    plt.show()

def d2_test_celm_sample_based_incremental_cascade():
    def gaussian_kernel_2d(x, y, xc, yc, sigma):
        return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x-xc) ** 2 + (y-yc) ** 2) / (2 * sigma ** 2))

    # Define the size, standard deviation (sigma), and step size
    kernel_size = 5
    sigma = 1
    sigma2 = 4.
    step = 0.1

    # Create arrays for x and y coordinates
    x_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)
    y_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)

    # Create the meshgrid of x and y coordinates
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the corresponding z values using the Gaussian kernel
    Z1 = gaussian_kernel_2d(X, Y, 1, -1, sigma)
    Z2 = gaussian_kernel_2d(X, Y, -1, 1, 0.8)
    Z_mixture = Z1 + Z2

    # Optionally, you might want to normalize the mixture
    Z = Z_mixture / np.max(Z_mixture)


    # Flatten the X, Y, and Z arrays to obtain 1D arrays
    x_coordinates = X.flatten()
    y_coordinates = Y.flatten()
    z_values = Z.flatten() + (np.random.random_sample(Z.flatten().shape)-0.5)/100.

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
    plt.show()

    x = np.hstack([x_coordinates.reshape(-1, 1), y_coordinates.reshape(-1, 1)])
    y = np.hstack([z_values.reshape(-1,1),])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=42)
    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
    tmp_x_train = x_train
    tmp_y_train = y_train
    old_test_error = np.finfo(np.float64).max
    errors = []
    for i in range(10):
        print(f"number of samples{tmp_x_train.shape}")
        cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": 50, "constr": {}}))
        cascade.train_networks(tmp_x_train,tmp_y_train, True)
        y_hat = cascade.predict(x_train, 1)
        print(f"cascade::train error {mean_squared_error(y_train.flatten(), y_hat.flatten(), squared=False)}")
        score = ((y_train - y_hat)**2).flatten()
        mean_error = np.mean(score)
        indexes = np.where(score > mean_error)
        #pdb.set_trace()
        tmp_x_train = x_train[indexes]
        tmp_y_train = y_train[indexes]

        y_hat = cascade.predict(x_test, 1)
        test_error = mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)
        errors.append(test_error)
        print(f"test error {test_error}")
        y_hat = cascade.predict(x, 1)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
        ax.scatter(x[:, 0], x[:, 1], y_hat,  c="red")
        plt.show()
        if test_error > old_test_error:
            break
        else:
            old_test_error = test_error

    plt.plot(errors)
    plt.show()


def d2_step_by_step_test_celm_sample_based_incremental_cascade():
    def gaussian_kernel_2d(x, y, xc, yc, sigma):
        return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x-xc) ** 2 + (y-yc) ** 2) / (2 * sigma ** 2))

    # Define the size, standard deviation (sigma), and step size
    kernel_size = 5
    sigma = 1
    sigma2 = 4.
    step = 0.1

    # Create arrays for x and y coordinates
    x_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)
    y_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)

    # Create the meshgrid of x and y coordinates
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the corresponding z values using the Gaussian kernel
    Z1 = gaussian_kernel_2d(X, Y, 1, -1, sigma)
    Z2 = gaussian_kernel_2d(X, Y, -1, 1, 0.8)
    Z_mixture = Z1 + Z2

    # Optionally, you might want to normalize the mixture
    Z = Z_mixture / np.max(Z_mixture)


    # Flatten the X, Y, and Z arrays to obtain 1D arrays
    x_coordinates = X.flatten()
    y_coordinates = Y.flatten()
    z_values = Z1.flatten() + (np.random.random_sample(Z.flatten().shape)-0.5)/100.

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
    plt.show()

    x = np.hstack([x_coordinates.reshape(-1, 1), y_coordinates.reshape(-1, 1)])
    y = np.hstack([z_values.reshape(-1,1),])
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.5, random_state=42)
    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
    tmp_x_train = x_train
    tmp_y_train = y_train
    old_test_error = np.finfo(np.float64).max
    errors = []
    for i in range(10):
        print(f"number of samples{tmp_x_train.shape}")
        cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": 50, "constr": {}}))
        cascade.train_networks(tmp_x_train,tmp_y_train, True)
        y_hat = cascade.predict(x_train, 1)
        print(f"cascade::train error {mean_squared_error(y_train.flatten(), y_hat.flatten(), squared=False)}")
        score = ((y_train - y_hat)**2).flatten()
        mean_error = np.mean(score)
        indexes = np.where(score > mean_error)
        #pdb.set_trace()
        tmp_x_train = x_train[indexes]
        tmp_y_train = y_train[indexes]

        y_hat = cascade.predict(x_test, 1)
        test_error = mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)
        errors.append(test_error)
        print(f"test error {test_error}")
        y_hat = cascade.predict(x, 1)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
        ax.scatter(x[:, 0], x[:, 1], y_hat,  c="red")
        plt.show()
        if test_error > old_test_error:
            break
        else:
            old_test_error = test_error

    plt.plot(errors)
    plt.show()


def swipe_d2_step_by_step_test_celm_sample_based_incremental_cascade():
    def gaussian_kernel_2d(x, y, xc, yc, sigma):
        return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x-xc) ** 2 + (y-yc) ** 2) / (2 * sigma ** 2))

    # Define the size, standard deviation (sigma), and step size
    kernel_size = 5
    sigma = 1
    sigma2 = 4.
    step = 0.1

    # Create arrays for x and y coordinates
    x_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)
    y_range = np.arange(-kernel_size // 2, kernel_size // 2 + step, step)

    # Create the meshgrid of x and y coordinates
    X, Y = np.meshgrid(x_range, y_range)

    # Calculate the corresponding z values using the Gaussian kernel
    Z1 = gaussian_kernel_2d(X, Y, 1, -1, sigma)
    Z2 = gaussian_kernel_2d(X, Y, -2, -2, 2)
    Z_mixture = Z1 + Z2

    # Optionally, you might want to normalize the mixture
    Z = Z_mixture / np.max(Z_mixture)


    # Flatten the X, Y, and Z arrays to obtain 1D arrays
    x_coordinates = X.flatten()
    y_coordinates = Y.flatten()
    z_values = Z.flatten() + (np.random.random_sample(Z.flatten().shape)-0.5)/100.

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
    plt.show()

    x = np.hstack([x_coordinates.reshape(-1, 1), y_coordinates.reshape(-1, 1)])
    y = np.hstack([z_values.reshape(-1,1),])

    cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")

    old_test_error = np.finfo(np.float64).max
    errors = []
    for i in np.linspace(0.1,0.7,5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i, random_state=42)
        print(f"number of samples{x_train.shape}")
        cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": x_train.shape[0]//10, "constr": {}}))
        cascade.train_networks(x_train,y_train, True)
        y_hat = cascade.predict(x_train, 1)
        print(f"cascade::train error {mean_squared_error(y_train.flatten(), y_hat.flatten(), squared=False)}")


        y_hat = cascade.predict(x_test, 1)
        test_error = mean_squared_error(y_test.flatten(), y_hat.flatten(), squared=False)
        errors.append(test_error)
        print(f"test error {test_error}")
        y_hat = cascade.predict(x, 1)
        print(f"HIDDEN LAYER SIZE {x_train.shape[0]//10}")
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x_coordinates, y_coordinates, z_values, c="green")
        ax.scatter(x[:, 0], x[:, 1], y_hat,  c="red")
        plt.show()
        if test_error > old_test_error:
            break
        else:
            old_test_error = test_error

    plt.plot(errors)
    plt.show()

if __name__ == '__main__':
    #with PyCallGraph(output=GraphvizOutput()):
    swipe_d2_step_by_step_test_celm_sample_based_incremental_cascade()
