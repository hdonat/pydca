import unittest

import numpy as np

from pydca.pydca.cascade.cascades import CascadeNetwork
from pydca.pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from pydca.pydca.learning_models.keras.autoencoders import AutoEncoder
from pydca.pydca.learning_models.old.SimpleELM import RRELM
from pydca.pydca.cascade.filters import ZScoreFilter
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
import sklearn.datasets

x, y = make_regression(500,100)
x, x_test, y, y_test = train_test_split(x, y, test_size=0.50)

class TestComapreTrainingAE(unittest.TestCase):

    def test_train_single_mlp_cascade(self):
        global x, x_test, y, y_test
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        cascade.add_network_layer(MLPModel(lambda tar,pred: np.sqrt(mean_squared_error(tar,pred)), {"hidden_layer_dim": (x.shape[1]*2,x.shape[1]*2,x.shape[1]*2)}), first_layer_filter=ZScoreFilter())
        target = y
        cascade.train_networks(x,target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0],1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:,-shape[1]:])
        plt.figure()
        plt.plot(y_test.T[-10:], c="green")
        plt.plot(y_hat[-10:], c="orange")
        plt.savefig("test_results/single_mlp.png")
        print(np.sqrt(mean_squared_error(y_test, y_hat)))


    def test_train_mlp_cascade(self):
        global x, x_test, y, y_test
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        loss_foo = lambda tar,pred: np.sqrt(mean_squared_error(tar,pred))
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": x.shape[1]*2}), first_layer_filter=ZScoreFilter(), filter=ZScoreFilter())
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": x.shape[1]*2}), filter=ZScoreFilter())
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": x.shape[1]*2}))


        target = y
        cascade.train_networks(x,target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0],1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:,-shape[1]:])
        plt.figure()
        plt.plot(y_test.T[-10:], c="green")
        plt.plot(y_hat[-10:], c="blue")
        plt.savefig("test_results/train_two_single_mlp.png")
        print(np.sqrt(mean_squared_error(y_test, y_hat)))

    def test_train_elm_cascade(self):
        global x, x_test, y, y_test
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)
        loss_foo = lambda tar,pred: np.sqrt(mean_squared_error(tar,pred))
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": x.shape[1]*2, "bias": 0, "alpha": 0.0001, "bip": False}), first_layer_filter=ZScoreFilter(), filter=ZScoreFilter())
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": x.shape[1]*2, "bias": 0, "alpha": 0.0001, "bip": False}), filter=ZScoreFilter())
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": x.shape[1]*2, "bias": 0, "alpha": 0.0001, "bip": False}))


        target = y
        cascade.train_networks(x,target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0],1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:,-shape[1]:])
        plt.figure()
        plt.plot(y_test.T[-10:], c="green")
        plt.plot(y_hat[-10:], c="blue")
        plt.savefig("test_results/train_two_single_elm.png")
        print(np.sqrt(mean_squared_error(y_test, y_hat)))

    def test_train_elm_and_autoencoder_cascade(self):
        global x, x_test, y, y_test
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_AE_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)

        loss_foo = lambda tar, pred: np.sqrt(mean_squared_error(tar, pred))
        cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 10}), connect_to_inlayer=0,
                                  first_layer_filter=ZScoreFilter(), filter=ZScoreFilter(),
                                  filter_layer_connections=[0])
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": 20, "bias": 0, "alpha": 0.0001, "bip": False}), filter=ZScoreFilter(),
                                  filter_layer_connections=[0, 1])
        cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 25}), connect_to_inlayer=0,
                                  filter=ZScoreFilter(), filter_layer_connections=[0, 1])
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": 40, "bias": 0, "alpha": 0.0001, "bip": False}), filter=ZScoreFilter(),
                                  filter_layer_connections=[1, 3])
        #cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 50}), connect_to_inlayer=0,
        #                          filter=ZScoreFilter(), filter_layer_connections=[0, 1, 3])
        cascade.add_network_layer(RRELM(loss_foo, {"hidden_size": x.shape[1]*2, "bias": 0, "alpha": 0.0001, "bip": False}), filter_layer_connections=[])

        target = y
        cascade.train_networks(x, target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0], 1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:, -shape[1]:])
        plt.figure()
        plt.plot(y_test.T[-10:], c="green")
        plt.plot(y_hat[-10:], c="red")
        plt.savefig("test_results/test_train_elm_and_autoencoder_cascade.png")
        cascade.draw_graph("test_results")
        print(np.sqrt(mean_squared_error(y_test, y_hat)))


    def test_train_mlp_and_autoencoder_cascade(self):
        global x, x_test, y, y_test
        cascade = CascadeNetwork("Test_Supervised_Unsupervised_AE_Cascade")
        self.assertEqual(isinstance(cascade, CascadeNetwork), True)

        loss_foo = lambda tar,pred: np.sqrt(mean_squared_error(tar,pred))
        cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 10}), connect_to_inlayer=0,  first_layer_filter=ZScoreFilter(), filter=ZScoreFilter(),filter_layer_connections=[0])
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": 20}), filter=ZScoreFilter(),filter_layer_connections=[0,1])
        cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 25}), connect_to_inlayer=0, filter= ZScoreFilter(),filter_layer_connections=[0,1])
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": 50}), filter=ZScoreFilter(),filter_layer_connections=[1,3])
        #cascade.add_network_layer(AutoEncoder(loss_foo, {"encoding_dim": 50}), connect_to_inlayer=0, filter= ZScoreFilter(),filter_layer_connections=[0,1,3])
        cascade.add_network_layer(MLPModel(loss_foo, {"hidden_layer_dim": x.shape[1]}),filter_layer_connections=[])


        target = y
        cascade.train_networks(x,target, True)
        cascade.apply_data(x_test)
        y_hat = cascade.get_last_layer_output()
        try:
            shape = (y.shape[0], y.shape[1])
        except IndexError:
            shape = (y.shape[0],1)
        y_hat = cascade.layers[-1]._filter.out_filter(y_hat[:,-shape[1]:])
        plt.figure()
        plt.plot(y_test.T[-10:], c="green")
        plt.plot(y_hat[-10:], c="red")
        plt.savefig("test_results/test_train_mlp_and_autoencoder_cascade.png")
        cascade.draw_graph("test_results")
        print(np.sqrt(mean_squared_error(y_test, y_hat)))


if __name__ == '__main__':
    unittest.main()
