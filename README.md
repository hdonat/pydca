# pydca

This package contains classes and functions for implementing direct cascaded machine learning models described by Littmann and Ritter.

## Installation

### Using Rye

To install the package using Rye, ensure you have Rye installed. Then, run the following command:

```sh
rye sync
```

### Using Pip

To install the package using Pip, run the following command:

```sh
pip install .
```

## Dependencies

The package requires the following dependencies:

- numpy>=2.2.1
- pandas>=2.2.3
- torch>=2.5.1
- scikit-learn>=1.6.0

## Usage

To use the package, import the necessary modules and classes in your Python script. For example:

```python
from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.sklearn.simple_regression_models import LinearModel
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from pydca.learning_models.my_models.celm import CELM
```

You can then create and train cascade networks using the provided classes and functions.

### Stacking Layers

To stack layers in the `CascadeNetwork`, you can use the `add_cascade_layer` method to add multiple layers sequentially. Here is an example of how to stack layers in the `CascadeNetwork`:

```python
from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel
from sklearn.metrics import root_mean_squared_error

# Create a CascadeNetwork instance
cascade = CascadeNetwork("Stacked_Layers_Cascade")

# Add multiple layers to the cascade
cascade.add_cascade_layer(MLPModel(root_mean_squared_error, {"hidden_layer_dim": (100,)}))
cascade.add_cascade_layer(MLPModel(root_mean_squared_error, {"hidden_layer_dim": (50,)}))
cascade.add_cascade_layer(MLPModel(root_mean_squared_error, {"hidden_layer_dim": (25,)}))

# Train the cascade with training data
x_train, y_train = ...  # Your training data here
cascade.train_networks(x_train, y_train, True)

# Apply the cascade to test data
x_test = ...  # Your test data here
cascade.apply_data(x_test)
y_hat = cascade.get_last_layer_output()
```

In this example, three layers with different hidden layer dimensions are added to the `CascadeNetwork`. You can adjust the parameters and number of layers as needed for your specific use case.

## Author

Heiko Donat (info@hdonat.net)

## License

This project is licensed under the MIT License.
