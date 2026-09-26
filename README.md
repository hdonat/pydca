# pydca

pydca is a Python implementation of the direct cascade architecture (DCA) described by
Littmann and Ritter [1]. A cascade is built from a sequence of learning models ("units").
The units are trained one after another. Each new unit receives the original input data
together with the outputs of all previous units, so every unit can correct the errors of
the units before it.

Because DCA does not need error backpropagation through the whole cascade, any regression
model can be used as a unit: for example a scikit-learn MLP, a linear regression or a
constrained extreme learning machine (CELM).

## Contents

- [How the cascade works](#how-the-cascade-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Available learning models](#available-learning-models)
- [CascadeNetwork reference](#cascadenetwork-reference)
- [Writing your own unit](#writing-your-own-unit)
- [Project structure](#project-structure)
- [Running the tests](#running-the-tests)
- [Documentation](#documentation)
- [References](#references)

## How the cascade works

A `CascadeNetwork` contains layers and units:

- **Layer 0** holds the input data `X`.
- Every call of `add_cascade_layer(unit)` connects the new unit to the last layer and
  creates a new layer for the output of the unit.
- Each new layer collects the data of **all** previous layers. The input of unit `k` is
  therefore `[X, y_0, ..., y_(k-1)]`, where `y_i` is the output of unit `i`.
- A **target layer** holds the training targets `Y`. All supervised units read their
  targets from it.

```
X ──► unit 0 ──► y_0
│                │
▼                ▼
[X, y_0] ──► unit 1 ──► y_1
│                        │
▼                        ▼
[X, y_0, y_1] ──► unit 2 ──► y_2

last layer = [X, y_0, y_1, y_2]
```

`predict(X, output_dim)` returns the last `output_dim` columns of the last layer, which
is the output of the last unit.

## Requirements

- Python 3.10 or newer. The tests pass with Python 3.13 and 3.14.
- The [Graphviz](https://graphviz.org/download/) system package, only if you want to
  draw the cascade with `draw_graph()`. The Python package `graphviz` needs the `dot`
  program.

The Python dependencies are declared in `pyproject.toml`: numpy, scikit-learn,
graphviz, scikit-optimize and qpsolvers (with the OSQP solver).

## Installation

### Using uv

To install the package using [uv](https://docs.astral.sh/uv/), ensure you have uv
installed. Then, run the following command in the project directory:

```sh
uv sync
```

This creates the virtual environment `.venv` and installs the package with all
dependencies, including the development tools (pytest, matplotlib, Zensical,
mkdocstrings).

### Using Pip

To install the package using Pip, run the following command:

```sh
pip install .
```

## Quick start

The following example trains a cascade of two CELM units and one linear regression on a
noisy sine curve:

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pydca.cascade.cascades import CascadeNetwork
from pydca.learning_models.my_models.celm import CELM
from pydca.learning_models.sklearn.simple_regression_models import LinearModel

# Noisy sine data, shape (n_samples, n_features)
x = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)
y = np.sin(x) + 0.1 * np.random.randn(*x.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

cascade = CascadeNetwork("sine_cascade")
cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": 50, "constr": {}}, name="celm_1"))
cascade.add_cascade_layer(CELM(mean_squared_error, {"hidden_size": 50, "constr": {}}, name="celm_2"))
cascade.add_cascade_layer(LinearModel(mean_squared_error, name="linear"))

cascade.train_networks(x_train, y_train, show_error=True)

# Output of the last unit (1 output dimension)
y_hat = cascade.predict(x_test, output_dim=1)
print("test MSE:", mean_squared_error(y_test, y_hat))

# Shape of the last layer: input + outputs of all 3 units -> (50, 4)
print(cascade.get_last_layer_output().shape)

# Draw the cascade structure (needs Graphviz): writes docs/sine_cascade.pdf
cascade.draw_graph("docs")
```

Notes:

- Input and target data must be 2D numpy arrays with the shape
  `(n_samples, n_features)`.
- The first argument of every unit is the loss function `loss(target, prediction)`. It
  is used for the training error that `show_error=True` prints.
- Each unit in a cascade needs a unique name. If you do not give a name, the unit uses
  its object id.

### Incremental training

You can add units after training and call `train_networks()` again. Units that are
already initialized are trained again on the new data, and new units are initialized
first. The test `test_incremental_cascade` in `tests/test_cascade_with_mlp_and_celm.py`
uses this to add one unit after the other and to repeat the samples with a high error.

```python
from pydca.learning_models.sklearn.multi_layer_perceptrons import MLPModel

for i in range(3):
    cascade.add_cascade_layer(MLPModel(mean_squared_error, {"hidden_layer_dim": (1,)}))
    cascade.train_networks(x_train, y_train)
```

## Available learning models

| Class | Module | Model | Parameters |
|-------|--------|-------|------------|
| `MLPModel` | `pydca.learning_models.sklearn.multi_layer_perceptrons` | scikit-learn `MLPRegressor` | `{"hidden_layer_dim": ...}` |
| `LinearModel` | `pydca.learning_models.sklearn.simple_regression_models` | scikit-learn `LinearRegression` | none |
| `CELM` | `pydca.learning_models.my_models.celm` | Constrained extreme learning machine | `{"hidden_size": int, "constr": dict}` |

### MLPModel

When an `MLPModel` is trained for the first time, it searches its hyperparameters with
Bayesian optimization (`skopt.gp_minimize`, 50 evaluations):

- `hidden_layer_sizes`: 50 to 500 neurons in one hidden layer
- `activation`: `"relu"` or `"tanh"`
- `learning_rate_init`: 1e-4 to 1e-1

The search makes the first training slow (several seconds per unit). The parameter
`hidden_layer_dim` is required, but the current implementation does not use it.

### LinearModel

An ordinary least squares regression. It has no parameters. Its training does not return
a loss, so `show_error=True` prints `None` for this unit.

### CELM

An extreme learning machine with a random `tanh` hidden layer (`hidden_size` neurons).
The output weights are computed with a constrained least squares solver (qpsolvers with
OSQP), so you can restrict the output of the unit on the training samples with `constr`:

```python
# No constraints
CELM(mean_squared_error, {"hidden_size": 100, "constr": {}})

# Output between -0.5 and 0.5
CELM(mean_squared_error, {"hidden_size": 100, "constr": {"upper": 0.5, "inv_const": -0.5}})
```

- The key `"inv_const"` sets a lower bound.
- Every other key (except `"deriv"`) sets an upper bound. The name of the key is free.
- `"deriv"` sets an upper bound on the derivative of the hidden layer. This constraint
  is experimental and not covered by tests.
- A value can also be a tuple `(bound, sample_indexes)`. The constraint is then not
  applied to the given samples.
- For more than one output dimension, the bound is a sequence with one value per output.

## CascadeNetwork reference

`CascadeNetwork` is a subclass of `graphviz.Digraph`.

| Method | Description |
|--------|-------------|
| `CascadeNetwork(name)` | Create an empty cascade. |
| `add_cascade_layer(unit)` | Append a unit and a new output layer. Raises `ValueError` if the name of the unit is already used. |
| `remove_cascade_layer()` | Remove the last unit and its layer. |
| `train_networks(X, Y, show_error=False)` | Initialize the new units and train all units in order. Returns the loss of the last unit. |
| `apply_data(X)` | Run the cascade on `X` without training. |
| `predict(X, output_dim=3)` | Run the cascade and return the last `output_dim` columns of the last layer. |
| `get_last_layer_output()` | Return the data of the last layer (`[X, y_0, ..., y_n]`). |
| `get_target_layer()` | Return the target layer. |
| `draw_graph(path="/")` | Render the cascade structure with Graphviz into the directory `path`. |

The default value of `output_dim` is 3. Set it to the number of target columns.

## Writing your own unit

A supervised unit is a subclass of `SupervisedUnit` in `pydca.cascade.network_units`.
The cascade calls the following methods:

| Method | Called | Task |
|--------|--------|------|
| `optimize_unit()` | Once, before the first training | Return the parameters for `initialize_unit` (for example from a hyperparameter search). |
| `initialize_unit(params)` | Once, before the first training | Create the model and set `self.initialized = True`. |
| `casscade_train()` | On every `train_networks()` call | Read `self.input_layer.get_data()` and `self.target_layer.get_data()`, train the model, return the loss. |
| `casscade_apply()` | On every run of the cascade | Compute the output for `self.input_layer.get_data()` and pass it to `set_output`. |
| `set_output(data)` | From `casscade_apply()` | Write the output as a 2D array to `self.output_layer.set_state(data)`. |
| `train(X, Y)`, `activate(X)` | From your own code | Train the model and compute the output. |
| `parameters()` | When the cascade is built | Return a dict. Its keys are used as the edge label in the Graphviz drawing. |

`LinearModel` in `pydca/learning_models/sklearn/simple_regression_models.py` is the
shortest complete example.

## Project structure

```
pydca/
├── cascade/
│   ├── cascades.py        CascadeNetwork
│   ├── layers.py          Layer (data exchange between units)
│   ├── network_units.py   Base classes: CascadeUnit, SupervisedUnit, UnsupervisedUnit, DataUnit
│   ├── filters.py         Data filters for layers (Z-score, column selection)
│   └── metrics.py         Loss functions
└── learning_models/
    ├── sklearn/           MLPModel, LinearModel
    └── my_models/         CELM
docs/                      Documentation sources (Zensical)
tests/                     Tests
```

### Current limitations

- `Layer.get_data()` prints the position of the layer on every call. This debug output
  appears during training and prediction.
- The filters in `cascade/filters.py` are not yet available through
  `add_cascade_layer()`.

## Running the tests

```sh
uv run pytest
```

The tests in `tests/test_cascade_with_mlp_and_celm.py` train cascades of MLP and CELM
units on sine data. They take about two minutes, mainly because of the hyperparameter
search of `MLPModel`.

## Documentation

The documentation is built with [Zensical](https://zensical.org/). The API reference is
generated from the Sphinx-style docstrings (`:param x:`, `:type x:`, `:return:`) by
mkdocstrings; the docstring style is set with `docstring_style` in `zensical.toml`.

To preview the documentation at http://localhost:8000, run:

```sh
uv run zensical serve
```

To build the static site into `site/`, run:

```sh
uv run zensical build
```

## References

[1] E. Littmann and H. Ritter, "Learning and generalization in cascade network
architectures", *Neural Computation*, vol. 8, no. 7, pp. 1521–1539, 1996.
[doi:10.1162/neco.1996.8.7.1521](https://doi.org/10.1162/neco.1996.8.7.1521)

## Author

Heiko Donat (info@hdonat.net)

## License

Copyright © 2019–2026 Heiko Donat

Licensed under the EUPL, version 1.2 ([EUPL-1.2](https://interoperable-europe.ec.europa.eu/collection/eupl/eupl-text-eupl-12)).
The full licence text is in the file [`LICENSE`](LICENSE). The EUPL is available in all
official languages of the European Union; all language versions are equally valid.
