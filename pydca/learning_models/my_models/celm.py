import pdb
import scipy
import numpy as np
import numpy.matlib as npmat
from qpsolvers import solve_qp, solve_ls
from sklearn.preprocessing import StandardScaler

from pydca.cascade.network_units import SupervisedUnit


class CELM(SupervisedUnit):
    def __init__(self, loss_function, parameters = {}, name: str = None):
        super(CELM, self).__init__(loss_function, parameters, name)
        self.network_model = None
        self.input_layer = None
        self.output_layer = None
        self.initialized = False

        self._hidden_layer_size = parameters["hidden_size"]
        self.constr = parameters["constr"]

    def initialize_unit(self, params):
        if len(params['X'].shape) == 1:
            X = params['X'].reshape(-1,1)
        else:
            X = params['X']
        if len(params['Y'].shape) == 1:
            Y = params['Y'].reshape(-1,1)
        else:
            Y = params['Y']

        self._W_in = np.random.randn( params['hidden_size'], X.shape[1] +1 )
        self._W_out = np.random.randn(Y.shape[1],  params['hidden_size'])
        self._output_dim= Y.shape[1]
        self.initialized = True

    def build_constr(self, x):
        A = []
        c = []
        for key in self.constr.keys():
            if isinstance(self.constr[key], tuple):
                exep = self.constr[key][1]
                val = self.constr[key][0]
            else:
                val = self.constr[key]
            dir = 1
            if key == "inv_const":
              dir = -1

            if key == "deriv":
              A_tmp = self.derive(x)
            else:
              A_tmp = dir*self._calc_hidden_state(x)
            if self._output_dim > 1:
              c_tmp = np.vstack([dir*val[i]*np.ones((x.shape[0],1)) for i in range(self._output_dim)])
            else:
              c_tmp = dir*val*np.ones((x.shape[0],1))

            if isinstance(self.constr[key], tuple):
                A_tmp[exep,:] = np.zeros((1,A_tmp.shape[1]))
                c_tmp[exep,:] = np.zeros((1,c_tmp.shape[1]))
            A.append(A_tmp.T)
            c.append(c_tmp)

        if len(A) == 0:
            A = None
            c = None
        else:
            A = np.vstack(A)
            c = np.vstack(c)
        return A, c

    def _backward(self, T, h,x):
        A, c = self.build_constr(x)
        T = np.vstack([T[:,i:i+1] for i in range(T.shape[1])])
        #h =  scipy.sparse.csc.csc_matrix(h)
        #A =  scipy.sparse.csc.csc_matrix(A)
        #T =  scipy.sparse.csc.csc_matrix(T)
        #c =  scipy.sparse.csc.csc_matrix(c)
        W_out = solve_ls(h, T, A, c, solver="osqp")
        W_out = W_out.reshape(self._output_dim,-1)
        return W_out

    def _calc_block_matrix(self, h):

      # create blocks
      blocks = [h for i in range(self._output_dim)]

      # Get the number of rows and columns in each block
      block_rows = [block.shape[0] for block in blocks]
      block_cols = [block.shape[1] for block in blocks]

      # Determine the total number of rows and columns in the block diagonal matrix
      total_rows = sum(block_rows)
      total_cols = sum(block_cols)

      # Initialize the block diagonal matrix with zeros
      block_diagonal_matrix = np.zeros((total_rows, total_cols))

      # Fill in the diagonal blocks
      row_start = 0
      col_start = 0

      for i, block in enumerate(blocks):
          row_end = row_start + block_rows[i]
          col_end = col_start + block_cols[i]
          block_diagonal_matrix[row_start:row_end, col_start:col_end] = block
          row_start = row_end
          col_start = col_end

      return block_diagonal_matrix

    def _calc_hidden_state(self,X, no_stacking=False):
       # pdb.set_trace()
        _bias = np.ones((X.shape[0],1))
        H =  np.tanh(self._W_in @ np.hstack([X, _bias]).T)
        if self._output_dim > 1 and not no_stacking:
          H = self._calc_block_matrix(H)
          print(f"{H.shape=}")
        return H#sc.linalg.block_diag(*[h for h in H.T]), H

    def derive(self,X):
        H = self._calc_hidden_state(X)
        return (1/np.cosh(H)**2) * npmat.repmat(self._W_in[:,:-1],1,X.shape[0])

    def fit(self,X , Y):
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        H = self._calc_hidden_state(X)
        self._W_out = self._backward(Y, H.T, X)
        return H.T

    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1,1)
        h = self._calc_hidden_state(X, True)
        return (self._W_out @ h).T

        
    def train(self, X, Y):
        self.fit(X,Y)
        return self.predict(X)

    def activate(self, X):
        return self.predict(X)

    def set_output(self, data):
        if len(data.shape) == 1:
            ret = data.reshape(-1,1)
        else:
            ret = data
        if not (self.output_layer is None):
            self.output_layer.set_state(ret)
        return ret

    def casscade_train(self):
        #pdb.set_trace()
        in_data = self.input_layer.get_data().copy()
        #in_data = StandardScaler().fit_transform(in_data)
        out_data = self.target_layer.get_data()
        #pdb.set_trace()
        yhats = self.train(in_data, out_data)
        #pdb.set_trace()
        return self._calc_loss_function(out_data, yhats)

    def casscade_apply(self):
        in_data = self.input_layer.get_data().copy()
        #in_data = StandardScaler().fit_transform(in_data)
        elm_output = self.activate(in_data)
        self.set_output(elm_output)
    def optimize_unit(self):
        X_train = self.input_layer.get_data()
        y_train = self.target_layer.get_data()
        params = {
            'X': X_train,
            'Y': y_train,
            'hidden_size': self._hidden_layer_size,
        }
        return params

    def parameters(self):
        return {"hidden_layer_dim": None}

