import numpy as np
from sklearn.preprocessing import StandardScaler
class Filter:
    def __init__(self):
        pass

    def in_filter(self, data, net_pos):
        return data

    def out_filter(self, data):
        return data

    def __str__(self):
        return "$\all$"

class LinkedFilter(Filter):
    def __init__(self, other_filter = None):
        self._linked_filter = other_filter

class ZScoreFilter(LinkedFilter):
    def __init__(self, name=None, linked_filter=None):
        super().__init__(other_filter=linked_filter)
        self._scaler = StandardScaler()
        if name is None:
            self.name = id(self)
        else:
            self.name = name

    def in_filter(self, data, net_pos):
        self._scaler.fit(data)
        return self._scaler.transform(data)

    def out_filter(self, data):
        if self._linked_filter is None:
            return self._scaler.inverse_transform(data)
        else:
            return self._linked_filter._scaler.inverse_transform(data)

    def __str__(self):
        return "$Z-Score$"

class PickInputFilter(Filter):
    def __init__(self, columns, network_position):
        self._columns = columns
        self._network_position = network_position

    def in_filter(self, data, net_pos):
        if self._network_position == net_pos:
            return data[:, self._columns]
        else:
            return data

    def __str__(self):
        return "pick  " + str(self._columns) + " from " + str(self._network_position)

class FilterResolution(Filter):
    def __init__(self, orginal_data_size, new_size, start_of_reducable_data=5):
        super().__init__()
        self._new_size = new_size
        self._start_of_reducable_data = start_of_reducable_data
        self._orginal_data_size = orginal_data_size

    def filter(self, data):
        osize = self._orginal_data_size 
       # print(osize)
        size = data.shape[1]
        print(int(self._new_size))
        indexes = np.array(range(0,osize,int(self._new_size)))
        indexes *= 4
        print(indexes)
        indexes = np.array([[i,i+1,i+2,i+3] for i in indexes]).flatten() # TODO refactor to be used with changing arch point size
        indexes += self._start_of_reducable_data + 1
        indexes = list(range(6)) + indexes.tolist() + list(range((osize)*4 + self._start_of_reducable_data +1,size))
        print(indexes)
        return data[:, indexes]

class FilterCalculateError(Filter):
    def __init__(self, tlayer, input_size, output_size):
        super().__init__()
        self._t_out = tlayer.get_data()
        if self._t_out is None:
            pdb.set_trace()
        self._input_size = input_size
        self._output_size = output_size

    def filter(self, data):
        X = data[:,0:self._input_size]
        y_hat = data[:,-1 - self._output_size:-1]
        return np.hstack([X,self._t_out-y_hat, data[:,1:]])



