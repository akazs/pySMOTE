"""
Python implementation of SMOTE.
Original paper: https://www.jair.org/media/953/live-953-2037-jair.pdf
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    def __init__(self,
                 percentage=100,
                 k_neighbors=6):
        # check input arguments
        if percentage > 0 and percentage < 100:
            self.percentage = percentage
        elif percentage >= 100:
            if percentage % 100 == 0:
                self.percentage = percentage
            else:
                raise ValueError(
                    'percentage over 100 should be multiples of 100')
        else:
            raise ValueError(
                'percentage should be greater than 0')

        if type(k_neighbors) == int:
            if k_neighbors > 0:
                self.k_neighbors = k_neighbors
            else:
                raise ValueError(
                    'k_neighbors should be integer greater than 0')
        else:
            raise TypeError(
                'Expect integer for k_neighbors')

    def _randomize(self,samples,ratio):
        length = samples.shape[0]
        target_size = length * ratio
        idx = np.random.randint(length, size=target_size)

        return samples[idx, :]

    def _populate(self, idx, nnarray):
        N = self.N

        while N > 0:
            nn = np.random.randint(low=0, high=self.k_neighbors)
            new_entry = np.empty(shape=self.numattrs)
            for attr in range(self.numattrs):
                dif = (self.samples[nnarray[nn]][attr]
                       - self.samples[idx][attr])
                gap = np.random.uniform()
                self.synthetic[self.newidx][attr] = (self.samples[idx][attr]
                                                     + gap * dif)
            else:
                self.synthetic = np.concatenate((self.synthetic,
                                                 [new_entry]))
            self.newidx += 1
            N -= 1

    def oversample(self,samples):
        if type(samples) == list:
            self.samples = np.array(samples)
        elif type(samples) == np.ndarray:
            self.samples = samples
        else:
            raise TypeError(
                'Expect a built-in list or an ndarray for samples')

        self.numattrs = samples.shape[1]

        if self.percentage < 100:
            ratio = percentage / 100.0
            self.samples = self._randomize(self.samples, ratio) 
            self.percentage = 100

        self.N = int(self.percentage / 100)
        new_shape = (self.samples.shape[0] * self.N, self.samples.shape[1])
        self.synthetic = np.empty(shape=new_shape)
        self.newidx = 0

        self.nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nbrs.fit(samples)
        self.knn = self.nbrs.kneighbors()[1]

        for idx in range(self.samples.shape[0]):
            nnarray = self.knn[idx]
            self._populate(idx, nnarray)

        return np.concatenate((self.samples, self.synthetic))
