"""
Python implementation of SMOTE.
This implementation is based on the original variant of SMOTE.
Original paper: https://www.jair.org/media/953/live-953-2037-jair.pdf
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    '''
        Python implementation of SMOTE.

        This implementation is based on the original variant of SMOTE.

        Parameters
        ----------
        ratio : int, optional (default=100)
            The ratio percentage of generated samples to original samples.

            - If ratio < 100, then randomly choose ratio% of samples to SMOTE.
            - If ratio >= 100, it must be a interger multiple of 100.

        k_neighbors : int, optional (defalut=6)
            Number of nearest neighbors to used to SMOTE.

        random_state : int, optional (default=None)
            The random seed of the random number generator.
    '''
    def __init__(self,
                 ratio=100,
                 k_neighbors=6,
                 random_state=None):
        # check input arguments
        if ratio > 0 and ratio < 100:
            self.ratio = ratio
        elif ratio >= 100:
            if ratio % 100 == 0:
                self.ratio = ratio
            else:
                raise ValueError(
                    'ratio over 100 should be multiples of 100')
        else:
            raise ValueError(
                'ratio should be greater than 0')

        if type(k_neighbors) == int:
            if k_neighbors > 0:
                self.k_neighbors = k_neighbors
            else:
                raise ValueError(
                    'k_neighbors should be integer greater than 0')
        else:
            raise TypeError(
                'Expect integer for k_neighbors')

        if type(random_state) == int:
            np.random.seed(random_state)

    def _randomize(self, samples, ratio):
        length = samples.shape[0]
        target_size = length * ratio
        idx = np.random.randint(length, size=target_size)

        return samples[idx, :]

    def _populate(self, idx, nnarray):
        for i in range(self.N):
            nn = np.random.randint(low=0, high=self.k_neighbors)
            for attr in range(self.numattrs):
                dif = (self.samples[nnarray[nn]][attr]
                       - self.samples[idx][attr])
                gap = np.random.uniform()
                self.synthetic[self.newidx][attr] = (self.samples[idx][attr]
                                                     + gap * dif)
            self.newidx += 1

    def oversample(self, samples, merge=False):
    '''
        Perform oversampling using SMOTE

        Parameters
        ----------
        samples : list or ndarray, shape (n_samples, n_features)
            The samples to apply SMOTE to.

        merge : bool, optional (default=False)
            If set to true, merge the synthetic samples to original samples.

        Returns
        -------
        output : ndarray
            The output synthetic samples.
    '''
        if type(samples) == list:
            self.samples = np.array(samples)
        elif type(samples) == np.ndarray:
            self.samples = samples
        else:
            raise TypeError(
                'Expect a built-in list or an ndarray for samples')

        self.numattrs = samples.shape[1]

        if self.ratio < 100:
            ratio = ratio / 100.0
            self.samples = self._randomize(self.samples, ratio) 
            self.ratio = 100

        self.N = int(self.ratio / 100)
        new_shape = (self.samples.shape[0] * self.N, self.samples.shape[1])
        self.synthetic = np.empty(shape=new_shape)
        self.newidx = 0

        self.nbrs = NearestNeighbors(n_neighbors=self.k_neighbors)
        self.nbrs.fit(samples)
        self.knn = self.nbrs.kneighbors()[1]

        for idx in range(self.samples.shape[0]):
            nnarray = self.knn[idx]
            self._populate(idx, nnarray)

        if merge:
            return np.concatenate((self.samples, self.synthetic))
        else:
            return self.synthetic
