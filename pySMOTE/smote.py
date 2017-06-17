"""
Python implementation of SMOTE.
Original paper: https://www.jair.org/media/953/live-953-2037-jair.pdf
"""

import numpy as np


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
