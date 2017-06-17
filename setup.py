#! /usr/bin/env python

from setuptools import setup, find_packages
from codecs import open
from os import path

if __name__ == '__main__':
    setup(
        name='pySMOTE',
        maintainer='Akaz',
        version='1.0.0',
        description='Python implementation of SMOTE',
        license='MIT',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 3'
        ],
        keywords='oversample oversampling SMOTE',
        packages=find_packages(),
        install_requires=['scipy>=0.19.0',
                          'numpy>=1.13.0',
                          'scikit-learn>=0.18.1'
        ]
    )
