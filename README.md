# pySMOTE
Python implementation of [SMOTE](https://www.jair.org/media/953/live-953-2037-jair.pdf): Synthetic Minority Over-sampling Technique

## Installation
### Dependencies
Tested on Python 2.7 and 3.6, with:
* scipy>=0.19.0
* numpy>=1.13.0
* scikit-learn>=0.18.1

Older versions may work but not tested.

### Installation
``` bash
git clone https://github.com/akazs/pySMOTE.git
cd pySMOTE
pip install .
```

## Usage
``` python
import pySMOTE

smote = pySMOTE.SMOTE(ratio=100, k_neighbors=6)
synthetic_samples = smote.oversample(sample_data)
new_samples = smote.oversample(sample_data, merge=True)
```
