# bp-mll-tensorflow
Efficient (vectorized) implementation of the BP-MLL loss function in TensorFlow (```bp_mll.py```). 

BP-MLL is a loss function designed for multi-label classification using neural networks. It was introduced by Zhang & Zhou in [1]. Note that in line with [1], every sample needs to have at least one label and no sample may have all labels.

# Installation 
`pip3 install bpmll`

# Usage
```
from bpmll import bp_mll_loss
```
Then simply use it as a function in your tensorflow or keras models.

Check out ```full_example.py``` for an example of training a simple multilayer perceptron using Keras with BP-MLL.

# References
[1] Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.
