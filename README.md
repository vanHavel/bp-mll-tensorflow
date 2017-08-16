# bp-mll-tensorflow
Efficient (vectorized) implementation of the BP-MLL loss function in TensorFlow (bp_mll.py). 

BP-MLL is a loss function designed for multi-label classification using neural networks. It was introduced by Zhang & Zhou in [1].

There is also an alternative implementation using the Keras API in bp_mll_keras.py, which can be used with any backend supported by Keras.

Check bp_mll_test.py and bp_mll_test_keras.py for an example.

# Requirements: 
- Python3
- Numpy 
- TensorFlow (or Keras)


# References
[1] Zhang, Min-Ling, and Zhi-Hua Zhou. "Multilabel neural networks with applications to functional genomics and text categorization." IEEE transactions on Knowledge and Data Engineering 18.10 (2006): 1338-1351.