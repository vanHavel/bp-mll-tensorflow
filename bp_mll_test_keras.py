import numpy as np
from keras import backend as K

from bp_mll_keras import bp_mll_loss


# 2 samples, 4 possible labels
y_true = np.asarray([[ 1, -1, -1,  1], 
                     [-1, -1,  1, -1]],
                    dtype='float32')
          
# predictions for the samples        
y_pred = np.asarray([[0.8, 0.3, 0.1, 0.9], 
                     [0.1, 0.0, 1.0, 0.5]],
                    dtype='float32')

# compute result in keras
result = K.eval(bp_mll_loss(y_true, y_pred))

# should print 0.49282038
print(result)