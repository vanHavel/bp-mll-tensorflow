import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from bpmll.bpmll import bp_mll_loss

if __name__ == '__main__':
    yeast = np.load('yeast.npz')
    X_train = yeast['X_train']
    Y_train = yeast['Y_train']

    n = X_train.shape[0]
    dim_no = X_train.shape[1]
    class_no = Y_train.shape[1]

    # create simple mlp
    model = Sequential()
    model.add(Dense(128, input_dim=dim_no, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(class_no, activation='sigmoid', kernel_initializer='glorot_uniform'))
    model.compile(loss=bp_mll_loss, optimizer='adagrad', metrics=[])

    # train a few epochs
    model.fit(X_train, Y_train, epochs=100)