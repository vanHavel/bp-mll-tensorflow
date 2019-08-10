import tensorflow as tf


def bp_mll_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.float64:
    """
    Computes bp mll loss function

    :param y_true: 2D integer tensor of true labels, of shape (number of samples, number of classes).
                   Values must be zero or one, where one means that the sample has the label.
                   Note that every sample must have at least one and at most (number of classes - 1) labels.

    :param y_pred: 2D float tensor of predictions, of shape (number of samples, number of classes).
                   Must have values between zero and one.

    :return: averaged bp mll loss
    """

    # get true and false labels
    shape = tf.shape(y_true)
    y_i = tf.equal(y_true, tf.ones(shape))
    y_i_bar = tf.not_equal(y_true, tf.ones(shape))

    # get indices to check
    truth_matrix = tf.to_float(pairwise_and(y_i, y_i_bar))

    # calculate all exp'd differences
    sub_matrix = pairwise_sub(y_pred, y_pred)
    exp_matrix = tf.exp(tf.negative(sub_matrix))

    # check which differences to consider and sum them
    sparse_matrix = tf.multiply(exp_matrix, truth_matrix)
    sums = tf.reduce_sum(sparse_matrix, axis=[1,2])

    # get normalizing terms and apply them
    y_i_sizes = tf.reduce_sum(tf.to_float(y_i), axis=1)
    y_i_bar_sizes = tf.reduce_sum(tf.to_float(y_i_bar), axis=1)
    normalizers = tf.multiply(y_i_sizes, y_i_bar_sizes)
    results = tf.divide(sums, normalizers)

    # average error
    return tf.reduce_mean(results)


def pairwise_sub(first_tensor: tf.Tensor, second_tensor: tf.Tensor) -> tf.Tensor:
    """
    Computes pairwise difference between elements of two tensors

    :param first_tensor: the first tensor
    :param second_tensor: the second tensor

    :return: pairwise difference between the two tensors
    """

    column = tf.expand_dims(first_tensor, 2)
    row = tf.expand_dims(second_tensor, 1)
    return tf.subtract(column, row)


def pairwise_and(first_tensor: tf.Tensor, second_tensor: tf.Tensor) -> tf.Tensor:
    """
    Computes pairwise logical and between elements of two tensors

    :param first_tensor: the first tensor
    :param second_tensor: the second tensor

    :return: pairwise logical and between the two tensors
    """

    column = tf.expand_dims(first_tensor, 2)
    row = tf.expand_dims(second_tensor, 1)
    return tf.logical_and(column, row)
