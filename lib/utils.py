import tensorflow as tf

def get_static_shape(tensor):
    """
    Get the statis shape of specified `tensor` as a tuple

    Args:
        tensor: the tensor object

    Return:
        tuple[int or None] or None: the static shape tuple,
        or None if the dimensions of `tensor` is not deterministic
    """
    tensor = tf.convert_to_tensor(tensor)
    shape = tensor.get_shape()
    if shape.ndims is None:
        shape = None
    else:
        shape = tuple((int(v) if v is not None else None)
                      for v in shape.as_list())
    return shape

def flatten_to_ndims(x, ndims):
    """
    Flatten the front dimension of `x`, such that the resulting tensor will have at most `ndims` dimension

    Args:
        x (Tensor): the tensor to be flatten
        ndims (int): the maximum number of dimensions for the resulting tensor

    Returns:
        (tf.Tensor, tuple[int or None], tuple[int] or tf.Tensor) or (tf.Tensor, None, None):
            (The flatten tensor, the static front shape, and the front shape),
            or (the originial tensor, None, None)
    """
    x = tf.convert_to_tensor(x)
    if ndims < 1:
        raise ValueError('`k` must be greater or equal to 1.')
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')
    shape = get_static_shape(x)
    if len(shape) < ndims:
        raise ValueError('`k` is {}, but `x` only has rank {}.'.
                         format(ndims, len(shape)))
    if len(shape) == ndims:
        return x, None, None

    if ndims == 1:
        static_shape = shape
        if None in shape:
            shape = tf.shape(x)
        return tf.reshape(x, [-1]), static_shape, shape
    else:
        front_shape, back_shape = shape[:-(ndims - 1)], shape[-(ndims - 1):]
        static_front_shape = front_shape
        static_back_shape = back_shape
        if None in front_shape or None in back_shape:
            dynamic_shape = tf.shape(x)
            if None in front_shape:
                front_shape = dynamic_shape[:-(ndims - 1)]
            if None in back_shape:
                back_shape = dynamic_shape[-(ndims - 1):]
        if isinstance(back_shape, tuple):
            x = tf.reshape(x, [-1] + list(back_shape))
        else:
            x = tf.reshape(x, tf.concat([[-1], back_shape], axis=0))
            x.set_shape(tf.TensorShape([None] + list(static_back_shape)))
        return x, static_front_shape, front_shape


def unflatten_from_ndims(x, static_front_shape, front_shape):
    """
    The inverse transformation of function flatten

    If both `static_front_shape` is None and `front_shape` is None,
    `x` will be returned without any change.

    Args:
        x (Tensor): the tensor to be unflatten
        static_front_shape (tuple[int or None] or None): the static front shape
        front_shape (tuple[int] or tf.Tensor or None): the front shape

    Return:
        tf.Tensor: the unflatten x
    """
    x = tf.convert_to_tensor(x)
    if static_front_shape is None and front_shape is None:
        return x
    if not x.get_shape():
        raise ValueError('`x` is required to have known number of '
                         'dimensions.')

    shape = get_static_shape(x)
    if len(shape) < 1:
        raise ValueError('`x` only has rank {}, required at least 1.'.
                         format(len(shape)))

    if not isinstance(front_shape, tf.Tensor):
        front_shape = tuple(front_shape)

    back_shape = shape[1:]
    static_back_shape = back_shape
    if None in back_shape:
        back_shape = tf.shape(x)[1:]
    if isinstance(front_shape, tuple) and isinstance(back_shape, tuple):
        x = tf.reshape(x, front_shape + back_shape)
    else:
        x = tf.reshape(x, tf.concat([front_shape, back_shape], axis=0))
        x.set_shape(tf.TensorShape(list(static_front_shape) + list(static_back_shape)))
    return x

def get_best_f1(score, label):
    '''
    :param score: 1-D array, input score, tot_length
    :param label: 1-D array, standard label for anomaly
    :return: list for results, threshold
    '''

    assert score.shape == label.shape
    print('***computing best f1***')
    search_set = []
    tot_anomaly = 0
    for i in range(label.shape[0]):
        tot_anomaly += (label[i] > 0.5)
    flag = 0
    cur_anomaly_len = 0
    cur_min_anomaly_score = 1e5
    for i in range(label.shape[0]):
        if label[i] > 0.5:
            # here for an anomaly
            if flag == 1:
                cur_anomaly_len += 1
                cur_min_anomaly_score = score[i] if score[i] < cur_min_anomaly_score else cur_min_anomaly_score
            else:
                flag = 1
                cur_anomaly_len = 1
                cur_min_anomaly_score = score[i]
        else:
            # here for normal points
            if flag == 1:
                flag = 0
                search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
                search_set.append((score[i], 1, False))
            else:
                search_set.append((score[i], 1, False))
    if flag == 1:
        search_set.append((cur_min_anomaly_score, cur_anomaly_len, True))
    search_set.sort(key=lambda x: x[0])
    best_f1_res = - 1
    threshold = 1
    P = 0
    TP = 0
    best_P = 0
    best_TP = 0
    for i in range(len(search_set)):
        P += search_set[i][1]
        if search_set[i][2]:  # for an anomaly point
            TP += search_set[i][1]
        precision = TP / (P + 1e-5)
        recall = TP / (tot_anomaly + 1e-5)
        f1 = 2 * precision * recall / (precision + recall + 1e-5)
        if f1 > best_f1_res:
            best_f1_res = f1
            threshold = search_set[i][0]
            best_P = P
            best_TP = TP

    print('***  best_f1  ***: ', best_f1_res)
    print('*** threshold ***: ', threshold)
    return (best_f1_res,
            best_TP / (best_P + 1e-5),
            best_TP / (tot_anomaly + 1e-5),
            best_TP,
            score.shape[0] - best_P - tot_anomaly + best_TP,
            best_P - best_TP,
            tot_anomaly - best_TP), threshold

def get_data_dim(dataset):
    if dataset == 'SWaT':
        return 51
    elif dataset == 'WADI':
        return 118
    elif str(dataset).startswith('machine'):
        return 38
    elif str(dataset).startswith('omi'):
        return 19
    elif dataset == 'PSM':
        return 25
    else:
        raise ValueError('unknown dataset '+str(dataset))

def get_data_name(dataset):
    if str(dataset).startswith('omi'):
        return 'ASD'
    elif str(dataset).startswith('machine'):
        return 'SMD'
    else:
        return dataset