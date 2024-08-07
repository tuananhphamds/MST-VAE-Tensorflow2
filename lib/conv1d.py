from tensorflow.keras.layers import Conv1D, Conv1DTranspose

def conv1d(hidden, filters, kernel_size, strides=1, activation=None,
           padding='same', kernel_regularizer=None):
    hidden = Conv1D(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    activation=activation,
                    padding=padding,
                    kernel_regularizer=kernel_regularizer)(hidden)
    return hidden


def deconv1d(hidden, filters, kernel_size, strides=1, activation=None,
             padding='same', kernel_regularizer=None, output_shape=None):
    """Calculate output_padding based on output_shape
    There is a confusion when upsampling the input
    For example:
    if the `steps` dimension of the input is 8, which one is original `step`
    dimension 15 or 16?

    Therefore, output_padding will remove the confusion
    output_padding = None -> original dimension is 16
    output_padding = 0 -> original dimension is 15
    """
    output_padding = None
    steps = hidden.get_shape()[-2]
    if strides > 1 and steps * 2 != output_shape:
        pad = kernel_size // 2
        output_padding = output_shape - ((steps - 1) * strides + kernel_size -
                                         2 * pad)
        if output_padding >= strides or output_padding < 0:
            raise ValueError('output_padding {} is invalid'.format(output_padding))

    hidden = Conv1DTranspose(filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             activation=activation,
                             padding=padding,
                             kernel_regularizer=kernel_regularizer,
                             output_padding=output_padding)(hidden)
    return hidden