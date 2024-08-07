import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K

class Normal:
    """
    The class of univariate Normal distribution

    : param mean: a `float` Tensor. the mean of the Normal distribution.
    : param logstd: a `float` Tensor. The log standard deviation of the Normal distribution.
    : param group_ndims: A 0-D int32 Tensor representing the number of dimensions in `batch_shape` (counted from the end)
        that are grouped into a single event, so that their probabilities are calculated together. Default is 0, which means
        a single value is an event.
    : param: is_reparameterized: bool. If true, gradients on samples from this distribution are allowed to propagate into inputs
    """

    def __init__(self, mean, logstd=None, group_ndims=0):
        self._mean = tf.convert_to_tensor(mean)

        if logstd is None:
            raise ValueError('Log standard deviation must not be None')
        self._logstd = tf.convert_to_tensor(logstd)

        std = tf.exp(self._logstd)
        std = tf.debugging.check_numerics(std, "exp(logstd)")
        self._std = std
        self._group_ndims = group_ndims
        self._dtype = self._mean.dtype
        try:
            tf.broadcast_static_shape(self._mean.get_shape(), self._std.get_shape())
        except ValueError:
            raise ValueError(
                "mean and logstd should be boardcastable to match each other. "
                "({} vs {})".format(self._mean.get_shape(), self._logstd.get_shape())
            )

    @property
    def dtype(self):
        """
        The sample type of the distribution
        """
        return self._dtype

    @property
    def mean(self):
        """
        The mean of the Normal distribution
        """
        return self._mean

    @property
    def logstd(self):
        """
        The log standard deviation of the Normal distribution
        """
        return self._logstd

    @property
    def std(self):
        """
        The standard deviation of the Normal distribution
        """
        return self._std

    @property
    def group_ndims(self):
        """
        The number of dimension counted from the end, that are grouped into a single
        event, so that their probabilities are calculated together
        """
        return self._group_ndims

    def get_batch_shape(self):
        try:
            tf.broadcast_static_shape(self._mean.get_shape(), self._std.get_shape())
        except ValueError:
            raise ValueError(
                "mean and logstd should be boardcastable to match each other. "
                "({} vs {})".format(self._mean.get_shape(), self._logstd.get_shape())
            )
        return self.mean.get_shape()

    @property
    def batch_shape(self):
        static_batch_shape = self.get_batch_shape()
        return static_batch_shape

    def _sample(self, n_samples):
        batch_shape = tf.shape(self.mean)[0]
        hidden_dim = K.int_shape(self.mean)[1:]

        mean, std = self.mean, self.std
        shape = (n_samples, batch_shape) + hidden_dim
        samples = tf.random.normal(shape=shape, dtype=self.dtype) * std + mean
        return samples

    def sample(self, n_samples=1):
        if n_samples is None or n_samples == 1:
            samples = self._sample(n_samples=1)
            samples = tf.squeeze(samples, axis=0)
        else:
            samples = self._sample(n_samples)

        return samples

    def _log_prob(self, given):
        c = -0.5 * np.log(2 * np.pi)
        precision = tf.exp(-2 * self.logstd)
        precision = tf.debugging.check_numerics(precision, "precision")
        return c - self.logstd - 0.5 * precision * tf.square(given - self.mean)

    def _prob(self, given):
        return tf.exp(self._log_prob(given))

    def _check_input_shape(self, given):
        given = tf.convert_to_tensor(given, dtype=self.dtype)

        err_msg = "The given argument should be able to broadcast to" \
                  "match batch_shape + value_shape of the distribution"

        if (given.get_shape() and self.get_batch_shape() and self._get_value_shape()):


            static_sample_shape = tf.TensorShape(
                self.get_batch_shape().as_list() +
                self._get_value_shape().as_list()
            )
            try:
                tf.broadcast_static_shape(given.get_shape(),
                                          static_sample_shape)
            except ValueError:
                raise ValueError(
                    err_msg + " ({} vs. {} + {})".format(
                        given.get_shape(), self.get_batch_shape(),
                        self._get_value_shape())
                )
        return given

    def _get_value_shape(self):
        return tf.TensorShape([])

    def log_prob(self, given, use_group_ndims=True):
        given = self._check_input_shape(given)
        log_p = self._log_prob(given)

        if use_group_ndims:
            log_p = tf.reduce_sum(log_p, tf.range(-self._group_ndims, 0))
        else:
            log_p = tf.reduce_sum(log_p, [])

        return log_p

    def prob(self, given, use_group_ndims=True):
        given = self._check_input_shape(given)
        p = self._prob(given)

        if use_group_ndims:
            p = tf.reduce_prod(p, tf.range(-self._group_ndims, 0))
        else:
            p = tf.reduce_prod(p, [])
        return p