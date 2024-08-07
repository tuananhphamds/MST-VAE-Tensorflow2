import os
import json
import tensorflow as tf
import numpy as np
from tqdm import tqdm


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, model_from_json
from tensorflow.data import Dataset
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from normal_distribution import Normal
from utils import flatten_to_ndims, unflatten_from_ndims

from conv1d import conv1d, deconv1d

class MSTVAEModel(Model):
    def __init__(self, cfg):
        super(MSTVAEModel, self).__init__()

        self.cfg = cfg

        self.encoder = self.encoder()
        self.decoder = self.decoder()

        """
                Because Tensorflow does not calculate average loss after each epoch,
                specifically the loss of the last batch is saved when training an epoch is
                finished. This will lead to bias if we use EarlyStopping callback to stop 
                training when the model is overfitting.
                Therefore, `self.epoch_losses` is created to calculate the average loss after
                each epoch
        """
        self.average_epoch_eval = dict()
        self.build(input_shape=(1, self.cfg['window_size'], self.cfg['x_dim']))

    def encoder(self):
        inputs = Input(shape=(self.cfg['window_size'], self.cfg['x_dim']))

        qz_mean, qz_logstd = self.h_for_qz(inputs)

        model = Model(inputs, (qz_mean, qz_logstd))

        return model

    def decoder(self):
        inputs = Input(shape=(self.cfg['z_dim'], self.cfg['x_dim']))

        px_mean, px_logstd = self.h_for_px(inputs)

        model = Model(inputs, (px_mean, px_logstd))

        return model

    def h_for_qz(self, x):
        output_shapes = self.cfg['output_shapes']

        # Extract features from short-scale module
        h_x1 = None
        for i in range(len(output_shapes)):
            if h_x1 is None:
                h_x1 = conv1d(x, kernel_size=2, filters=self.cfg['x_dim'], strides=2,
                              activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))
            else:
                h_x1 = conv1d(h_x1, kernel_size=2, filters=self.cfg['x_dim'], strides=2,
                              activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        # Extract features from long-scale module
        h_x2 = None
        for i in range(len(output_shapes)):
            if h_x2 is None:
                h_x2 = conv1d(x, kernel_size=15, filters=self.cfg['x_dim'], strides=2,
                              activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))
            else:
                h_x2 = conv1d(h_x2, kernel_size=2, filters=self.cfg['x_dim'], strides=2,
                              activation='relu', padding='same',
                              kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        # Concatenate layer
        concat = tf.concat([h_x1, h_x2], axis=-1)

        # Feature reduction layer
        conv_concat = conv1d(concat, kernel_size=1, strides=1,
                             filters=self.cfg['x_dim'], activation='relu',
                             kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        # Mean and log standard deviation layers
        qz_mean = conv1d(conv_concat, kernel_size=1, filters=self.cfg['x_dim'])
        qz_logstd = conv1d(conv_concat, kernel_size=1, filters=self.cfg['x_dim'])
        qz_logstd = tf.clip_by_value(qz_logstd,
                                     clip_value_min=self.cfg['logstd_min'],
                                     clip_value_max=self.cfg['logstd_max'])

        return qz_mean, qz_logstd

    def h_for_px(self, z):
        output_shapes = self.cfg['output_shapes']

        # Reconstruct input from short-scale features
        h_z1 = deconv1d(z, filters=self.cfg['x_dim'], kernel_size=1,
                        strides=1, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        for i in range(len(output_shapes)):
            h_z1 = deconv1d(h_z1, filters=self.cfg['x_dim'], kernel_size=2,
                            strides=2, activation='relu', padding='same',
                            kernel_regularizer=regularizers.l2(self.cfg['l2_reg']),
                            output_shape=output_shapes[i])

        # Reconstruct input from long-scale features
        h_z2 = deconv1d(z, filters=self.cfg['x_dim'], kernel_size=1,
                        strides=1, activation='relu', padding='same',
                        kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        for i in range(len(output_shapes)):
            if i == len(output_shapes) - 1:
                h_z2 = deconv1d(h_z2, filters=self.cfg['x_dim'], kernel_size=15,
                                strides=2, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(self.cfg['l2_reg']),
                                output_shape=output_shapes[i])
            else:
                h_z2 = deconv1d(h_z2, filters=self.cfg['x_dim'], kernel_size=2,
                                strides=2, activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(self.cfg['l2_reg']),
                                output_shape=output_shapes[i])

        # Concatenate layer
        concat = tf.concat([h_z1, h_z2], axis=-1)

        # Feature reduction layer
        h_z = deconv1d(concat, filters=self.cfg['x_dim'], kernel_size=1,
                       strides=1, activation='relu',
                       kernel_regularizer=regularizers.l2(self.cfg['l2_reg']))

        # Mean and log standard deviation layers
        px_mean = conv1d(h_z, kernel_size=1, filters=self.cfg['x_dim'])
        px_logstd = conv1d(h_z, kernel_size=1, filters=self.cfg['x_dim'])
        px_logstd = tf.clip_by_value(px_logstd,
                                     clip_value_min=self.cfg['logstd_min'],
                                     clip_value_max=self.cfg['logstd_max'])

        return px_mean, px_logstd

    def sgvb_loss(self,
                  inputs, px_dist,
                  qz_samples, prior_dist,
                  posterior_dist):
        """
        ELBO = Eq(z|x)[logpx_z + logpz - logqz_x]
        Reconstruction error: logpx_z
        KL: logpz - logqz_x
        Maximize ELBO means minimize negative ELBO
        Loss = -ELBO
        """
        logpx_z = px_dist.log_prob(inputs)
        logpz = prior_dist.log_prob(qz_samples)
        logqz_x = posterior_dist.log_prob(qz_samples)

        recons_term = tf.reduce_mean(logpx_z)
        kl_term = tf.reduce_mean(logqz_x - logpz)

        return -tf.reduce_mean(logpx_z + 0.2 * (logpz - logqz_x)), recons_term, kl_term

    def get_config(self):
        config = {"cfg": self.cfg}
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as encoder_tape, tf.GradientTape() as decoder_tape:
            # Get mean_z and logstd_z
            qz_mean, qz_logstd = self.encoder(inputs, training=True)

            # Create posterior distribution and sampling from z(posterior)
            posterior_dist = Normal(mean=qz_mean, logstd=qz_logstd, group_ndims=2)
            qz_samples = posterior_dist.sample()

            # Create prior distribution
            prior_dist = Normal(mean=tf.zeros([self.cfg['z_dim'], self.cfg['x_dim']]),
                                logstd=tf.zeros([self.cfg['z_dim'], self.cfg['x_dim']]),
                                group_ndims=2)

            # Get mean_x and logstd_x
            px_mean, px_logstd = self.decoder(qz_samples, training=True)

            # Create distribution of x
            px_dist = Normal(mean=px_mean, logstd=px_logstd, group_ndims=2)

            # Calculate loss
            loss, recons_term, kl_term = self.sgvb_loss(inputs, px_dist,
                                                      qz_samples, prior_dist,
                                                      posterior_dist)

            # Add regularization loss
            loss += tf.add_n(self.encoder.losses)
            loss += tf.add_n(self.decoder.losses)

        gradients_of_enc = encoder_tape.gradient(loss, self.encoder.trainable_variables)
        gradients_of_dec = decoder_tape.gradient(loss, self.decoder.trainable_variables)

        grads_of_enc = [tf.clip_by_norm(g, 10) for g in gradients_of_enc]
        grads_of_dec = [tf.clip_by_norm(g, 10) for g in gradients_of_dec]

        self.optimizer.apply_gradients(zip(grads_of_enc, self.encoder.trainable_variables))
        self.optimizer.apply_gradients(zip(grads_of_dec, self.decoder.trainable_variables))

        return {'loss': loss, 'recons': recons_term, 'kl': kl_term}

    @tf.function
    def test_step(self, inputs):
        if self.cfg['n_samples'] <= 1:
            raise ValueError('Number of samples drawing from latent '
                             'representation must be larger than 1 '
                             'current is {}'.format(self.cfg['n_samples']))

        # Get mean_z and logstd_z
        qz_mean, qz_logstd = self.encoder(inputs, training=False)

        # Create posterior distribution and sampling from z (posterior)
        posterior_dist = Normal(mean=qz_mean, logstd=qz_logstd, group_ndims=2)
        qz_samples = posterior_dist.sample(self.cfg['n_samples'])

        # Create prior distribution
        prior_dist = Normal(mean=tf.zeros([self.cfg['z_dim'], self.cfg['x_dim']]),
                            logstd=tf.zeros([self.cfg['z_dim'], self.cfg['x_dim']]),
                            group_ndims=2)

        # Get mean_x and logstd_x
        hidden, static_front_shape, front_shape = flatten_to_ndims(qz_samples, 3)
        px_mean, px_logstd = self.decoder(hidden, training=False)
        px_mean = unflatten_from_ndims(px_mean, static_front_shape, front_shape)
        px_logstd = unflatten_from_ndims(px_logstd, static_front_shape, front_shape)

        # Create distribution of x
        px_dist = Normal(mean=px_mean, logstd=px_logstd, group_ndims=2)

        # Calculate loss
        loss, recons_term, kl_term = self.sgvb_loss(inputs, px_dist,
                                                  qz_samples, prior_dist,
                                                  posterior_dist)

        return {'loss': loss, 'recons': recons_term, 'kl': kl_term}

    @tf.function
    def call(self, inputs):
        """
        This function is to predict the given inputs (a batch data)
        This is used within predict() function
        """
        if self.cfg['n_samples'] <= 1:
            raise ValueError('Number of samples drawing from latent '
                             'representation must be larger than 1 '
                             'current is {}'.format(self.cfg['n_samples']))

        # Get mean_z and logstd_z
        qz_mean, qz_logstd = self.encoder(inputs, training=False)

        # Create posterior distribution and sampling from z (posterior)
        posterior_dist = Normal(mean=qz_mean, logstd=qz_logstd, group_ndims=2)
        qz_samples = posterior_dist.sample(self.cfg['n_samples'])

        # Get mean_x and logstd_x
        hidden, static_front_shape, front_shape = flatten_to_ndims(qz_samples, 3)
        px_mean, px_logstd = self.decoder(hidden, training=False)
        px_mean = unflatten_from_ndims(px_mean, static_front_shape, front_shape)
        px_logstd = unflatten_from_ndims(px_logstd, static_front_shape, front_shape)

        # Create distribution of x
        px_dist = Normal(mean=px_mean, logstd=px_logstd, group_ndims=2)

        px_log_prob = px_dist.log_prob(inputs, use_group_ndims=False)

        # Calculate average over sampling dimension
        px_mean = tf.reduce_mean(px_mean, axis=0)
        px_log_prob = tf.reduce_mean(px_log_prob, axis=0)

        return px_mean[:, -1, :], px_log_prob[:, -1, :]

    def evaluate(self, **kwargs):
        val_logs = super(MSTVAEModel, self).evaluate(**kwargs)
        # Instead of using the loss of the last batch when evaluating model
        # We use the average losses instead
        if self.average_epoch_eval:
            val_logs = self.average_epoch_eval.copy()
        return val_logs

    @tf.function
    def _sampling_from_x(self, batch_inputs):
        # Get mean_z and logstd_z
        qz_mean, qz_logstd = self.encoder(batch_inputs, training=False)

        # Create posterior distribution and sampling from z (posterior)
        posterior_dist = Normal(mean=qz_mean, logstd=qz_logstd, group_ndims=2)
        qz_samples = posterior_dist.sample()  # n_samples=1

        # Get mean_x and logstd_x
        px_mean, px_logstd = self.decoder(qz_samples, training=False)

        # Create distribution of x
        px_dist = Normal(mean=px_mean, logstd=px_logstd, group_ndims=2)

        x_samples = px_dist.sample()  # Reconstruct x
        return x_samples

    @tf.function
    def _reconstruct_x_with_mcmc_recons(self, batch_inputs, recons):
        # Get mean_z and logstd_z
        qz_mean, qz_logstd = self.encoder(recons, training=False)

        # Create posterior distribution and sampling from z (posterior)
        posterior_dist = Normal(mean=qz_mean, logstd=qz_logstd, group_ndims=2)
        qz_samples = posterior_dist.sample(self.cfg['n_samples'])

        # Get mean_x and logstd_x
        hidden, static_front_shape, front_shape = flatten_to_ndims(qz_samples, 3)
        px_mean, px_logstd = self.decoder(hidden, training=False)
        px_mean = unflatten_from_ndims(px_mean, static_front_shape, front_shape)
        px_logstd = unflatten_from_ndims(px_logstd, static_front_shape, front_shape)

        # Create distribution of x
        px_dist = Normal(mean=px_mean, logstd=px_logstd, group_ndims=2)

        px_log_prob = px_dist.log_prob(batch_inputs, use_group_ndims=False)

        # Calculate average over sampling dimension
        px_mean = tf.reduce_mean(px_mean, axis=0)
        px_log_prob = tf.reduce_mean(px_log_prob, axis=0)

        return px_mean, px_log_prob

    def mcmc_reconstruct(self, inputs,
                         n_mc_chain=10,
                         mcmc_iter=10,
                         get_last_obser=True):
        px_means = []
        px_log_probs = []

        for batch_inputs in tqdm(inputs):
            mask = np.zeros(batch_inputs.shape)
            mask[:, -1, :] = 1  # mask all dims of the last point in x
            if n_mc_chain > 1:
                expand_inputs = tf.expand_dims(batch_inputs, 1)
                tiled_inputs = tf.tile(expand_inputs, [1, n_mc_chain, 1, 1])
                flatten_inputs, static_front_shape, front_shape = \
                    flatten_to_ndims(tiled_inputs, 3)
                expand_mask = tf.expand_dims(mask, 1)
                tiled_mask = tf.tile(expand_mask, [1, n_mc_chain, 1, 1])
                flatten_mask, _, _ = flatten_to_ndims(tiled_mask, 3)

                flatten_x_recons = None
                for i in range(mcmc_iter):
                    if flatten_x_recons is None:
                        flatten_x_recons = self._sampling_from_x(flatten_inputs)
                    else:
                        flatten_x_recons = self._sampling_from_x(flatten_x_recons)
                    flatten_x_recons = tf.where(tf.cast(flatten_mask, dtype=tf.bool),
                                                flatten_x_recons, flatten_inputs)

                x_mcmc = unflatten_from_ndims(flatten_x_recons,
                                              static_front_shape,
                                              front_shape)
                x_mcmc = tf.reduce_mean(x_mcmc, axis=1)
            else:
                x_mcmc = self._sampling_from_x(batch_inputs)
                x_mcmc = tf.where(tf.cast(mask, dtype=tf.bool),
                                  x_mcmc, batch_inputs)

            px_mean, px_log_prob = \
                self._reconstruct_x_with_mcmc_recons(batch_inputs, x_mcmc)

            if get_last_obser:
                px_means.append(px_mean.numpy()[:, -1, :])
                px_log_probs.append(px_log_prob.numpy()[:, -1, :])
            else:
                px_means.append(px_mean.numpy())
                px_log_probs.append(px_log_prob.numpy())

        px_means = np.concatenate(px_means, axis=0)
        px_log_probs = np.concatenate(px_log_probs, axis=0)

        return px_means, px_log_probs

    def compile(self, optimizer, **kwargs):
        super(MSTVAEModel, self).compile(**kwargs)
        self.optimizer = optimizer

    def calculate_anomaly_scores(self, inputs, get_last_obser=True, batch_size=50):
        if 'BatchDataset' not in str(type(inputs)):
            inputs = Dataset.from_tensor_slices(inputs).batch(batch_size)

        px_means, px_log_probs = self.mcmc_reconstruct(inputs,
                                                             self.cfg['n_mc_chain'],
                                                             self.cfg['mcmc_iter'],
                                                             get_last_obser)
        anomaly_scores = np.sum(px_log_probs, axis=-1)
        return px_means, anomaly_scores


class AverageLossCallback(Callback):
    def on_test_batch_end(self, epoch, logs=None):
        if logs is not None:
            for metric, value in logs.items():
                if metric not in self.epoch_losses_eval:
                    self.epoch_losses_eval[metric] = [value]
                else:
                    self.epoch_losses_eval[metric].append(value)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_losses_eval = {}

    def on_test_end(self, logs=None):
        avg_losses_eval = dict()
        if self.epoch_losses_eval:
            for metric, values in self.epoch_losses_eval.items():
                avg_losses_eval[metric] = np.mean(values)
            self.model.average_epoch_eval = avg_losses_eval