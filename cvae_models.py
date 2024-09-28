# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 Daniyar Mazitov <daniyarttt@gmail.com>
#  This file is part of RC_CVAE
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
import math
import operator
import collections
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Input, Lambda, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model, load_model

from hyperspherical_vae.distributions import HypersphericalUniform, VonMisesFisher

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

tf.compat.v1.experimental.output_all_intermediates(True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
        
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed
    
    
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed
    
    
def FTSwish(threshold=-0.2):
    def _FTSwish(x):
        return K.relu(x) * K.sigmoid(x) + threshold
    return Lambda(_FTSwish)
    
    
def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(cdf_ytrue - cdf_ypred), axis=-1)+ K.epsilon())
    return K.mean(samplewise_emd)
    
    
def mean_score(scores):
    si = np.arange(1, len(scores)+1, 1)
    mean = np.sum(scores * si)
    return mean - 1
    
    
def std_score(scores):
    si = np.arange(1, len(scores)+1, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - (mean+1)) ** 2) * scores))
    return std
    

def sampling():
    def _sampling(args):
        z_mean, z_log_var = args
        return tfp.distributions.Normal(z_mean, K.exp(0.5 * z_log_var)).sample()
    return Lambda(_sampling)
    
    
def h_sampling():
    def _sampling(args):
        z_mean, z_log_var = args
        return VonMisesFisher(z_mean, z_log_var+1).sample()
    return Lambda(_sampling)
    
    
def S_loss():
    def _S_loss(args):
        z_mean, z_log_var = args
        latent_dim = K.int_shape(z_mean)[-1]
        return VonMisesFisher(z_mean, z_log_var+1).kl_divergence(HypersphericalUniform(latent_dim-1))
    return Lambda(_S_loss)


def KMeans(z, n_clusters, tol=1e-4):
    def pairwise_distance(data1, data2):
        A = K.expand_dims(data1, axis=1)
        B = K.expand_dims(data2, axis=0)
        dis = K.pow(A-B, 2)
        return K.sum(dis, axis=-1)

    def random_choice(inputs, n_samples):
        p = tf.random.uniform([K.shape(inputs)[0]], 0, 1)
        _, ind = tf.math.top_k(p, n_samples)
        return tf.gather(inputs, ind)

    def bucket_mean(data, bucket_ids, num_buckets):
        total = tf.math.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.math.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    def _iter(initial_state, center_shift):
        dis = pairwise_distance(z, initial_state)
        choice_cluster = K.argmin(dis, axis=1)
        
        means = bucket_mean(z, choice_cluster, n_clusters)
        
        return means, K.sum(K.sqrt(K.sum(K.pow(means-initial_state, 2), axis=1))) #new initial_state, center_shift
    
    def cond(initial_state, center_shift):
        return tf.math.greater_equal(K.pow(center_shift, 2), tol)
    
    initial_state  = random_choice(z, n_clusters)
    center_shift = 100.
    
    initial_state, center_shift = tf.while_loop(cond, _iter, [initial_state, center_shift])

    return initial_state
    
    
def im_metric(z, centers):
    batch_size = K.shape(z)[0]
    n = K.int_shape(centers)[0] # number of centers
    z_dim = K.int_shape(z)[1]
    
    d = tf.tile(K.expand_dims(z, axis=1), [1, n, 1]) - tf.tile(K.expand_dims(centers, axis=0), [batch_size, 1, 1])
    
    norm = K.transpose(K.squeeze(tf.norm(d, axis=-1, keepdims=True), axis=-1))
    measure = K.min(norm, axis=0)
    
    Cbase = 2 * z_dim
    stat = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        stat += C / (C + K.pow(measure, 2))
        
    return stat
    
    
class PlanarNormalizingFlow(Layer):
    def __init__(self, in_features, band=0.01, **kwargs):
        self.in_features = in_features
        self.band = band
        super(PlanarNormalizingFlow, self).__init__(**kwargs)

    def build(self, input_shape):
        self.u = self.add_weight(name='u', 
                                 shape=(self.in_features,),
                                 initializer='random_normal',
                                 trainable=True)
        self.w = self.add_weight(name='w', 
                                 shape=(self.in_features,),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='b', 
                                 shape=(1,),
                                 initializer='ones',
                                 trainable=True)
        self.built = True

    def call(self, z):
        centers = KMeans(z, 3) #10
        beta = im_metric(z, centers)
        
        uw = K.sum(self.u * self.w)
        muw = -1 + K.softplus(uw)
        uhat = self.u + (muw - uw) * K.transpose(self.w) / K.sum(K.pow(self.w, 2))
        
        zwb = K.squeeze(K.dot(z, K.expand_dims(self.w, -1)), axis=-1) + self.b
        
        f_z = z + (K.reshape(uhat, [1, -1]) * K.reshape(K.tanh(zwb), [-1, 1]))
        psi = (1 - K.reshape(K.pow(K.tanh(zwb), 2), [-1, 1])) * K.reshape(self.w, [1, -1])
        psi_u = K.squeeze(K.dot(psi, K.expand_dims(uhat, -1)), axis=-1)
        
        logdet_jacobian = K.log(K.abs(1 + psi_u) + K.epsilon())
        
        penalty = self.band * K.log(K.abs(beta) + K.epsilon())
                
        return [f_z, logdet_jacobian, penalty]


def MMD_loss(args):
    z_mean, z_log_var = args
    
    p_z = tfp.distributions.Normal(tf.zeros_like(z_mean), K.exp(0.5 * tf.zeros_like(z_log_var)))
    q_z = tfp.distributions.Normal(z_mean, K.exp(0.5 * z_log_var))
    
    sample_qz = q_z.sample()
    batch_size = K.shape(sample_qz)[0]
    z_dim = K.int_shape(sample_qz)[-1]
    sample_pz = p_z.sample()
    C_base = 2. * z_dim

    norms_pz = K.sum(K.square(sample_pz), axis=1, keepdims=True)
    dotprods_pz = K.dot(sample_pz, K.transpose(sample_pz))
    distances_pz = norms_pz + K.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = K.sum(K.square(sample_qz), axis=1, keepdims=True)
    dotprods_qz = K.dot(sample_qz, K.transpose(sample_qz))
    distances_qz = norms_qz + K.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = K.dot(sample_qz, K.transpose(sample_pz))
    distances = norms_qz + K.transpose(norms_pz) - 2. * dotprods
    
    batch_size_f = tf.cast(batch_size, tf.float32)

    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = C_base * scale
        res1 = C / (C + distances_qz)
        res1 += C / (C + distances_pz)
        res1 = tf.multiply(res1, 1. - tf.eye(batch_size))
        res1 = K.sum(res1) / (batch_size_f * batch_size_f - batch_size_f)
        res2 = C / (C + distances)
        res2 = K.sum(res2) * 2. / (batch_size_f * batch_size_f)
        stat += res1 - res2

    return stat


def KL_loss(args):
    z_mean, z_log_var, z, z0 = args
    batch_size = tf.cast(K.shape(z)[0], dtype=tf.float32)
    
    p_z = tfp.distributions.Normal(tf.zeros_like(z_mean), K.exp(0.5 * tf.zeros_like(z_log_var)))
    q_z = tfp.distributions.Normal(z_mean, K.exp(0.5 * z_log_var))
    
    loss = K.mean(K.sum(q_z.log_prob(z0), axis=-1) - K.sum(p_z.log_prob(z), axis=-1))
    
    return loss


def Flow_loss(args):
    sum_log_jacobian, sum_penalty, z = args
    batch_size = tf.cast(K.shape(z)[0], dtype=tf.float32)
    
    loss = -(K.sum(sum_log_jacobian) - K.sum(sum_penalty)) / batch_size
    
    return loss
    
    
def get_emb_model(in_dim=1000, out_dim=433):
    model = Sequential()
    model.add(Dense(2000, activation='relu', kernel_initializer='he_normal', input_shape=(in_dim,)))
    model.add(Dense(out_dim, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
    
    return model
    
    
def get_gauss_model(trained_emb_model_path, in_dim, out_dim, latent_dim=32, adds_dim=129, catalyst_dim=297, pressure_dim=4, temperature_dim=3):
    emb_model = load_model(trained_emb_model_path)
    base_cond = Model(inputs=emb_model.inputs, outputs=emb_model.layers[0].output)
    base_cond.trainable=False
    #--------------------------------------------------------------------------
    
    In1 = Input(shape=(in_dim,))
    In2 = Input(shape=(out_dim,))
    
    x = Dense(512, kernel_initializer='he_normal')(In1)
    x = BatchNormalization(axis=-1)(x)
    x = FTSwish()(x)

    cond = Model(inputs=In1, outputs=x)
    #--------------------------------------------------------------------------
    
    x1 = cond(In1)

    x2 = Dense(64, kernel_initializer='he_normal')(In2)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = FTSwish()(x2)

    x = Concatenate()([x1, x2])

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    z = sampling()([z_mean, z_log_var])
    z0 = z

    encoder = Model(inputs=[In1, In2], outputs=z)
    #--------------------------------------------------------------------------
    
    In_latent = Input(shape=(latent_dim,))
    
    x1 = cond(In1)
    x1_base = base_cond(In1)

    x = Concatenate()([x1, x1_base, In_latent])

    x = Dropout(0.5)(x)

    xt = Dense(temperature_dim, activation='softmax', kernel_initializer='glorot_normal')(x)
    xp = Dense(pressure_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    adds = Dense(adds_dim, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    catalyst = Dense(catalyst_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    decoder = Model(inputs=[In1, In_latent], outputs=[xt, xp, adds, catalyst])
    #--------------------------------------------------------------------------

    e = encoder([In1, In2])
    d = decoder([In1, e])

    model = Model(inputs=[In1, In2], outputs=d)
    #--------------------------------------------------------------------------
    
    kl_loss = KL_loss([z_mean, z_log_var, z, z0])
    model.add_loss(0.001*K.mean(kl_loss))
    
    model.compile(optimizer=Adam(lr=0.001), 
                  loss=[earth_mover_loss, earth_mover_loss, 
                        binary_focal_loss(),
                        categorical_focal_loss()])
                        
    return model, decoder
    
    
def get_rnf_gauss_model(trained_emb_model_path, in_dim, out_dim, latent_dim=32, n_flows=3, adds_dim=129, catalyst_dim=297, pressure_dim=4, temperature_dim=3):
    emb_model = load_model(trained_emb_model_path)
    base_cond = Model(inputs=emb_model.inputs, outputs=emb_model.layers[0].output)
    base_cond.trainable=False
    #--------------------------------------------------------------------------
    
    In1 = Input(shape=(in_dim,))
    In2 = Input(shape=(out_dim,))
    
    x = Dense(512, kernel_initializer='he_normal')(In1)
    x = BatchNormalization(axis=-1)(x)
    x = FTSwish()(x)

    cond = Model(inputs=In1, outputs=x)
    #--------------------------------------------------------------------------
    
    x1 = cond(In1)

    x2 = Dense(64, kernel_initializer='he_normal')(In2)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = FTSwish()(x2)

    x = Concatenate()([x1, x2])

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    z = sampling()([z_mean, z_log_var])
    z0 = z

    flows = []
    for i in range(n_flows):
        flows.append(PlanarNormalizingFlow(latent_dim))

    log_det_jacobian = []
    penalty = []

    for flow in flows:
        z, j, d = flow(z)
        log_det_jacobian.append(j)
        penalty.append(d)
        
    sum_log_jacobian = K.sum(log_det_jacobian, axis=0)
    sum_penalty = K.sum(penalty, axis=0)

    encoder = Model(inputs=[In1, In2], outputs=z)
    #--------------------------------------------------------------------------
    
    In_latent = Input(shape=(latent_dim,))
    
    x1 = cond(In1)
    x1_base = base_cond(In1)

    x = Concatenate()([x1, x1_base, In_latent])

    x = Dropout(0.5)(x)

    xt = Dense(temperature_dim, activation='softmax', kernel_initializer='glorot_normal')(x)
    xp = Dense(pressure_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    adds = Dense(adds_dim, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    catalyst = Dense(catalyst_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    decoder = Model(inputs=[In1, In_latent], outputs=[xt, xp, adds, catalyst])
    #--------------------------------------------------------------------------

    C = Input(shape=(1,))

    e = encoder([In1, In2])
    d = decoder([In1, e])

    model = Model(inputs=[In1, In2, C], outputs=d)
    #--------------------------------------------------------------------------
    
    mmd_w = 10
    
    flow_loss = Flow_loss([sum_log_jacobian, sum_penalty, z])
    model.add_loss(K.mean(flow_loss))
    
    kl_loss = KL_loss([z_mean, z_log_var, z, z0])
    model.add_loss(K.mean(C*kl_loss))
    
    mmd_loss = MMD_loss([z_mean, z_log_var])
    model.add_loss(K.mean((mmd_w - C)*mmd_loss))
    
    model.compile(optimizer=Adam(lr=0.001), 
                  loss=[earth_mover_loss, earth_mover_loss, 
                        binary_focal_loss(),
                        categorical_focal_loss()])
                        
    return model, decoder


def fit_rnf_gauss_model(model, epochs, batch_size, X_train, y_train, t_dim, p_dim, add_dim):
    kl_w = 0.8

    for i in range(1, epochs+1):
        print('-------------------------------------------------------------')
        print(str(i))
        print('-------------------------------------------------------------')
        
        c = kl_w/epochs*i
        model.fit([X_train, y_train, np.zeros(len(X_train))+c], 
              [y_train[:, :t_dim].astype('float32'), y_train[:, t_dim:p_dim+t_dim].astype('float32'), 
               y_train[:, p_dim+t_dim:add_dim+p_dim+t_dim], 
               y_train[:, add_dim+p_dim+t_dim:]],
              batch_size=batch_size, epochs=1)
        
        lr = 0.00025 + (0.001 - 0.00025) * (1 + math.cos(math.pi * i / epochs)) / 2
        K.set_value(model.optimizer.lr, lr)
        
        
def get_h_model(trained_emb_model_path, in_dim, out_dim, latent_dim=32, adds_dim=129, catalyst_dim=297, pressure_dim=4, temperature_dim=3):
    emb_model = load_model(trained_emb_model_path)
    base_cond = Model(inputs=emb_model.inputs, outputs=emb_model.layers[0].output)
    base_cond.trainable=False
    #--------------------------------------------------------------------------
    
    In1 = Input(shape=(in_dim,))
    In2 = Input(shape=(out_dim,))
    
    x = Dense(512, kernel_initializer='he_normal')(In1)
    x = BatchNormalization(axis=-1)(x)
    x = FTSwish()(x)

    cond = Model(inputs=In1, outputs=x)
    #--------------------------------------------------------------------------
    
    x1 = cond(In1)

    x2 = Dense(64, kernel_initializer='he_normal')(In2)
    x2 = BatchNormalization(axis=-1)(x2)
    x2 = FTSwish()(x2)

    x = Concatenate()([x1, x2])

    z_mean = Dense(latent_dim, activation=lambda x: tf.nn.l2_normalize(x, axis=-1))(x)
    z_log_var = Dense(1, activation='softplus')(x)

    z = h_sampling()([z_mean, z_log_var])

    encoder = Model(inputs=[In1, In2], outputs=z)
    #--------------------------------------------------------------------------
    
    In_latent = Input(shape=(latent_dim,))
    
    x1 = cond(In1)
    x1_base = base_cond(In1)

    x = Concatenate()([x1, x1_base, In_latent])

    x = Dropout(0.5)(x)

    xt = Dense(temperature_dim, activation='softmax', kernel_initializer='glorot_normal')(x)
    xp = Dense(pressure_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    adds = Dense(adds_dim, activation='sigmoid', kernel_initializer='glorot_normal')(x)
    catalyst = Dense(catalyst_dim, activation='softmax', kernel_initializer='glorot_normal')(x)

    decoder = Model(inputs=[In1, In_latent], outputs=[xt, xp, adds, catalyst])
    #--------------------------------------------------------------------------

    e = encoder([In1, In2])
    d = decoder([In1, e])

    model = Model(inputs=[In1, In2], outputs=d)
    #--------------------------------------------------------------------------
    
    kl_loss = S_loss()([z_mean, z_log_var])
    model.add_loss(0.001*K.mean(kl_loss))
    
    model.compile(optimizer=Adam(lr=0.001), 
                  loss=[earth_mover_loss, earth_mover_loss, 
                        binary_focal_loss(),
                        categorical_focal_loss()])
                        
    return model, decoder
    
    
def generate_gauss(X_test, decoder, latent_dim=32, s=5000, adds_dim=129, catalyst_dim=297, pressure_dim=4, temperature_dim=3):
    hpu = tfp.distributions.Normal([0]*latent_dim, [1]*latent_dim)

    ans = np.concatenate(decoder.predict([X_test, K.eval(hpu._sample_n(len(X_test)))], batch_size=1024), axis=-1).astype('float16').reshape(-1, adds_dim+catalyst_dim+pressure_dim+temperature_dim)
    ans = np.concatenate([(np.array([mean_score(n[:temperature_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                          (np.array([mean_score(n[temperature_dim:temperature_dim+pressure_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                          (ans[:, temperature_dim+pressure_dim:temperature_dim+pressure_dim+adds_dim]+0.5).astype('int16'), 
                          np.argmax(ans[:, temperature_dim+pressure_dim+adds_dim:], axis=1).reshape(-1, 1)], axis=1)

    d = []
    for item in ans:
        tmp = {}
        tmp[tuple(item.tolist())] = 1
        d.append(tmp)

    for i in range(s):
        print('\r%d/%d' % (i+1, s), end='')
        
        ans = np.concatenate(decoder.predict([X_test, K.eval(hpu._sample_n(len(X_test)))], batch_size=1024), axis=-1).astype('float16').reshape(-1, adds_dim+catalyst_dim+pressure_dim+temperature_dim)
        ans = np.concatenate([(np.array([mean_score(n[:temperature_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                              (np.array([mean_score(n[temperature_dim:temperature_dim+pressure_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                              (ans[:, temperature_dim+pressure_dim:temperature_dim+pressure_dim+adds_dim]+0.5).astype('int16'), 
                              np.argmax(ans[:, temperature_dim+pressure_dim+adds_dim:], axis=1).reshape(-1, 1)], axis=1)
        
        for i, item in enumerate(ans):
            tmp = tuple(item.tolist())
            if tmp in d[i]:
                d[i][tmp] += 1
            else: # new
                d[i][tmp] = 1
                
        del ans
        
    top_ans = []
    for i in range(len(d)):
        od = collections.OrderedDict(sorted(d[i].items(), key=operator.itemgetter(1), reverse=True))
        top_n = np.array(list(od.items()))[:, 0][:100]
        top_ans.append(top_n)
        
    for i in range(len(top_ans)):
        while len(top_ans[i]) < 100:
            tmp = list(top_ans[i])
            tmp.append(top_ans[i][-1])
            top_ans[i] = tmp
            
    top_ans = np.array(top_ans)
    
    return top_ans
    
    
def generate_h(X_test, decoder, latent_dim=32, s=5000, adds_dim=129, catalyst_dim=297, pressure_dim=4, temperature_dim=3):
    hpu = HypersphericalUniform(latent_dim-1)

    ans = np.concatenate(decoder.predict([X_test, K.eval(hpu._sample_n(len(X_test)))], batch_size=1024), axis=-1).astype('float16').reshape(-1, adds_dim+catalyst_dim+pressure_dim+temperature_dim)
    ans = np.concatenate([(np.array([mean_score(n[:temperature_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                          (np.array([mean_score(n[temperature_dim:temperature_dim+pressure_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                          (ans[:, temperature_dim+pressure_dim:temperature_dim+pressure_dim+adds_dim]+0.5).astype('int16'), 
                          np.argmax(ans[:, temperature_dim+pressure_dim+adds_dim:], axis=1).reshape(-1, 1)], axis=1)

    d = []
    for item in ans:
        tmp = {}
        tmp[tuple(item.tolist())] = 1
        d.append(tmp)

    for i in range(s):
        print('\r%d/%d' % (i+1, s), end='')
        
        ans = np.concatenate(decoder.predict([X_test, K.eval(hpu._sample_n(len(X_test)))], batch_size=1024), axis=-1).astype('float16').reshape(-1, adds_dim+catalyst_dim+pressure_dim+temperature_dim)
        ans = np.concatenate([(np.array([mean_score(n[:temperature_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                              (np.array([mean_score(n[temperature_dim:temperature_dim+pressure_dim]) for n in ans])+0.5).astype('int16').reshape(-1, 1), 
                              (ans[:, temperature_dim+pressure_dim:temperature_dim+pressure_dim+adds_dim]+0.5).astype('int16'), 
                              np.argmax(ans[:, temperature_dim+pressure_dim+adds_dim:], axis=1).reshape(-1, 1)], axis=1)
        
        for i, item in enumerate(ans):
            tmp = tuple(item.tolist())
            if tmp in d[i]:
                d[i][tmp] += 1
            else: # new
                d[i][tmp] = 1
                
        del ans
        
    top_ans = []
    for i in range(len(d)):
        od = collections.OrderedDict(sorted(d[i].items(), key=operator.itemgetter(1), reverse=True))
        top_n = np.array(list(od.items()))[:, 0][:100]
        top_ans.append(top_n)
        
    for i in range(len(top_ans)):
        while len(top_ans[i]) < 100:
            tmp = list(top_ans[i])
            tmp.append(top_ans[i][-1])
            top_ans[i] = tmp
            
    top_ans = np.array(top_ans)
    
    return top_ans
    
