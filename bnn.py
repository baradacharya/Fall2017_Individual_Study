import numpy as np
import tensorflow as tf

### Base neural network                                                                                                                  
def init_mlp(layer_sizes, std=.001):
    params = {'mu':[], 'log_sigma':[], 'b':[]}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['mu'].append(tf.Variable(tf.random_normal([n_in, n_out], stddev=std)))
        params['log_sigma'].append(tf.Variable(tf.random_normal([n_in, n_out], stddev=std*.1) - 5.))
        params['b'].append(tf.Variable(tf.zeros([n_out,])))
    return params


def mlp_w_mc(X, params):
    h = [X]
    for idx in xrange(len(params['mu'][:-1])):
        a = tf.matmul( h[-1], params['mu'][idx] + tf.mul(tf.exp(params['log_sigma'][idx]), tf.random_normal(tf.shape(params['mu'][idx]))) ) + params['b'][idx]
        h.append( tf.nn.relu(a) )
    idx = -1
    return tf.matmul( h[-1], params['mu'][idx] + tf.mul(tf.exp(params['log_sigma'][idx]), tf.random_normal(tf.shape(params['mu'][idx]))) ) + params['b'][idx]


def mlp_deterministic(X, params):
    h = [X]
    for idx in xrange(len(params['mu'][:-1])):
        a = tf.matmul( h[-1], params['mu'][idx] ) + params['b'][idx]
        h.append( tf.nn.relu(a) )
    idx = -1
    return tf.matmul( h[-1], params['mu'][idx] ) + params['b'][idx]


def gauss2gauss_KLD(mu_post, sigma_post, mu_prior=0., sigma_prior=.001):
    d = (mu_post - mu_prior)
    d = tf.mul(d,d)
    return -.5 * tf.reduce_mean(-tf.div(d + tf.mul(sigma_post,sigma_post),sigma_prior*sigma_prior) \
                                    - 2*tf.log(sigma_prior) + 2.*tf.log(sigma_post) + 1.)

### Bayesian Neural Network
class BNN(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.Y = tf.placeholder("float", [None, hyperParams['output_d']])

        self.params = init_mlp([hyperParams['input_d'], hyperParams['hidden_d']]+\
                                        [hyperParams['hidden_d'], hyperParams['hidden_d']]*(hyperParams["n_hidden_layers"]-1)+\
                                        [hyperParams['hidden_d'],hyperParams['output_d']])

        self.y_hat_linear = self.f_prop()
        self.loss_fn = self.get_elbo(hyperParams['prior'])

        self.y_hat = tf.nn.softmax(mlp_deterministic(self.X, self.params))


    def f_prop(self):
        return  mlp_w_mc(self.X, self.params)


    def get_elbo(self, prior):
        nll = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat_linear, self.Y))
        kld = gauss2gauss_KLD(self.params['mu'][0], tf.exp(self.params['log_sigma'][0]), mu_prior=prior['mu'], sigma_prior=prior['sigma'])
        for idx in xrange(len(self.params['mu'])-1):
            kld += gauss2gauss_KLD(self.params['mu'][idx+1], tf.exp(self.params['log_sigma'][idx+1]), mu_prior=prior['mu'], sigma_prior=prior['sigma'])
        return nll + tf.reduce_mean(kld) 
