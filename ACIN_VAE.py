import numpy as np
import tensorflow as tf
import h5py
from sampling_utils import *
from tensorflow.examples.tutorials.mnist import input_data
try:
    import PIL.Image as Image
except ImportError:
    import Image

# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 128, "batch size.")
flags.DEFINE_integer("nEpochs", 2, "number of epochs to train.")
flags.DEFINE_float("adamLr", 1e-4, "AdaM learning rate.")
flags.DEFINE_integer("hidden_size", 200, "number of hidden units in en/decoder.")
flags.DEFINE_integer("latent_size", 50, "dimensionality of latent variables.")
flags.DEFINE_integer("K", 5, "Maximun number of loops in RNN.")
inArgs = flags.FLAGS

def init_mlp(layer_sizes, std=.01, bias_init=0.):
    params = {'w': [], 'b': []}
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        params['w'].append(tf.Variable(tf.random_normal([n_in, n_out], stddev=std)))
        params['b'].append(tf.Variable(tf.multiply(bias_init, tf.ones([n_out, ]))))
    return params


def mlp(X, params):
    h = [X]
    for w, b in zip(params['w'][:-1], params['b'][:-1]):
        h.append(tf.nn.relu(tf.matmul(h[-1], w) + b))
    return tf.matmul(h[-1], params['w'][-1]) + params['b'][-1]

def compute_nll(x, x_recon_linear):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits= x_recon_linear, labels= x), reduction_indices=1,
                         keep_dims=True)

def log_normal_pdf(x, mu, sigma):
    d = mu - x
    d2 = tf.multiply(-1., tf.multiply(d,d))
    s2 = tf.multiply(2., tf.multiply(sigma,sigma))
    return tf.reduce_sum(tf.div(d2,s2) - tf.log(tf.multiply(sigma, 2.506628)))

#Adaptive Computation Inference Network
class ACIN_VAE(object):
    def __init__(self, hyperParams):

        self.X = tf.placeholder("float", [None, hyperParams['input_d']])
        self.K = hyperParams['K']

        self.encoder_params = self.init_encoder(hyperParams)
        self.decoder_params = self.init_decoder(hyperParams)

        self.x_recons_linear = self.f_prop(hyperParams)

        self.elbo_obj = self.get_ELBO()

        self.final_pis = tf.concat([tf.expand_dims(t, 2) for t in self.pis], 2)
        self.final_mus = tf.concat([tf.expand_dims(t, 2) for t in self.mus], 2)
        self.final_sigmas = tf.concat([tf.expand_dims(t, 2) for t in self.sigmas], 2)

    def init_encoder(self, hyperParams):
        return {'h': {'W': tf.Variable(tf.random_normal([hyperParams['input_d'], hyperParams['rnn_hidden_d']], stddev=.00001)),
                      'U': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['rnn_hidden_d']], stddev=.00001)),
                      'b': tf.Variable(tf.zeros([hyperParams['rnn_hidden_d'], ]))
                     },
                'mu': {'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['z_space_d']], stddev=.00001)),
                       'b': tf.Variable(tf.zeros([hyperParams['z_space_d'], ]))
                      },
                'sigma':{'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['z_space_d']], stddev=.00001)),
                            'b': tf.Variable(tf.zeros([hyperParams['z_space_d'], ]))
                        },
                'pi': {'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], 1], stddev=.00001) - .2),
                         'b': tf.Variable(tf.zeros([1, ]))
                      }
                  }

    def init_decoder(self, hyperParams):
        return init_mlp([hyperParams['z_space_d'], hyperParams['hidden_d'], hyperParams['hidden_d'], hyperParams['input_d']])


    def prop_RNN(self,hidden_state):
        # X*hw + hs *hu + hb
        hidden_state = tf.nn.relu(
            tf.matmul(self.X, self.encoder_params['h']['W']) + tf.matmul(hidden_state, self.encoder_params['h']['U']) \
            + self.encoder_params['h']['b'])

        return hidden_state

    def f_prop(self,hyperParams):
        self.pis = []
        self.mus = []
        self.sigmas = []
        self.z = []

        x_recon_linear = []
        remaining_stick = 1.

        hidden_states = [tf.zeros([tf.shape(self.X)[0], hyperParams['rnn_hidden_d']])]

        for loop_idx in range(self.K):
            hidden_states.append(self.prop_RNN(hidden_states[-1]))

        hidden_states = hidden_states[1:]


        stick_eps = .01
        for idx in range(self.K - 1):
            # compute component params
            self.mus.append(tf.matmul(hidden_states[idx], self.encoder_params['mu']['W']) + self.encoder_params['mu']['b'])
            self.sigmas.append(tf.nn.softplus(tf.matmul(hidden_states[idx], self.encoder_params['sigma']['W']) + self.encoder_params['sigma']['b']))
            # compute component weights
            gamma = tf.nn.sigmoid(tf.matmul(hidden_states[idx], self.encoder_params['pi']['W']) + self.encoder_params['pi']['b'])

            length_check = tf.reduce_max((1. - gamma) * remaining_stick)
            gamma = tf.cond(length_check < stick_eps, lambda: 0. * gamma + 1., lambda: gamma)

            self.pis.append(gamma * remaining_stick)
            remaining_stick = (1. - gamma) * remaining_stick

            self.z.append(self.mus[-1] + tf.multiply(self.sigmas[-1], tf.random_normal(tf.shape(self.sigmas[-1]))))  # sampling of z from mean and variance
            x_recon_linear.append(mlp(self.z[-1], self.decoder_params))  # z (decoder)-> x_reconstruction

        # perform last iteration with pi set to remaining stick
        self.mus.append(tf.matmul(hidden_states[-1], self.encoder_params['mu']['W']) + self.encoder_params['mu']['b'])
        self.sigmas.append(tf.nn.softplus(tf.matmul(hidden_states[-1], self.encoder_params['sigma']['W']) + self.encoder_params['sigma']['b']))
        self.pis.append(remaining_stick)
        self.z.append(self.mus[-1] + tf.multiply(self.sigmas[-1], tf.random_normal(tf.shape(self.sigmas[-1]))))
        x_recon_linear.append(mlp(self.z[-1], self.decoder_params))  # z (decoder)-> x_reconstruction

        return x_recon_linear

    def get_ELBO(self):
        self.regularization_weight = tf.placeholder(tf.float32, shape=(), name="regWeight")

        # data term: \sum_k \pi_k E[log p(x,z)]
        nll = 0
        for k in range(self.K):
            nll -=  self.pis[k] * compute_nll(self.X, self.x_recons_linear[k])

        # entropy lower bound term : -\sum_k pi_k log \sum_j pi_j N(mu_k; mu_j, sigma_k**2 + sigma_j**2)
        ent_lb_term = 0.
        for k in range(self.K):
            temp_val = 0.
            for j in range(self.K):
                temp_val += self.pis[j] * tf.exp(log_normal_pdf(self.mus[k], self.mus[j], tf.sqrt(self.sigmas[k] ** 2 + self.sigmas[j] ** 2)))
            ent_lb_term += -self.pis[k] * tf.log(temp_val + .0001)  # why negative sign

        # entropy of mixture weights
        ent_mix_weights = 0.
        for k in range(self.K):
            ent_mix_weights += -self.pis[k] * tf.log(self.pis[k] + .001)

        #Expectation of Prior  \sum_k \pi_k E(q_k)[log(p(z))]
        expectation_prior = 0.
        for k in range(self.K):
            expectation_prior += -self.pis[k] * (self.sigmas[k] ** 2 + self.mus[k] ** 2 + 0.79817986835) #log(2pie)

        # final objective
        elbo = tf.reduce_mean(nll + ent_lb_term - self.regularization_weight * ent_mix_weights)

        return elbo

    def get_log_margLL(self):

        nll = 0
        for k in range(self.K):
            nll -= self.pis[k] * compute_nll(self.X, self.x_recons_linear[k])

        # calc prior prior':{'mu':0., 'sigma':1.}
        log_prior = 0
        for k in range(self.K):
            log_prior += self.pis[k] * log_normal_pdf(self.z[k], 0., 1.)

        # calc post
        log_post = 0
        for k in range(self.K):
            log_post += self.pis[k] * log_normal_pdf(self.z[k], self.mus[k], self.sigmas[k])

        return nll + log_prior - log_post

    def get_samples(self, nImages):
        samples_from_each_component = []
        for k in xrange(self.K):
            z = 0. + tf.multiply(1.,tf.random_normal((nImages, tf.shape(self.decoder_params['w'][0])[0])))
            samples_from_each_component.append(tf.sigmoid(mlp(z, self.decoder_params)))
        return samples_from_each_component




def trainVAE(data, vae_hyperParams, hyperParams):
    N_train, d = data['train'].shape
    N_valid, d = data['valid'].shape
    nTrainBatches = N_train / hyperParams['batchSize']
    nValidBatches = N_valid / hyperParams['batchSize']
    vae_hyperParams['batchSize'] = hyperParams['batchSize']

    # init Mix Density VAE
    model = ACIN_VAE(vae_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(hyperParams['adamLr']).minimize(-model.elbo_obj)

    reg_weights = np.linspace(0., 1., int(hyperParams['nEpochs'] * .75)).astype('float32').tolist()
    with tf.Session() as s:
        s.run(tf.initialize_all_variables())

        # for early stopping
        best_epoch = 0

        for epoch_idx in xrange(hyperParams['nEpochs']):
            # perform update
            if epoch_idx < len(reg_weights):
                w = reg_weights[epoch_idx]
            else:
                w = reg_weights[-1]

            # training
            train_elbo = 0.
            for batch_idx in xrange(nTrainBatches):
                x = data['train'][batch_idx * hyperParams['batchSize']:(batch_idx + 1) * hyperParams['batchSize'],:]
                _, elbo_train = s.run([optimizer, model.elbo_obj], {model.X: x, model.regularization_weight: w})
                train_elbo += elbo_train

            # validation
            valid_elbo = 0.
            for batch_idx in xrange(nValidBatches):
                x = data['valid'][batch_idx * hyperParams['batchSize']:(batch_idx + 1) * hyperParams['batchSize'], :]
                valid_elbo += s.run(model.elbo_obj, {model.X: x,model.regularization_weight: w})

            # check for ELBO improvement
            star_printer = ""
            train_elbo /= nTrainBatches
            valid_elbo /= nValidBatches

            # log training progress
            logging_str = "Epoch %d.  Train ELBO: %.3f,  Validation ELBO: %.3f %s" % (epoch_idx + 1, train_elbo, valid_elbo, star_printer)
            print logging_str

        x = data['train'][0:hyperParams['batchSize'], :]
        mu, si, pi = s.run([model.final_mus, model.final_sigmas, model.final_pis],feed_dict={model.X: x})
        mu, si, pi = mu[0, :, :], si[0, :, :], pi[0, :, :]
        print "Mu: " + str(mu)
        print "Sigma: " + str(si)
        print "Pis: " + str(pi)

        # evaluate marginal likelihood
        print "Calculating the marginal likelihood..."
        N, d = mnist['test'].shape
        sample_collector = []
        nSamples = 50
        for s_idx in xrange(nSamples):
            samples = s.run(model.get_log_margLL(), {model.X: mnist['test']})
            if not np.isnan(samples.mean()) and not np.isinf(samples.mean()):
                sample_collector.append(samples)

        if len(sample_collector) < 1:
            print "\tMARG LIKELIHOOD CALC: No valid samples were collected!"
            return np.nan

        all_samples = np.hstack(sample_collector)
        m = np.amax(all_samples, axis=1)
        mLL = m + np.log(np.mean(np.exp(all_samples - m[np.newaxis].T), axis=1))

        logging_str = "\nTest Marginal Likelihood: %.3f" % (mLL.mean())
        print logging_str

        nImages = 10
        sample_list = s.run(model.get_samples(nImages))
        for i, samples in enumerate(sample_list):
            image = Image.fromarray(tile_raster_images(X=samples, img_shape=(28, 28),
                                                       tile_shape=(int(np.sqrt(nImages)), int(np.sqrt(nImages))),
                                                       tile_spacing=(1, 1)))
            image.save("./" + "_component" + str(i) + ".png")

def calc_margLikelihood(data, model, vae_hyperParams, nSamples=50):
    N, d = data.shape

    # get op to load the model
    persister = tf.train.Saver()

    with tf.Session() as s:
        # persister.restore(s, param_file_path)

        sample_collector = []
        for s_idx in xrange(nSamples):
            samples = s.run(model.get_log_margLL(N), {model.X: data})
            if not np.isnan(samples.mean()) and not np.isinf(samples.mean()):
                sample_collector.append(samples)

    if len(sample_collector) < 1:
        print "\tMARG LIKELIHOOD CALC: No valid samples were collected!"
        return np.nan

    all_samples = np.hstack(sample_collector)
    m = np.amax(all_samples, axis=1)
    mLL = m + np.log(np.mean(np.exp(all_samples - m[np.newaxis].T), axis=1))
    return mLL.mean()

if __name__ == "__main__":
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # load MNIST
    f = h5py.File('./binarized_mnist.h5')
    mnist = {'train': np.copy(f['train']), 'valid': np.copy(f['valid']), 'test': np.copy(f['test'])}
    np.random.shuffle(mnist['train'][0])

    # set architecture params

    vae_hyperParams = {'input_d':mnist['train'].shape[1], 'rnn_hidden_d': inArgs.hidden_size,'hidden_d': inArgs.hidden_size,
                       'z_space_d': inArgs.latent_size, 'K': inArgs.K}

    # set training hyperparameters
    train_hyperParams = {'adamLr': inArgs.adamLr, 'nEpochs': inArgs.nEpochs, 'batchSize': inArgs.batchSize}

    # train
    print "Training model..."

    trainVAE(mnist, vae_hyperParams, train_hyperParams)
