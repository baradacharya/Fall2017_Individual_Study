import tensorflow as tf


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
    return tf.reduce_sum(tf.div(d2,s2) - tf.log(tf.multiply(sigma, 2.506628)), reduction_indices=1, keep_dims=True)
    #o/p[batchsize,1] #normal pdf for each data points (mini batch)

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
        return {'h': {'W': tf.Variable(tf.random_normal([hyperParams['input_d'], hyperParams['rnn_hidden_d']], stddev=.00001)), #784,200
                      'U': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['rnn_hidden_d']], stddev=.00001)),#200,200
                      'b': tf.Variable(tf.zeros([hyperParams['rnn_hidden_d'], ])) #200,_
                     },
                'mu': {'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['z_space_d']], stddev=.00001)),#200,50
                       'b': tf.Variable(tf.zeros([hyperParams['z_space_d'], ]))#50,_
                      },
                'sigma':{'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], hyperParams['z_space_d']], stddev=.00001)),#200,50
                            'b': tf.Variable(tf.zeros([hyperParams['z_space_d'], ]))#50,_
                        },
                'pi': {'W': tf.Variable(tf.random_normal([hyperParams['rnn_hidden_d'], 1], stddev=.00001) - .2),#200,1
                         'b': tf.Variable(tf.zeros([1, ]))#1,_
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

        hidden_states = [tf.zeros([tf.shape(self.X)[0], hyperParams['rnn_hidden_d']])] #batchsize, 50

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

            length_check = tf.reduce_max((1. - gamma) * remaining_stick)  #we have batchsize * 1  , if maximum of remainning stick is less than eps
            # we should stop,then

            gamma = tf.cond(length_check < stick_eps, lambda: 0. * gamma + 1., lambda: gamma)
            #Note: we haven't find a way to break the loop when remaining stick size is less than some value. so masking them with zero
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
            expectation_prior += -self.pis[k] * 0.5 * (self.sigmas[k] ** 2 + self.mus[k] ** 2 + 1.837877) #log(2pie) 0.79817986835

        # final objective
        elbo = tf.reduce_mean(nll + expectation_prior + ent_lb_term - self.regularization_weight * ent_mix_weights)

        return elbo

    def get_log_margLL(self):

        # sample a component index #batchsize = tf.shape(self.mus[0])[0]
        uni_samples = tf.random_uniform((tf.shape(self.mus[0])[0], self.K), minval=1e-8, maxval=1-1e-8)
        gumbel_samples = -tf.log(-tf.log(uni_samples))
        component_samples = tf.to_int32(tf.argmax(tf.log(tf.concat(self.pis,1)) + gumbel_samples, 1))
        component_samples = tf.concat([tf.expand_dims(tf.range(0,tf.shape(self.mus[0])[0]),1), tf.expand_dims(component_samples,1)],1)
        #will gebnerate random sequence of indices

        # calc likelihood term for *all* components
        all_ll = []
        for k in xrange(self.K): all_ll.append(-compute_nll(self.X, self.x_recons_linear[k]))
        all_ll = tf.concat(all_ll,1)

        # pick out likelihood terms for sampled K
        ll = tf.gather_nd(all_ll, component_samples)
        ll = tf.expand_dims(ll,1)

        #we will store only those randomly selected values for marginal likelihood
        # calc prior prior':{'mu':0., 'sigma':1.}
        all_log_priors = []
        for k in range(self.K):
            all_log_priors.append( log_normal_pdf(self.z[k], 0., 1.) )
        all_log_priors = tf.concat(all_log_priors,1)

        # pick out prior terms for sampled K
        log_prior = tf.gather_nd(all_log_priors, component_samples)
        log_prior = tf.expand_dims(log_prior,1)

        # calculate all posterior probs
        all_log_gauss_posts = []
        for k in xrange(self.K):
            all_log_gauss_posts.append(log_normal_pdf(self.z[k], self.mus[k], self.sigmas[k]))
        all_log_gauss_posts = tf.concat(all_log_gauss_posts,1)

        # pick out post terms for sampled K
        log_gauss_post = tf.gather_nd(all_log_gauss_posts, component_samples)
        log_gauss_post = tf.expand_dims(log_gauss_post,1)

        return ll + log_prior - log_gauss_post


    def get_samples(self, nImages):
        samples_from_each_component = []
        for k in xrange(self.K):
            #prior':{'mu':0., 'sigma':1.}
            z = 0. + tf.multiply(1.,tf.random_normal((nImages, tf.shape(self.decoder_params['w'][0])[0])))
            samples_from_each_component.append(tf.sigmoid(mlp(z, self.decoder_params)))
        return samples_from_each_component
