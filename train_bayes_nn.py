import os
from os.path import join as pjoin
import cPickle as cp

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from models.bnn import BNN

# command line arguments
flags = tf.flags
flags.DEFINE_integer("batchSize", 128, "batch size.")
flags.DEFINE_integer("nEpochs", 1500, "number of epochs to train.")
flags.DEFINE_float("adamLr", 3e-4, "AdaM learning rate.")
flags.DEFINE_integer("hidden_size", 1000, "number of hidden units in en/decoder.")
flags.DEFINE_integer("n_hidden_layers", 2, "number of hidden layers.")
flags.DEFINE_float("prior_mu", 0., "prior mean.")
flags.DEFINE_float("prior_sigma", 1., "prior sigma.")
flags.DEFINE_string("experimentDir", "results/rotatedMNIST/", "directory to save training artifacts.")
inArgs = flags.FLAGS


def get_file_name(expDir, vaeParams, trainParams):     
    # concat hyperparameters into file name
    output_file_base_name = '_'+''.join('{}_{}_'.format(key, val) for key, val in sorted(vaeParams.items()) if key not in ['output_d', 'input_d', 'prior'])
    output_file_base_name += ''.join('{}_{}_'.format(key, vaeParams['prior'][key]) for key in sorted(['mu', 'sigma']))
    output_file_base_name += 'adamLR_'+str(trainParams['adamLr'])
                                                                               
    # check if results file already exists, if so, append a number                                                                                               
    results_file_name = pjoin(expDir, "train_logs/bayesNN_trainResults"+output_file_base_name+".txt")
    file_exists_counter = 0
    while os.path.isfile(results_file_name):
        file_exists_counter += 1
        results_file_name = pjoin(expDir, "train_logs/bayesNN_trainResults"+output_file_base_name+"_"+str(file_exists_counter)+".txt")
    if file_exists_counter > 0:
        output_file_base_name += "_"+str(file_exists_counter)

    return output_file_base_name


def compute_class_error(onehot_labels, preds):
    pred_idxs = np.argmax(preds, axis=1)
    true_idxs = np.argmax(onehot_labels, axis=1)
    return 1 - accuracy_score(true_idxs, pred_idxs)


### Training function
def trainNN(data, nn_hyperParams, train_hyperParams, param_save_path, logFile=None):

    N_train, d = data['train'][0].shape
    N_valid, d = data['valid'][0].shape
    nTrainBatches = N_train/train_hyperParams['batchSize']
    nValidBatches = N_valid/train_hyperParams['batchSize']

    # init Mix Density VAE
    model = BNN(nn_hyperParams)

    # get training op
    optimizer = tf.train.AdamOptimizer(train_hyperParams['adamLr']).minimize(model.loss_fn)

    # get op to save the model
    persister = tf.train.Saver()

    with tf.Session(config=train_hyperParams['tf_config']) as s:
        s.run(tf.initialize_all_variables())
        
        # for early stopping
        best_loss = 10000000.
        best_epoch = 0

        for epoch_idx in xrange(train_hyperParams['nEpochs']):

            # training
            train_loss = 0.
            for batch_idx in xrange(nTrainBatches):
                x = data['train'][0][batch_idx*train_hyperParams['batchSize']:(batch_idx+1)*train_hyperParams['batchSize'],:]
                y = data['train'][1][batch_idx*train_hyperParams['batchSize']:(batch_idx+1)*train_hyperParams['batchSize'],:]

                _, loss_fn_val = s.run([optimizer, model.loss_fn], {model.X: x, model.Y: y})
                train_loss += loss_fn_val

            # validation
            valid_loss = 0.
            for batch_idx in xrange(nValidBatches):
                x = data['valid'][0][batch_idx*train_hyperParams['batchSize']:(batch_idx+1)*train_hyperParams['batchSize'],:]
                y = data['valid'][1][batch_idx*train_hyperParams['batchSize']:(batch_idx+1)*train_hyperParams['batchSize'],:]

                valid_loss += s.run(model.loss_fn, {model.X: x, model.Y: y})

            # check for ELBO improvement
            star_printer = ""
            train_loss /= nTrainBatches
            valid_loss /= nValidBatches
            if valid_loss < best_loss: 
                best_loss = valid_loss
                best_epoch = epoch_idx
                star_printer = "***"
                # save the parameters
                persister.save(s, param_save_path)

            # log training progress
            logging_str = "Epoch %d.  Train Loss: %.3f,  Validation Loss: %.3f %s" %(epoch_idx+1, train_loss, valid_loss, star_printer)
            print logging_str
            if logFile: 
                logFile.write(logging_str + "\n")
                logFile.flush()

            # check for convergence
            if epoch_idx - best_epoch > train_hyperParams['lookahead_epochs'] or np.isnan(train_loss): break  

    return model


### Marginal Likelihood Calculation            
def testNN(data, model, param_file_path):
    N,d = data[0].shape

    # get op to load the model                                                                                               
    persister = tf.train.Saver()

    with tf.Session() as s:
        persister.restore(s, param_file_path)
        predictions = s.run(model.y_hat, {model.X: data[0]})
                 
    return compute_class_error(data[1], predictions)


if __name__ == "__main__":

    # load MNIST
    mnist = cp.load(open("rotated_MNIST.pkl", "rb"))

    # set architecture params
    nn_hyperParams = {'input_d':mnist['train'][0].shape[1], 'output_d':mnist['train'][1].shape[1], 'hidden_d':inArgs.hidden_size, 'n_hidden_layers':inArgs.n_hidden_layers, 'prior':{'mu':inArgs.prior_mu, 'sigma':inArgs.prior_sigma}}

    # set training hyperparameters
    train_hyperParams = {'adamLr':inArgs.adamLr, 'nEpochs':inArgs.nEpochs, 'batchSize':inArgs.batchSize, 'lookahead_epochs':25, \
                         'tf_config': tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5), log_device_placement=False)}

    # setup files to write results and save parameters
    outfile_base_name = get_file_name(inArgs.experimentDir, nn_hyperParams, train_hyperParams)
    logging_file = open(inArgs.experimentDir+"train_logs/bayesNN_trainResults"+outfile_base_name+".txt", 'w')
    param_file_name = inArgs.experimentDir+"params/bayesNN_params"+outfile_base_name+".ckpt"

    # train
    print "Training model..."
    model = trainNN(mnist, nn_hyperParams, train_hyperParams, param_file_name, logging_file)

    # evaluate the model
    print "\n\nCalculating test performance..."
    valid_error = testNN(mnist['valid'], model, param_file_name)
    test_error = testNN(mnist['test'], model, param_file_name)
    logging_str = "*FINAL* Valid Error: %.4f, Test Error: %.4f" %(valid_error, test_error)
    print logging_str
    logging_file.write(logging_str+"\n")
    logging_file.close()
