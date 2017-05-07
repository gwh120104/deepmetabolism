import csv
import sys
import math
import time
import sklearn
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
import warnings
import logging

from collections import defaultdict
from numpy import genfromtxt
from autoencoder import PlainAutoEncoder
from scipy.stats.stats import pearsonr
from six.moves import xrange
from config import config

data_process = config['data_process']
gene_max = data_process['gene_expression_maximum_level']
gene_min = data_process['gene_expression_minimum_level']
up = config['unsupervised']
sp = config['supervised']
model_setup = config['model_construct']
pheno_scale = data_process['phenotype_scale_factor']
model_directory = sp['model_save_to']

def get_random_block_from_data(data, batch_size, epoch_start=0):
    start_index = batch_size * epoch_start % len(data)
    end_index = min(start_index + batch_size, len(data))
    return (data[start_index: end_index], (start_index + end_index) / 2)

def read_data():
    print("Reading data...")
    gene_protein_mask = genfromtxt(model_setup['gene_protein_connection_mask'], delimiter=',', dtype="float32")[1:, 1:]
    protein_phenotype_mask = genfromtxt(model_setup['protein_phenotype_connection_mask'], delimiter=',', dtype="float32")[1:,1:].transpose()
    gene_train_raw = genfromtxt(up['gene'], delimiter=',', dtype="float32")
    supervised_gene = genfromtxt(sp['gene'], delimiter=',', dtype="float32")
    supervised_growth = genfromtxt(sp['phenotype'], delimiter=',', dtype="float32")
    pheno_indices_file = open(sp['indices'],'r')
    reader = csv.reader(pheno_indices_file, delimiter=',')
    supervised_pheno_indices = list(reader)[0]
    supervised_pheno_indices = map(int, supervised_pheno_indices)

    # Normalize data as we only handle 0-1 in neural network
    gene_train = np.maximum(np.minimum(gene_train_raw, gene_max), gene_min) / gene_max
    supervised_gene = np.maximum(np.minimum(supervised_gene, gene_max), gene_min) / gene_max
    supervised_growth /= pheno_scale

    n_gene = gene_protein_mask.shape[0]
    n_protein = gene_protein_mask.shape[1]
    n_phenotype = protein_phenotype_mask.shape[1]

    print("Number of Gene: {}\nNumber of Protein: {}\nNumber of Phenotype: {}\n".format(n_gene, n_protein, n_phenotype))

    gene_protein_mask_tensor = tf.constant(gene_protein_mask)
    print("Gene-Protein Connections: {}".format(np.count_nonzero(gene_protein_mask)))

    protein_pheno_mask_tensor = tf.constant(protein_phenotype_mask)
    print("Protein-Phenotype Connections: {}".format(np.count_nonzero(protein_phenotype_mask)))

    auto_encoder = PlainAutoEncoder(n_gene, n_protein, n_phenotype, gene_protein_mask_tensor,
                                    protein_pheno_mask_tensor, supervised_pheno_indices, model_directory)
    return gene_train, gene_train_raw, supervised_gene, supervised_growth, auto_encoder

def predict_and_save(auto_encoder, data):
    # Get the whole data as a batch and predict the results.
    sample_batch_size = len(data)
    sample, _ = get_random_block_from_data(data, sample_batch_size, sample_batch_size)
    result = auto_encoder.get_reconstruction(sample)
    np.savetxt(up['save_to'], np.hstack((np.array(sample), result))[:len(data) / 2], delimiter=',')


def run_unsupervised_training(auto_encoder, gene_train_repeat, non_repeat):
    print("Unsupervised training is started.")
    for i in xrange(up['epochs']):
        batch_x, middle_point = get_random_block_from_data(gene_train_repeat, up['batch_size'], i)
        loss = auto_encoder.partial_fit(batch_x)
        print("\x1b[2K\tUnsupervised batch: {} of {}\t sample middle: {} \trmse: {} \r".format(
           i, up['epochs'], middle_point, math.sqrt(loss / up['batch_size']))),
    if up['save_to'] != '':
        predict_and_save(auto_encoder, non_repeat)


def array_pretty(a):
    return ', '.join(map(str, a.tolist()))


def run_supervised_training(auto_encoder, supervised_gene, supervised_growth):
    print ("\nSupervised training with 10-fold cross validation is started!")
    kf = KFold(n_splits=10, shuffle=True)
    j = 0
    kf.get_n_splits(supervised_gene)
    predictions = []
    actual_values = []

    f = None
    f_sim = None
    if sp['save_to'] != '':
        f = open(sp['save_to'], 'w')

    for train_index, test_index in kf.split(supervised_gene):
        j += 1
	print("{}-fold training is optimizing...".format(j))
        gene_train, gene_test = supervised_gene[train_index], supervised_gene[test_index]
        pheno_train, pheno_test = supervised_growth[train_index], supervised_growth[test_index]
	if len(pheno_train.shape) == 1:
	    pheno_train = np.vstack(pheno_train)
	if len(pheno_test.shape) == 1:
	    pheno_test = np.vstack(pheno_test)
        delta = 10.0
	min_delta = sp['converge_delta']
	max_epoch = sp['epochs']
        last_cost = 1000000
        epoch = 0
        start_time = time.time()
        while math.fabs(delta) > min_delta and epoch < max_epoch:
            supervised_cost = auto_encoder.supervised_fit(gene_train, pheno_train)
            delta = last_cost - supervised_cost
            last_cost = supervised_cost
            epoch +=1
	if epoch == max_epoch:
	    warnings.warn("The supervised training for this fold is not converged!")
	    print("Uncoverged delta is {}.".format(delta))
	else:
	    print("Training supervised until delta converges {} at epoch {}".format(delta, epoch))
        time_cost = time.time()-start_time
        print("The final cost of {}-fold supervised training is {}, time cost is {}.\n".format(j, last_cost, time_cost))
        w, b = auto_encoder.get_weight_and_bias()
        predicted_pheno = auto_encoder.predict_pheno(gene_test) * pheno_scale
        predictions.append(predicted_pheno)
        actual_values.append(pheno_test * pheno_scale)
        for one_gene_sample in range(gene_test.shape[0]):
            actual = pheno_test[one_gene_sample] * pheno_scale
            predicted = predicted_pheno[one_gene_sample]
            if sp['save_to'] != '':
                f.write("%s, %s\n" % (array_pretty(actual), array_pretty(predicted)))
    if f:
	f.close()
    return predictions, actual_values


def model_generation(auto_encoder, supervised_gene, supervised_growth):
    delta = 10.0
    min_delta = sp['converge_delta']
    max_epoch = sp['epochs']
    last_cost = 1000000
    epoch = 0
    start_time = time.time()
    if len(supervised_growth.shape) == 1:
	supervised_growth = np.vstack(supervised_growth)
    while math.fabs(delta) > min_delta and epoch < max_epoch:
	supervised_cost = auto_encoder.supervised_fit(supervised_gene, supervised_growth)
	delta = last_cost - supervised_cost
	last_cost = supervised_cost
	epoch += 1
    if epoch == max_epoch:
	warnings.warn("The final model is not converged.")
	print("Uncoverged delta is {}.".format(delta))
    else:
	print("Final training of DeepMetabolism model converges {} at epoch {}.".format(delta, epoch))
    time_cost = time.time() - start_time
    print("The final cost of model training is {}, the time cost is {}.\n".format(last_cost, time_cost))
        

def run_training(auto_encoder, gene_train_repeat, gene_train, supervised_gene, supervised_growth):
    with tf.Session() as sess:
        auto_encoder.init_variables(sess)
        run_unsupervised_training(auto_encoder, gene_train_repeat, gene_train)
        predict, actual = run_supervised_training(auto_encoder, supervised_gene, supervised_growth)
	model_generation(auto_encoder, supervised_gene, supervised_growth)
	save_path = auto_encoder.model_saver()
	print("Your graphic model is saved to {}.".format(save_path))


#### Do not change anything below
def main(argv):
    gene_train, gene_train_raw, supervised_gene, supervised_growth, auto_encoder = read_data()
    gene_train_repeat = np.repeat(gene_train, axis=0, repeats=2)
    run_training(auto_encoder, gene_train_repeat, gene_train, supervised_gene, supervised_growth)

if __name__ == '__main__':
    tf.app.run()
