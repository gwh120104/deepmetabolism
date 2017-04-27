import csv
import sys
import math
import time
import sklearn
from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np

from collections import defaultdict
from numpy import genfromtxt
from autoencoder import PlainAutoEncoder, ShortAutoEncoder
from scipy.stats.stats import pearsonr
# from sklearn.model_selection import KFold
from six.moves import xrange
from config import config

## All constants
growth_scale = 1.0  # divide by

flags = tf.flags
FLAGS = flags.FLAGS

data_process = config['data_process']
gene_max = data_process['gene_expression_maximum_level']
gene_min = data_process['gene_expression_minimum_level']
up = config['unsupervised']
sp = config['supervised']

flags.DEFINE_string('gene_protein_mask', 'gene_pro_rule.csv',
                    '0-1 file indicating connections between gene and protein')
flags.DEFINE_string('protein_phenotype_mask', 'pro_pheno_rule_1.csv',
                    '0-1 file indicating connections between protein and phenotype.')
flags.DEFINE_string('pheno_indices','','The indices for the phenotype data in supervised training data')

flags.DEFINE_string('supervised_gene', '', 'Supervised gene training file.')
flags.DEFINE_string('phenotype', '', 'Phenotype data for supervised training.')
flags.DEFINE_string('up_save_to', '', 'File to save unsupervised predictions to')
flags.DEFINE_string('sup_save_to', '', 'File to save supervised predictions to')
flags.DEFINE_string('save_weight_to', '', 'File to save weights to ')


def get_random_block_from_data(data, batch_size, epoch_start=0):
    start_index = batch_size * epoch_start % len(data)
    end_index = min(start_index + batch_size, len(data))
    return (data[start_index: end_index], (start_index + end_index) / 2)


""" Read in biological regulations for the model (connections between autoencoder layers).
    Both masks are 2-dimensional matries stored as CSV files, with header and first column labels
"""


def read_data():
    print("Reading data...")
    gene_protein_mask = genfromtxt(FLAGS.gene_protein_mask, delimiter=',', dtype="float32")[1:, 1:]
    protein_phenotype_mask = genfromtxt(FLAGS.protein_phenotype_mask, delimiter=',', dtype="float32")[1:,1:].transpose()
    gene_train_raw = genfromtxt(up['gene'], delimiter=',', dtype="float32")# [1:, 1:]
    supervised_gene = genfromtxt(sp['gene'], delimiter=',', dtype="float32")
    supervised_growth = genfromtxt(sp['phenotype'], delimiter=',', dtype="float32")
    pheno_indices_file = open(sp['indices'],'r')
    reader = csv.reader(pheno_indices_file, delimiter=',')
    supervised_pheno_indices = list(reader)[0]
    supervised_pheno_indices = map(int, supervised_pheno_indices)
    # supervised_pheno_indices = genfromtxt(FLAGS.pheno_indices, delimiter=',')
    print supervised_pheno_indices

    # Normalize data as we only handle 0-1 in neural network
    gene_train = np.maximum(np.minimum(gene_train_raw, gene_max), gene_min) / gene_max
    supervised_gene = np.maximum(np.minimum(supervised_gene, gene_max), gene_min) / gene_max
    supervised_growth /= growth_scale

    n_gene = gene_protein_mask.shape[0]
    n_protein = gene_protein_mask.shape[1]
    n_phenotype = protein_phenotype_mask.shape[1]

    print("Number of Gene: {}\nNumber of Protein: {}\nNumber of Phenotype: {}\n".format(n_gene, n_protein, n_phenotype))

    gene_protein_mask_tensor = tf.constant(gene_protein_mask)
    print("Gene-Protein Connections: {}".format(np.count_nonzero(gene_protein_mask)))

    protein_pheno_mask_tensor = tf.constant(protein_phenotype_mask)
    print("Protein-Phenotype Connections: {}".format(np.count_nonzero(protein_phenotype_mask)))

    auto_encoder = PlainAutoEncoder(n_gene, n_protein, n_phenotype, gene_protein_mask_tensor,
                                    protein_pheno_mask_tensor, supervised_pheno_indices)
    return gene_train, gene_train_raw, supervised_gene, supervised_growth, auto_encoder


def print_middle_tier(auto_encoder, supervised_gene):
    results = auto_encoder.inspect_tiers(supervised_gene)
    print("Protein encoding for genes would be: ")
    print(results[0])
    print("Growth rate would be:")
    print(results[1])


def predict_and_save(auto_encoder, data):
    # Get the whole data as a batch and predict the results.
    sample_batch_size = len(data)
    sample, _ = get_random_block_from_data(data, sample_batch_size, sample_batch_size)
    result = auto_encoder.get_reconstruction(sample)
    np.savetxt(FLAGS.up_save_to, np.hstack((np.array(sample), result))[:len(data) / 2], delimiter=',')


def run_unsupervised_training(auto_encoder, gene_train_repeat, non_repeat):
    for i in xrange(up['epochs']):
        batch_x, middle_point = get_random_block_from_data(gene_train_repeat, up['batch_size'], i)
        loss = auto_encoder.partial_fit(batch_x)
        print("Unsupervised batch: %d\t sample middle: %d \trmse: %f" % (
            i, middle_point, math.sqrt(loss / up['batch_size'])))
    if FLAGS.up_save_to != '':
        predict_and_save(auto_encoder, non_repeat)


def array_pretty(a):
    return ', '.join(map(str, a.tolist()))


def run_supervised_training(auto_encoder, supervised_gene, supervised_growth):
    kf = KFold(n_splits=10, shuffle=True)
    j = 0
    kf.get_n_splits(supervised_gene)
    predictions = []
    actual_values = []
    simulations = []
    actual_train_values = []

    f = None
    f_sim = None
    if FLAGS.sup_save_to != '':
        f = open(FLAGS.sup_save_to, 'w')
    if FLAGS.sim_save_to != '':
        f_sim = open(FLAGS.sim_save_to, 'w')

    for train_index, test_index in kf.split(supervised_gene):
        j += 1
        gene_train, gene_test = supervised_gene[train_index], supervised_gene[test_index]
        pheno_train, pheno_test = supervised_growth[train_index], supervised_growth[test_index]
	if len(pheno_train.shape) == 1:
	    pheno_train = np.vstack(pheno_train)
	if len(pheno_test.shape) == 1:
	    pheno_test = np.vstack(pheno_test)
        delta = 1.0
	min_delta = 0.0001
	max_epoch = 100000
        last_cost = 1000000
        print(" Optimizing supervised learning problem ...")
        epoch = 0

        start_time = time.time()
        while math.fabs(delta) > min_delta and epoch < max_epoch:
            # print(epoch, delta)
	# print pheno_train.shape
            supervised_cost = auto_encoder.supervised_fit(gene_train, pheno_train)
            delta = last_cost - supervised_cost
            last_cost = supervised_cost
            epoch +=1
        print("Training supervised until delta converges {} at epoch {}".format(delta, epoch))
        time_cost = time.time()-start_time
        print("The final cost of {}-fold supervised training is {}, time cost is {}.".format(j, last_cost, time_cost))

        # print("--- %s seconds ---" % (time.time() - start_time))
        w, b = auto_encoder.get_weight_and_bias()

        if FLAGS.save_weight_to != '':
            # TODO(eric): to save the model in a way for prediction, but not the weight here.
            print(w.shape, b.shape, w.dtype, b.dtype)
            np.savetxt(FLAGS.save_weight_to + "_w_" + str(j) + ".txt", w)
            with open(FLAGS.save_weight_to + "_b_" + str(j) + ".txt", "w") as fb:
                fb.write(str(b))

        # Print the supervised testing performance. TODO(weihua)
        predicted_pheno = auto_encoder.predict_pheno(gene_test)  # * growth_scale # Change
        predictions.append(predicted_pheno)
        actual_values.append(pheno_test)  # * growth_scale
	# print pheno_test.shape
        for one_gene_sample in range(gene_test.shape[0]):
            actual = pheno_test[one_gene_sample]  # * growth_scale
            predicted = predicted_pheno[one_gene_sample]
            # TODO: Eric: could you please print all the five phenotypes in one file? Thanks!
            if FLAGS.sup_save_to != '':
		# print actual
		# print predicted
                f.write("%s, %s\n" % (array_pretty(actual), array_pretty(predicted)))
            # else:
            #     print("%s, %s" % (array_pretty(actual), array_pretty(predicted)))

        # Print the training performance.
        simulated_pheno = auto_encoder.predict_pheno(gene_train)  # * growth_scale # Change
        simulations.append(simulated_pheno)
        actual_train_values.append(pheno_train)  # * growth_scale
        for one_gene_sample in range(gene_train.shape[0]):
            actual_train = pheno_train[one_gene_sample]  # * growth_scale
            simulated = simulated_pheno[one_gene_sample]

    if f:
        f.close()
    if f_sim:
        f_sim.close()
    return predictions, actual_values #, simulations, actual_train_values


def run_training(auto_encoder, gene_train_repeat, gene_train, supervised_gene,
                 supervised_growth):
    with tf.Session() as sess:
        auto_encoder.init_variables(sess)
        run_unsupervised_training(auto_encoder, gene_train_repeat, gene_train)
        predict, actual = run_supervised_training(auto_encoder, supervised_gene, supervised_growth)


#### Do not change anything below
def main(argv):
    # start_time = time.time()
    gene_train, gene_train_raw, supervised_gene, supervised_growth, auto_encoder = read_data()
    gene_train_repeat = np.repeat(gene_train, axis=0, repeats=2)
    run_training(auto_encoder, gene_train_repeat, gene_train, supervised_gene, supervised_growth)
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    tf.app.run()
