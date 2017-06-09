import csv
import sys
import math
import time
import sklearn
import tensorflow as tf
import numpy as np

from collections import defaultdict
from numpy import genfromtxt
from autoencoder import PlainAutoEncoder
from confige import config

data_process = config['data_process']
gene_max = data_process['gene_expression_maximum_level']
gene_min = data_process['gene_expression_minimum_level']
sp = config['supervised']
model_input = config['model_input']
model_setup = config['model_construct']
pheno_scale = data_process['phenotype_scale_factor']
model_directory = model_input['model_load_from']
predict_save_to = model_input['result_save_to']

def read_data():
    print("Reading data...")
    gene_protein_mask = genfromtxt(model_setup['gene_protein_connection_mask'], delimiter=',', dtype="float32")[1:, 1:]
    protein_phenotype_mask = genfromtxt(model_setup['protein_phenotype_connection_mask'], delimiter=',', dtype="float32")[1:,1:].transpose()
    input_gene = genfromtxt(model_input['gene'], delimiter=',', dtype="float32")
    pheno_indices_file = open(sp['indices'],'r')
    reader = csv.reader(pheno_indices_file, delimiter=',')
    supervised_pheno_indices = list(reader)[0]
    supervised_pheno_indices = map(int, supervised_pheno_indices)

    # Normalize data as we only handle 0-1 in neural network
    gene_train = np.maximum(np.minimum(input_gene, gene_max), gene_min) / gene_max

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
    return gene_train, input_gene, auto_encoder


def run_training(auto_encoder,  gene_train):
    with tf.Session() as sess:
        auto_encoder.init_variables(sess)
	auto_encoder.model_loader()
	prediction = auto_encoder.predict_pheno(gene_train)
	np.savetxt(predict_save_to, prediction, delimiter = ',')
	print("Your phenotype prediction is save to {}.".format(predict_save_to))

#### Do not change anything below
def main(argv):
    gene_train, gene_train_raw, auto_encoder = read_data()
    run_training(auto_encoder, gene_train)

if __name__ == '__main__':
    tf.app.run()
