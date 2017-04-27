 # -*- coding: utf-8 -*-
config = {
	'input':
		{
			'gene_expression_maximum_level': 10.0,
			'gene_expression_minimum_level': -10.0,
		},
	'unsupervised':
		{
			'gene': 'toy_model/toy_unsupervised_transcriptomics.csv',
			'epochs': 200,
			'batch_size': 5000,
		},

	'supervised':
		{
			'gene': 'toy_model/toy_supervised_transcriptomics.csv',
			'phenotype': 'toy_model/toy_supervised_phenotype_growth.csv',
			'indices': 'toy_model/toy_phenotype_indices.txt',
		},

}

"""
gene_protein_connection_mask
connection_mast
flags.DEFINE_string('protein_phenotype_mask', 'pro_pheno_rule_1.csv',                                                                                           │····················

unsupervised_learning:

flags.DEFINE_string('up_save_to', '', 'File to save unsupervised predictions to')                                                                               │····················

supervised:

flags.DEFINE_string('supervised_gene', '', 'Supervised gene training file.')                                                                                    │····················
flags.DEFINE_string('pheno_indices','','The indices for the phenotype data in supervised training data')                                                        │····················
flags.DEFINE_string('phenotype', '', 'Phenotype data for supervised training.')                                                                                 │····················
flags.DEFINE_string('sup_save_to', '', 'File to save supervised predictions to')                                                                                │····················

"""
