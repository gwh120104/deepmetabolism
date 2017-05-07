 # -*- coding: utf-8 -*-
config = {
	'data_process':
		{
			'gene_expression_maximum_level': 10.0,
			'gene_expression_minimum_level': -10.0,
			'phenotype_scale_factor': 1.0,
		},
	'model_construct':
		{
			'gene_protein_connection_mask': 'toy_model/toy_gene_protein_rule.csv',
			'protein_phenotype_connection_mask': 'toy_model/toy_protein_phenotype_rule_growth.csv',

		},
	'unsupervised':
		{
			'gene': 'toy_model/toy_unsupervised_transcriptomics.csv',
			'epochs': 200,
			'batch_size': 5000,
			'save_to': 'toy_model/your_toy_unsupervised_result.csv',
		},

	'supervised':
		{
			'gene': 'toy_model/toy_supervised_transcriptomics.csv',
			'phenotype': 'toy_model/toy_supervised_phenotype_growth.csv',
			'epochs': 100000,
			'converge_delta': 0.0001,
			'indices': 'toy_model/toy_phenotype_indices.txt',
			'save_to': 'toy_model/your_toy_supervised_result.csv',
			'model_save_to': 'toy_model/toy_graphic_model.ckpt',
		},
	'model_input':
		{
			'gene': 'toy_model/toy_supervised_transcriptomics.csv',
			'model_load_from': 'toy_model/toy_graphic_model.ckpt',
			'result_save_to': 'your_prediction_result.csv',
		},
	'model_preparation':
		{
			'input_model': 'toy_model/e_coli_core.xml',
			'available_phenotypes': 'toy_model/pheno_name.txt',
			'gene_protein_connection_mask': 'toy_model/toy_gene_protein_rule.csv',
			'protein_phenotype_connection_mask': 'toy_model/toy_protein_phenotype_rule.csv ',
			'phenotype_indices': 'toy_model/toy_phenotype_indices.txt',

		},

}

