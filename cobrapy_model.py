# Read the genome scale metabolic model from SBML files with CoBRApy 2.0
# Usage python cobrapy_model.py $input genome scale model (json) $ save files for the mask matrix between protein and phenotype layers $ save files for the mask matrix between gene and protein layers
# Usage:  python cobrapy_model.py e_coli_core.json test_PPM.csv test_GPM.csv
import sys
import cobra
import json
import math
import pickle
import numpy as np
from cobra.core.Gene import eval_gpr, parse_gpr, ast2str
from config import config
from collections import defaultdict

input_dict = config['model_preparation']
input_file_name = input_dict['input_model']
pheno_name = input_dict['available_phenotypes']
pp_mask_matrix = input_dict['protein_phenotype_connection_mask']
gp_mask_matrix = input_dict['gene_protein_connection_mask']
pheno_indices_save = input_dict['phenotype_indices']


"""Function to read multiple formats of CoBRA model."""
def read_model(input_file_name):
	if input_file_name.endswith("json"):
		model = cobra.io.load_json_model(input_file_name)
		return model
	elif input_file_name.endswith("xml"):
		model = cobra.io.read_sbml_model(input_file_name)
		return model
	elif input_file_name.endswith("mat"):
		model = cobra.io.load_matlab_model(input_file_name)
		return model
	else:
		print "{} is not parsed.".format(input_file_name)
		return None

reaction_id_rec = []
phenotype_id_rec = []
phenotype_name = []
phenotype_name_id_dict = defaultdict(list)
pseudo_gene_rec = []
uptake_id = []
substrate_name = '_glucose_'
"""Shut down the target flux"""
def ko_flux(input_file_name, reac_id, target_id, uptake_id):
	# model = cobra.io.load_json_model(input_file_name)
	model = fix_uptake(input_file_name, uptake_id)
	target_obj = model.reactions.get_by_id(target_id)
	model.change_objective(target_obj)
	output_model = model
	target_reac = output_model.reactions.get_by_id(reac_id)
	target_reac.lower_bound = 0.0
	target_reac.upper_bound = 0.0
	check_reac = output_model.reactions.get_by_id(reac_id)
	check_lower_bound = check_reac.lower_bound
	check_upper_bound = check_reac.upper_bound
	if check_lower_bound == 0.0 and check_upper_bound == 0.0:
		# print "Reaction {} is knocked out.".format(reac_id)
		return output_model
	else:
		return None
"""Fix the uptake flux"""
def fix_uptake(input_file_name, uptake_id):
	model = read_model(input_file_name)
# 	model = cobra.io.load_json_model(input_file_name)
	if uptake_id:
		uptake_reac = model.reactions.get_by_id(uptake_id)
		uptake_reac.lower_bound = -100.0
		uptake_reac.upper_bound = -100.0
		# print "Uptake reaction {} is fixed to {}.".format(uptake_id, uptake_reac.upper_bound)
		return model
	else:
		return None
model = read_model(input_file_name)
"""Record all the reaction ids in the model and the phenotypic reactions (including metabolites exchanges and biomass)"""
for x in model.reactions:
	reac_id = x.id
	reac_name = x.name
	reaction_id_rec.append(x.id)
	temp_reac = model.reactions.get_by_id(reac_id)
	if 'exchange' in reac_name.lower():
		phenotype_id_rec.append(reac_id)
 		phenotype_name.append(reac_name)
		phenotype_name_id_dict[reac_name] = reac_id
		if substrate_name in reac_name.lower():
			uptake_id = reac_id
	if 'biomass' in reac_name.lower():
		phenotype_id_rec.append(reac_id)
 		phenotype_name.append(reac_name)
		phenotype_name_id_dict[reac_name] = reac_id
		growth_id = reac_id
	if temp_reac.lower_bound == -10.0:
		uptake_id = reac_id
print "Number of phenotypes: {}".format(len(phenotype_id_rec))

# print("bounds before knockout:", (temp_reac.lower_bound, temp_reac.upper_bound))
# print type(model.genes)
gene_id_rec = []
for y in model.genes:
	gene_id = y.id
	"""Do not include the pseudo-genes"""
	if "None" in y.name:
# 		print y.name
#		gene_id_rec.append(gene_id)
		pseudo_gene_rec.append(gene_id)
#		temp_gene = model.genes.get_by_id(gene_id)
	else:
		gene_id_rec.append(gene_id)
		temp_gene = model.genes.get_by_id(gene_id)

	# sys.exit()
"""Start to screen each phenotype and find the corresponding essencial reactions."""
max_pheno_rec = []
flux_pheno_rec = []
pheno_max_value_dict = defaultdict(list) # Key: id of phenotypic reactionsfv
pheno_ess_reac_dict = defaultdict(list) # Key: id of phenotypic reactions, values: corresponding essential reactions (vKO<0.95vOrg)
ess_thr = 0.95
for reac in phenotype_id_rec:
	# Re-initialize the model
	# model = cobra.io.load_json_model(input_file_name)
	model = fix_uptake(input_file_name, uptake_id)
	target_obj = model.reactions.get_by_id(reac)
	model.change_objective(target_obj)
	growth = model.reactions.get_by_id(growth_id)
	growth_oc = growth.objective_coefficient
	target_oc = target_obj.objective_coefficient
	# print "Objective coefficent of growth is {}.".format(growth_oc)
	print "Check objetive function of {}: {}.".format(target_obj, target_oc)
	org_solution = model.optimize()
	if reac == uptake_id:
		print "Minimize the uptake reaction."
		org_solution = model.optimize(objective_sense = 'minimize')
	if org_solution.status == 'optimal':
		org_pheno_f = org_solution.f
		max_pheno_rec.append(org_solution.f)
		temp_flux = org_solution.x
		flux_pheno_rec.append(temp_flux)
		pheno_max_value_dict[reac] = org_pheno_f
		print "Objective value is {}.".format(org_solution.f)
		print "Start to screen phenotypic reaction {}.\n".format(target_obj)
	"""Screen all the reactions of a phenotype"""
	for scr_reac in model.reactions:
		scr_id = scr_reac.id
		test_model = ko_flux(input_file_name, scr_id, reac, uptake_id)
		if test_model:
			test_solution = test_model.optimize()
			test_f = test_solution.f
			test_stat = test_solution.status
		if test_stat != 'optimal':
			pheno_ess_reac_dict[reac].append(scr_id)
			# print "Reaction {} is an essential reaction for {}.".format(scr_id, reac)
		elif test_stat == 'optimal':
			vKO = math.fabs(test_f)
			vOrg = math.fabs(org_pheno_f)
			if vKO <= vOrg*ess_thr:
				pheno_ess_reac_dict[reac].append(scr_id)
				# print "Reaction {} is an essential reaction for {}.".format(scr_id, reac)
"""Extract the related genes in the essential reactions as the essential genes of the phenotypic data"""
pheno_ess_gene_dict = defaultdict(list)
ess_gene_pheno_dict = defaultdict(list)
for keys, values in pheno_ess_reac_dict.items():
	print "Reaction {} has {} essential reactions.".format(keys, len(values))
	for ess_reac_id in values:
		ess_reac = model.reactions.get_by_id(ess_reac_id)
		temp_gpr = parse_gpr(ess_reac.gene_reaction_rule)
		# print len(temp_gpr[1])
		for z in temp_gpr[1]:
			if z not in pheno_ess_gene_dict[keys] and z not in pseudo_gene_rec:
				pheno_ess_gene_dict[keys].append(z)
			ess_gene_pheno_dict[z].append(keys)
	print "Reaction {} has {} essential genes.\n".format(keys, len(pheno_ess_gene_dict[keys]))

# pheno_ess_bool_dict = defaultdict(list)
# ess_pheno_bool_dict = defaultdict(list)
"""Generate the final output protein-phenotype masking matrix."""
row_num = len(phenotype_id_rec)
col_num = len(gene_id_rec)
mask_matrix = np.zeros((row_num, col_num),dtype = np.int)
pheno_id = pheno_ess_gene_dict.keys()
for keys, values in pheno_ess_gene_dict.items():
	for genes in values:
		row_index = pheno_id.index(keys)
		col_index = gene_id_rec.index(genes)
		# print "{}, {}".format(pheno_id.index(keys), gene_id_rec.index(genes))
		mask_matrix[row_index, col_index] = 1
np.savetxt(pp_mask_matrix, mask_matrix, delimiter = ',')

"""Print the gene list"""
gene_file = open('gene_id_name.txt', 'wb')
pheno_file = open('pheno_id_name.txt','wb')
for genes in gene_id_rec:
	print>>gene_file, genes
for pheno in pheno_id:
	print>>pheno_file, pheno

gp_mask_matrix_one = np.ones((col_num, col_num))
gp_mask_matrix_value = np.diag(np.diag(gp_mask_matrix_one))
np.savetxt(gp_mask_matrix, gp_mask_matrix_value, delimiter = ',')

pheno_name_file = open(pheno_name, "rb")
pheno_indices = []
for line in pheno_name_file:
	line = line.strip()
	matching = [s for s in phenotype_name if line in s.lower()]
	matched_id = phenotype_name_id_dict.get(matching[0])
	print matching
	pheno_indices.append(str(pheno_id.index(matched_id)))
pheno_indices_file = open(pheno_indices_save, "wb")
print>>pheno_indices_file, ','.join(pheno_indices)
