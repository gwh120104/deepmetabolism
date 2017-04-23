What is DeepMetabolism?
--------------------------

DeepMetabolism is a biology-guided deep learning algorithm to predict phenotype from genome sequencing. DeepMetabolism is a project to use deep learning techniques to decipher complicated relations in biological network and to further bridge the gap between genotype and phenotype. The general design principles are:

* Interactions in biological networks can be simulated by a a computational network, represented as a graph model.
* Unsupervised machine learning techniques (such as autoencoder) can be used to train the computational network to approximate the biological network.
* The whole process is end-to-end, without tweaking between layers.


How the network is built?
----------------------------------

The layers in the deep learning network of DeepMetabolism is not fully connected due to the biological constraints based on the constraint-based metabolic modeling. The connections between gene and protein layers are defined by the gene-protein-reaction (GPR) association from a well-developed genome-scale metabolic model (GSM model). We recommand to download the high-quality GSM models from [BiGG model database](bigg.ucsd.edu). These connections can be parsed from `cobrapy_model.py`, which used the [cobrapy package](https://github.com/opencobra/cobrapy). The connections between protein and phenotype layers are defined by the essential proteins of each phenotpe based on the GPR association. In general, if the gene knock-out will lead to 5% decrease of the maximum value of the phenotype of interest, the protein node(s) related to this gene will be connected to this phenotype node. To define the connections between each two layers, we provide the `cobrapy_model.py` to directly generate two masking matrices used in the DeepMetabolism training code.

How the network is modeled?
------------------------------

Right now we use a vanilla autoencoder, where the input is gene expression level. We try to reconstruct gene expression level by going through the protein layer and the phenotype layer. Protein and phenotype layers are feed-forward layers (no recurrent or backward connections). The overall architecture of the autoencoder is:

<kbd>Gene Expression</kbd> -> <kbd>Protein </kbd> -> <kbd>Phenotype</kbd> -> <kbd>Protein Reconstructed</kbd> -> <kbd> Gene Reconstructed</kbd>

The autoencoder model is defined in `autoencoder.py`. Note that layers are not fully connected (unlike your normal autoencoder) due to biological constraints from the gene-protein-reaction association. In other words, not all gene is affecting all proteins, and not all protein is affecting all phenotypes. It is an obvious truth in biology. When defining the model parameters, we pad the parameters such that weights between layers are defined by a 2-dimensional `[OUT, IN]` tensor `W` such that `W * v_in = v_out`. We mask `W` with `M` such that `W * M * v_in = v_out` where `M` is a 0/1 matrix.

DeepMetabolism usage
-------------------------------

Activate the virtual environment to use GPU
> source ~/tf/bin/activate

Run the training code

> python train_auto_encoder.py --gene_protein_mask masking_matrix_between_gene_protein_layer --protein_phenotype_mask masking_matrix_between_protein_phenotype_layer --gene unsupervised_training_data --supervised_gene supervised_training_x --phenotype supervised_training_y --pheno_indices phenotype_indices_with_training_data_in_phenotype_layer --unsupervised_epochs epoch_number_of_unsupervised_training  --up_save_to unsupervised_training_result --sup_save_to supervised_training_result


An example would be:

Activate the virtual environment `tf` to use GPU
> source ~/tf/bin/activate

Run the training code for `toy model`

> python train_auto_encoder.py --gene_protein_mask ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_gene_pro_rule.csv --protein_phenotype_mask ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_pro_pheno_rule_growth.csv --gene ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_un_trans.csv --supervised_gene ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_su_trans.csv --phenotype ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_su_pheno_growth.csv --pheno_indices ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_pheno_indices.txt --unsupervised_epochs 50  --up_save_to un_result_toy.csv --sup_save_to su_result_toy.csv
