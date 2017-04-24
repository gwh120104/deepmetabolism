What is DeepMetabolism?
--------------------------

DeepMetabolism is a deep learning algorithm that predicts cell phenotypes from genome sequencing data such as transcriptomics data. DeepMetabolism uses biological knowledge to design a neural network model and integrates unsupervised learning with supervised learning for model prediction. DeepMetabolism meets the design criteria of high accuracy, high speed, and high robustness. By doing so, DeepMetabolism can be used to enrich our understanding of genotype-phenotype correlation and be applied in fields of synthetic biology, metabolic engineering and precision medicine. 

The general design principles are:
* Interactions in biological networks can be simulated by a a computational network, represented as a graph model.
* Unsupervised machine learning techniques (such as autoencoder) can be used to train the computational network to approximate the biological network.
* The whole process is end-to-end, without tweaking between layers.

To develop DeepMetabolism, we implemented a two-step, deep-learning process that used different sets of data. The first step was unsupervised learning that used transcriptomics data, and the second step was supervised learning that used paired data of transcriptomics and phenotype (Figure 1). We expected that unsupervised learning would provide a “rough” neural network model that captured the essence of connections between transcriptomics data and phenotype, while supervised training would fine tune this model and lead to highly accurate predictions of phenotype from transcriptomics data.

![overview_deepmetabolism](https://github.com/gwh120104/deepmetabolism/blob/master/img/Figure_README.png)
Figure 1. Overview of DeepMetabolism algorithm


DeepMetabolism model
----------------------------------

In DeepMetabolism, we use a vanilla autoencoder, where the input is transcriptomics data (i.e., log2 fold changes of gene expression levels), to implement unsupervised learning process. We try to reconstruct gene expression level by going through the protein layer and the phenotype layer. Protein and phenotype layers are feed-forward layers (no recurrent or backward connections). The overall architecture of the autoencoder is:

<kbd>Gene Expression</kbd> -> <kbd>Protein </kbd> -> <kbd>Phenotype</kbd> -> <kbd>Protein Reconstructed</kbd> -> <kbd> Gene Reconstructed</kbd>

The autoencoder model is defined in `autoencoder.py`. Note that the layers of the autoencoder model were not fully connected. Instead, we applied biological knowledge to rationally define the connections, which could reduce the risk of over-parameterization and increase the calculation speed. To connect the first layer (i.e., gene layer) and the second layer (i.e., protein layer), we applied the gene-protein association from well-developed, genome-scale metabolic models. We recommand to download the high-quality GSM models from [BiGG model database](bigg.ucsd.edu). To connect the second layer (i.e., protein layer) and the third layer (i.e., phenotype layer), we applied [cobrapy package](https://github.com/opencobra/cobrapy) on the well-developed genome-scale model to identify the proteins that were essential for a certain phenotype (e.g., proteins that were essential for cell growth) and connected these proteins with the corresponding phenotype. 

When defining the model parameters, we pad the parameters such that weights between layers are defined by a 2-dimensional `[OUT, IN]` tensor `W` such that `W * v_in = v_out`. We mask `W` with masking matrix `M` such that `W * M * v_in = v_out` where `M` is a 0/1 matrix. `M` between each two layers can be obtained from `cobrapy_model.py`.


DeepMetabolism usage
-------------------------------

Generate masking matrices `M`, and the lists of related genes `gene_id_name.txt` and phenotypes `pheno_id_name.txt` from genome-scale metabolic model
>python cobrapy_model.py genome_scale_metabolic_model_in_sbml_formate masking_matrix_between_protein_phenotype_layer masking_matrix_between_gene_protein_layer

Activate the virtual environment to use GPU
> source ~/tf/bin/activate

Run the training code
> python train_auto_encoder.py --gene_protein_mask masking_matrix_between_gene_protein_layer --protein_phenotype_mask masking_matrix_between_protein_phenotype_layer --gene unsupervised_training_data --supervised_gene supervised_training_x --phenotype supervised_training_y --pheno_indices phenotype_indices_with_training_data_in_phenotype_layer --unsupervised_epochs epoch_number_of_unsupervised_training  --up_save_to unsupervised_training_result --sup_save_to supervised_training_result


A toy model example:

TODO: Weihua
Give the description of toy model... (Files, how we set up, size, reaction number, phenotype number.)


Activate the virtual environment `tf` to use GPU

> source ~/tf/bin/activate

Run the training code for `toy model`

> python train_auto_encoder.py --gene_protein_mask ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_gene_pro_rule.csv --protein_phenotype_mask ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_pro_pheno_rule_growth.csv --gene ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_un_trans.csv --supervised_gene ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_su_trans.csv --phenotype ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_su_pheno_growth.csv --pheno_indices ~/Dropbox/MachineLearning/autoencoder/Model_Design/toy_model/toy_pheno_indices.txt --unsupervised_epochs 50  --up_save_to un_result_toy.csv --sup_save_to su_result_toy.csv

If you want to run the genome model, you can ...
