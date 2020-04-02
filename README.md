# DNN-RE
Optimising Rule Extraction for Deep Neural Networks.  The project is motivated by the work of the DeepRED algorithm that performs Rule Extraction on a DNN.  If you have any questions feel free to message on sk940@cam.ac.uk


## Folder structure
```markdown
`<dataset-name>`

`data.csv`

`cross_validation/` - contains data from cross-validated rule_extraction

​	`<n>_folds/` - contains data from e.g. 10_folds/

​		`rule_extraction/`

​			`<rule_ex_mode>/` - e.g. pedagogical, decomp

​				`results.csv` - results for rule extraction using that mode

​				`rules_extracted/` - saving the rules extracted to disk

​					`fold_<n>.rules`

​		`trained_models/` - trained neural network models for each fold

​			`fold_<n>_model.h5`

​		`data_split_indices.txt` - indices for n stratified folds of the data

`neural_network_initialisation/` - contains data from finding the best neural network initialisation

​	`re_results.csv` - rule extraction results from each of the initialisations 

​	`grid_search_results.txt` - results from neural network hyperparameter grid search

​	`data_split_indices.txt` - indices of data split for train and test data

​	`best_initialisation.h5` - best neural network initialisation i.e. the one that generated the smallest ruleset
```

data folder can be found at: https://github.com/sumaiyah/DNN-RE-data