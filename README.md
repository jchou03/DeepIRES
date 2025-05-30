# DeepIRES
DeepIRES: a hybird deep learning model for indentifying internal ribosome entry site in mRNA
## EXPLANATION
This repository contains four folders: model, dataset, weights, data, and result.
### Model
This folder contains python code files for constructing model.
### Dataset
This folder contains orginal data, traing dataet and testing dataset we constructed.
### Weights
This folder saves the model weights we trained.
### Data
You can put your input data in .fa format  in this folder to run prediction program. It already contains our testing sets
### Result
This folder is used to save prediction output file
## Installation of DeepIRES and environment
Download the repository and create corresponding environment.

```
git clone https://github.com/SongLab-at-NUAA/DeepIRES.git
cd ./DeepIRES
conda env create -f environment.yml
```
Then activate virtual enviroment

```
conda activate DeepIRES
```
## USAGE
Run DeepIRES.py file to predict IRES
```
python DeepIRES.py -i input_file -o output_name 
```
```
-i input file : the input file in .fa or .fasta format.This file should be located in data folder containing the sequneces you want to predict.
-o output name : the output name you want to use.The result is saved in .csv format in result folder.
```
### For example
```
python DeepIRES.py -i main_independent.fa -o main
python DeepIRES.py -i core.fa -o core
python DeepIRES.py -i 5utr_independent.fa -o 5utr
```
There are four columns in the prediction results tabular：sequence name, IRES score, the start locations of region may contain IRES, the termination locations of region may contain IRES.

### Notes for Green Lab Research
This repository is a fork of the DeepIRES repository that implements code testing the SANDSTORM model. 
the `Sandstorm` folder stores code defining the SANDSTORM model alongside its utility functions as well as Jupyter notebooks analyzing the performance of the SANDSTORM model on DeepIRES's training data as well as evaluating the outputs from hidden layers of the SANDSTORM model.
Furthermore, there are notebooks in the `model` folder that tests DeepIRES's performance as well as analyzing the outputs from hidden layers from DeepIRES.

The `data_subsets` folder stores different subsets of the IRES sequences of interest. All sequenes were put through DeepIRES with different activation thresholds, then a TSNE was performed on the hidden outputs from DeepIRES and sections of interest were extracted. `sandstorm_dense_4_output.csv` contains a subset of sequences that a hidden layer in Sandstorm identified as a cluster with more active sequences.
