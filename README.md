## MMG4: Recognition of G4-forming sequences based on Markov model

### 1、DATA DETAILS

We have two folders: 'Training and test datasets' and 'Independent dataset'.

In the 'Training and test datasets' folder, there are six files, which are as follows:
1) CATmp.g4seqFirst.F0.1.ex1000.seq.csv: This represents the positive training dataset, stored in CSV format. In this file, the numbers 0, 1, 2, and 3 correspond to the nucleotides A, C, G, and T, respectively. Each row represents a sample, and the length of each sample is uniformly 2000 nt.
2) CATmp_v.g4seqFirst.F0.1.ex1000.seq.csv: This represents the negative training dataset, stored in CSV format. The correspondence between the numbers and nucleotides is the same as in the positive dataset.
It is important to note that the model was trained using a ten-fold cross-validation approach, so you will see that the training and test sets are not separated.
3) pos_ratio_final_log.csv: This file contains the ratio features (log-transformed) of the positive samples obtained using the Markov model, stored in CSV format.
4) neg_ratio_final_log.csv: This file contains the ratio features (log-transformed) of the negative samples obtained using the Markov model, stored in CSV format.
5) norm_pos_feature.csv: This file contains the normalized ratio features of the positive samples.
6) norm_neg_feature.csv: This file contains the normalized ratio features of the negative samples.


### 2、CODE DETAILS
We have one folders: 'code' and there are three files in it.
1) First_Markov.m: This script uses a first-order Markov model to identify G4-forming sequences. It is designed to be run using MATLAB.
2) Second_Markov.m: This script uses a second-order Markov model to identify G4-forming sequences. It is designed to be run using MATLAB.
3) markov_vs_all_the_model.py: This script includes three machine learning models (SVM, Random Forest, and BPnn). It is intended to be run using the Spyder IDE or any other Python code editor.


#### All of the above files contain the entire design concept and training process of the model. By running the relevant files, you can reproduce the results presented in the paper. Additionally, it is worth mentioning that a few plots in the paper were enhanced using other .m or .py scripts to improve their aesthetic quality (e.g., adjusting line colors, modifying font sizes, etc.). As these modifications are minor, we have not included these scripts in the public repository. However, all the core results of the study are fully and authentically represented in the datasets and code that have been made available.
