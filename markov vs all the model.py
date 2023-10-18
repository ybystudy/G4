# -*- coding: utf-8 -*-
"""

Compare all the machine learning model with Markov

"""

#%%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import matplotlib.pyplot as plt
#%%
#The normalized ln(Ratio) data was imported
Bp_ratio_pos = pd.read_csv('norm_BP_pos_feature.csv', header=None, sep=',');
Bp_ratio_neg = pd.read_csv('norm_BP_neg_feature.csv', header=None, sep=',');
#The unnormalized ln(Ratio) data was imported
ln_unnorm_ratio_pos = pd.read_csv('pos_ratio_final_log.csv', header=None, sep=',');
ln_unnorm_ratio_neg = pd.read_csv('neg_ratio_final_log.csv', header=None, sep=',');

Bp_ratio_posmatrix = np.asarray(Bp_ratio_pos.values);
Bp_ratio_negmatrix = np.asarray(Bp_ratio_neg.values);
Bp_ratio_data = np.concatenate((Bp_ratio_posmatrix, Bp_ratio_negmatrix), axis=0);

ln_unnorm_ratio_posmatrix = np.asarray(ln_unnorm_ratio_pos.values);
ln_unnorm_ratio_negmatrix = np.asarray(ln_unnorm_ratio_neg.values);
ln_unnorm_ratio_data = np.concatenate((ln_unnorm_ratio_posmatrix, ln_unnorm_ratio_negmatrix), axis=0);
#%%
#Markov model data were verified with interval 10
markov_interval10 = ln_unnorm_ratio_data[:, 9] 
  
#%%Generate random tags
label_vector = np.concatenate((np.ones((Bp_ratio_posmatrix.shape[0],1)), np.zeros((Bp_ratio_negmatrix.shape[0],1))))
BP_idx = np.random.permutation(Bp_ratio_data.shape[0])
BP_ratio_data_mess = Bp_ratio_data[BP_idx, :]
label_vector_mess = label_vector[BP_idx]
markov_interval10_mess = markov_interval10[BP_idx]
#%%Ten-fold cross validation method is used to divide the training set and the test set

from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

roc_data = []
prc_data = []
acc_list = []  

# Start cross-validation
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
   
    xTrain = torch.from_numpy(np.array(BP_ratio_data_mess[train_idx], dtype="float32"))
    yTrain = torch.from_numpy(np.array(label_vector_mess[train_idx], dtype="float32"))
    xTest = torch.from_numpy(np.array(BP_ratio_data_mess[test_idx], dtype="float32"))
    yTest = torch.from_numpy(np.array(label_vector_mess[test_idx], dtype="float32"))
    class Net(nn.Module):
        def __init__(self,nInput,nHidden,nOutput):
            super(Net,self).__init__()
            self.hidden1 = nn.Linear(nInput,nHidden)  
            self.relu1 = nn.ReLU()                    
            self.hidden2 = nn.Linear(nHidden,nHidden) 
            self.relu2 = nn.ReLU()                    
            self.pre = nn.Linear(nHidden,nOutput)
        def forward(self,x):
            out = self.hidden1(x)
            out = self.relu1(out)
            out = self.hidden2(out)
            out = self.relu2(out)
            out = self.pre(out)
            return F.sigmoid(out)
    net=Net(20,30,1)         
    
    #Define the optimizer and the loss function
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01) 
    loss = F.mse_loss      
    epoch = 25000  
    for i in range(epoch):
        yPre=net(xTrain)
        lossValue = loss(yPre,yTrain)  
        optimizer.zero_grad()     #Parameter gradient return to zero, this step cannot be omitted
        lossValue.backward()     
        optimizer.step()
        if(i+1)%1000 == 0:
            print(f"{i+1}：{lossValue}")
    #Model evaluation on the test set
    print("-"*60)
    yPre = net(xTest)   #Gets the prediction label value
    yPre_numpy = yPre.detach().numpy()  # change tensor to numpy
    yPre_round = np.round(yPre_numpy) 
    yTest_numpy = yTest.numpy()  
    acc = sum(yPre_round == yTest_numpy) / len(yTest)  
    acc_list.append(acc)  
    print(fold_idx+1, "step，accuracy on testing dataset", acc)
  
    # yTest_numpy is the classifier's prediction, and yPre_numpy is the real label
    
    fpr, tpr, thresholds = roc_curve(yTest_numpy, yPre_numpy)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(yTest_numpy, yPre_numpy)
    prc_auc = auc(recall, precision)

    roc_data.append((fpr, tpr, roc_auc))
    prc_data.append((recall, precision, prc_auc))


avg_acc = sum(acc_list) / len(acc_list)
print("Average accuracy", avg_acc)
    
#%%Markov model

fpr_markov, tpr_markov, thresholds_markov = roc_curve(label_vector_mess, markov_interval10_mess)
roc_auc_markov = auc(fpr_markov, tpr_markov)
# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_markov, tpr_markov, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_markov)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

#%%
precision_markov, recall_markov, thresholds_markov = precision_recall_curve(label_vector_mess, markov_interval10_mess)
prc_auc_markov = auc(recall_markov, precision_markov)

plt.figure(figsize=(8, 6))
plt.plot(recall_markov, precision_markov, color='dodgerblue', lw=2, label='PRC curve (area = %0.2f)' % prc_auc_markov)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()

#%% SVM
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)


roc_data_svm = []
prc_data_svm = []
acc_list_svm = []  


for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 
    xTrain_svm = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_svm = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_svm  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_svm  = np.array(label_vector_mess[test_idx], dtype="float32")
    
    #Model training
    svm = LinearSVC()
    svm.fit(xTrain_svm, yTrain_svm)
    
    yPred_prob_svm = svm.decision_function(xTest_svm)
    yPred_prob_svm_norm = (yPred_prob_svm - yPred_prob_svm.min()) / (yPred_prob_svm.max() - yPred_prob_svm.min())
    
    acc_svm = accuracy_score(yTest_svm, yPred_prob_svm_norm.round())
    acc_list_svm.append(acc_svm) 
    print(fold_idx+1, "step,SVM accuracy on testing dataset:", acc_svm)
  
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(yTest_svm, yPred_prob_svm_norm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    precision_svm, recall_svm, thresholds_svm = precision_recall_curve(yTest_svm, yPred_prob_svm_norm)
    prc_auc_svm = auc(recall_svm, precision_svm)

  
    roc_data_svm.append((fpr_svm, tpr_svm, roc_auc_svm))
    prc_data_svm.append((recall_svm, precision_svm, prc_auc_svm))

avg_acc_svm = sum(acc_list_svm) / len(acc_list_svm)  
print("SVM average accuracy", avg_acc_svm)



#%%Random Forest
from sklearn.metrics import classification_report       
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import StratifiedKFold


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

roc_data_rf = []
prc_data_rf = []
acc_list_rf = []  


for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
  
    xTrain_rf = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_rf = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_rf  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_rf  = np.array(label_vector_mess[test_idx], dtype="float32")
   
    rf = RandomForestClassifier(n_estimators=100)  
    rf.fit(xTrain_rf,yTrain_rf)     
   
    yPred_rf = rf.predict_proba(xTest_rf)[:, 1]
    
    #yPred_rf = rf.predict(xTest_rf)
    print("The classification accuracy of the random forest model on the test set is", rf.score(xTest_rf, yTest_rf))
    print("Other evaluation indicators：\n", classification_report(yTest_rf, yPred_rf.round()))
    acc_rf = accuracy_score(yTest_rf, yPred_rf.round())
    acc_list_rf.append(acc_rf)  
    print(fold_idx+1, "step，RF accuracy on testing dataset", acc_rf)
  

    fpr_rf, tpr_rf, thresholds_rf = roc_curve(yTest_rf , yPred_rf )
    roc_auc_rf = auc(fpr_rf , tpr_rf )
    precision_rf , recall_rf , thresholds_rf  = precision_recall_curve(yTest_rf , yPred_rf )
    prc_auc_rf  = auc(recall_rf , precision_rf )

    
    roc_data_rf.append((fpr_rf , tpr_rf , roc_auc_rf ))
    prc_data_rf.append((recall_rf , precision_rf , prc_auc_rf ))


avg_acc_rf  = sum(acc_list_rf ) / len(acc_list_rf )  
print("The average accuracy of the random forest model on the test set is", avg_acc_rf )

#%%
# 
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
plt.figure(figsize=(16, 6))
#BP avg
mean_tpr = 0.0
mean_auc = 0.0
for fpr, tpr, roc_auc in roc_data:
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_auc += roc_auc
mean_tpr /= len(roc_data)
mean_auc /= len(roc_data)

#SVM avg
mean_tpr_svm = 0.0
mean_auc_svm = 0.0
for fpr_svm, tpr_svm, roc_auc_svm in roc_data_svm:
    mean_tpr_svm += np.interp(mean_fpr, fpr_svm, tpr_svm)
    mean_auc_svm += roc_auc_svm
mean_tpr_svm /= len(roc_data_svm)
mean_auc_svm /= len(roc_data_svm)


#RF avg
mean_tpr_rf = 0.0
mean_auc_rf = 0.0
for fpr_rf, tpr_rf, roc_auc_rf in roc_data_rf:
    mean_tpr_rf += np.interp(mean_fpr, fpr_rf, tpr_rf)
    mean_auc_rf += roc_auc_rf
mean_tpr_rf /= len(roc_data_rf)
mean_auc_rf /= len(roc_data_rf)


plt.subplot(1, 2, 1)
plt.plot(fpr_markov, tpr_markov, color='dodgerblue', lw=1.5, label='Markov ROC curve (area = %0.2f)' % roc_auc_markov)
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=1.5, label='BP combined with Markov ROC curve (area = %0.2f)' % mean_auc)
plt.plot(mean_fpr, mean_tpr_svm, color='limegreen', lw=1.5, label='SVM combined with Markov ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot(mean_fpr, mean_tpr_rf, color='tomato', lw=1.5, label='RandomForest combined with Markov ROC curve (area = %0.2f)' % roc_auc_rf)
#plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")


#PRC
#BP PRC avg
mean_precision = 0.0
mean_ap = 0.0
for recall, precision, ap in prc_data:
    mean_precision += np.interp(mean_recall[::-1], recall[::-1], precision[::-1])
    mean_ap += ap
mean_precision /= len(prc_data)
mean_ap /= len(prc_data)

#SVM PRC avg
mean_precision_svm = 0.0
mean_ap_svm = 0.0
for recall_svm, precision_svm, ap_svm in prc_data_svm:
    mean_precision_svm += np.interp(mean_recall[::-1], recall_svm[::-1], precision_svm[::-1])
    mean_ap_svm += ap_svm
mean_precision_svm /= len(prc_data_svm)
mean_ap_svm /= len(prc_data_svm)

#RF PRC avg
mean_precision_rf = 0.0
mean_ap_rf = 0.0
for recall_rf, precision_rf, ap_rf in prc_data_rf:
    mean_precision_rf += np.interp(mean_recall[::-1], recall_rf[::-1], precision_rf[::-1])
    mean_ap_rf += ap_rf
mean_precision_rf /= len(prc_data_rf)
mean_ap_rf /= len(prc_data_rf)

plt.subplot(1, 2, 2)
plt.plot(recall_markov, precision_markov, color='dodgerblue', lw=2, label='Markov PRC curve (area = %0.2f)' % prc_auc_markov)
plt.plot(mean_recall[::-1], mean_precision, color='darkorange', lw=2, label='BP combined with Markov PRC curve (area = %0.2f)' % mean_ap)
plt.plot(mean_recall[::-1], mean_precision_svm, color='limegreen', lw=2, label='SVM combined with Markov PRC curve (area = %0.2f)' % mean_ap_svm)
plt.plot(mean_recall[::-1], mean_precision_rf, color='tomato', lw=2, label='RandomForset combined with Markov PRC curve (area = %0.2f)' % mean_ap_rf)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()



