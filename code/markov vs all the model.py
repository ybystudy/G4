# -*- coding: utf-8 -*-
"""
Created on Wed May 17 20:20:36 2023

@author: pc
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
#导入归一化之后的ln(Ratio)数据
Bp_ratio_pos = pd.read_csv('D:/毕业设计-整个优秀/论文程序相关/2023毕设代码/norm_BP_pos_feature.csv', header=None, sep=',');
Bp_ratio_neg = pd.read_csv('D:/毕业设计-整个优秀/论文程序相关/2023毕设代码/norm_BP_neg_feature.csv', header=None, sep=',');
#导入归一化之前的ln(Ratio)数据
ln_unnorm_ratio_pos = pd.read_csv('D:/毕业设计-整个优秀/论文程序相关/2023毕设代码/pos_ratio_final_log.csv', header=None, sep=',');
ln_unnorm_ratio_neg = pd.read_csv('D:/毕业设计-整个优秀/论文程序相关/2023毕设代码/neg_ratio_final_log.csv', header=None, sep=',');
#%%将神经网络dataframe数据【已归一化】转为矩阵形式,并合并为一个矩阵
Bp_ratio_posmatrix = np.asarray(Bp_ratio_pos.values);
Bp_ratio_negmatrix = np.asarray(Bp_ratio_neg.values);
Bp_ratio_data = np.concatenate((Bp_ratio_posmatrix, Bp_ratio_negmatrix), axis=0);
#未归一化数据
ln_unnorm_ratio_posmatrix = np.asarray(ln_unnorm_ratio_pos.values);
ln_unnorm_ratio_negmatrix = np.asarray(ln_unnorm_ratio_neg.values);
ln_unnorm_ratio_data = np.concatenate((ln_unnorm_ratio_posmatrix, ln_unnorm_ratio_negmatrix), axis=0);
#%%
#以区间10验证马尔科夫模型数据
markov_interval10 = ln_unnorm_ratio_data[:, 9] # 注意Python的索引从0开始

#%%生成随机标签
label_vector = np.concatenate((np.ones((Bp_ratio_posmatrix.shape[0],1)), np.zeros((Bp_ratio_negmatrix.shape[0],1))))
BP_idx = np.random.permutation(Bp_ratio_data.shape[0])
BP_ratio_data_mess = Bp_ratio_data[BP_idx, :]
label_vector_mess = label_vector[BP_idx]
markov_interval10_mess = markov_interval10[BP_idx]
#%%使用十折交叉验证法划分训练集和测试集

from sklearn.model_selection import StratifiedKFold

# 将数据分成10折
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# 定义存储各折ROC和PRC曲线数据的列表
roc_data = []
prc_data = []
acc_list = []  # 存储每次循环得到的准确率
#
# 开始交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 获取当前折的训练集和测试集
    xTrain = torch.from_numpy(np.array(BP_ratio_data_mess[train_idx], dtype="float32"))
    yTrain = torch.from_numpy(np.array(label_vector_mess[train_idx], dtype="float32"))
    xTest = torch.from_numpy(np.array(BP_ratio_data_mess[test_idx], dtype="float32"))
    yTest = torch.from_numpy(np.array(label_vector_mess[test_idx], dtype="float32"))
    class Net(nn.Module):
        def __init__(self,nInput,nHidden,nOutput):
            super(Net,self).__init__()
            self.hidden1 = nn.Linear(nInput,nHidden)  #隐藏层1
            self.relu1 = nn.ReLU()                    #ReLU激活函数
            self.hidden2 = nn.Linear(nHidden,nHidden) #隐藏层2
            self.relu2 = nn.ReLU()                    #ReLU激活函数
            self.pre = nn.Linear(nHidden,nOutput)
        def forward(self,x):
            out = self.hidden1(x)
            out = self.relu1(out)
            out = self.hidden2(out)
            out = self.relu2(out)
            out = self.pre(out)
            return F.sigmoid(out)
    net=Net(20,30,1)          #实则化神经网络
    #本网络输入层包含20个神经元，输出层包含1个神经元，外加两个隐藏层，分别有20个神经元
    #定义优化器和损失函数
    optimizer=torch.optim.SGD(net.parameters(),lr=0.01)  #net.parameters表示神经网络参数
    #SGD表示随机梯度下降法，是求解损失函数最小的一种优化方法，lr表示学习率
    loss = F.mse_loss       #将损失函数定义为均方误差损失函数
     #损失函数也可以自己根据损失函数的计算公式进行定义
     #步骤4：循环训练
    epoch = 25000  #训练的次数
    for i in range(epoch):
        yPre=net(xTrain)
        lossValue = loss(yPre,yTrain)  #计算损失函数值
        optimizer.zero_grad()     #参数梯度归零，这一步不可省略
        lossValue.backward()      #反向传播
        optimizer.step()
        if(i+1)%1000 == 0:
            print(f"第{i+1}次训练得到损失函数值：{lossValue}")
     #步骤5:测试集上进行模型评价
    print("-"*60)
    yPre = net(xTest)   #获取预测标签值
    yPre_numpy = yPre.detach().numpy()  # 将tensor类型转换为numpy数组类型
    yPre_round = np.round(yPre_numpy)  # 对yPre进行四舍五入，将大于0.5的值变成1，小于0.5的值变成0
    yTest_numpy = yTest.numpy()  # 将tensor类型转换为numpy数组类型
    acc = sum(yPre_round == yTest_numpy) / len(yTest)  # 计算准确率
    acc_list.append(acc)  # 将准确率加入列表中
    print("第", fold_idx+1, "次循环，在测试集上的准确率为", acc)
  
    # yTest_numpy是分类器的预测结果，yPre_numpy是真实标签
    
    fpr, tpr, thresholds = roc_curve(yTest_numpy, yPre_numpy)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(yTest_numpy, yPre_numpy)
    prc_auc = auc(recall, precision)

    # 将当前折的ROC和PRC曲线数据存入列表中
    roc_data.append((fpr, tpr, roc_auc))
    prc_data.append((recall, precision, prc_auc))


avg_acc = sum(acc_list) / len(acc_list)  # 计算平均准确率
print("所有循环结束后，在测试集上的平均准确率为", avg_acc)
    
#%%马尔科夫模型

fpr_markov, tpr_markov, thresholds_markov = roc_curve(label_vector_mess, markov_interval10_mess)
roc_auc_markov = auc(fpr_markov, tpr_markov)
# 绘制ROC曲线
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
#绘制PRC曲线
plt.figure(figsize=(8, 6))
plt.plot(recall_markov, precision_markov, color='dodgerblue', lw=2, label='PRC curve (area = %0.2f)' % prc_auc_markov)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()
#%%
'''#%%支持向量机模型
from sklearn.svm import LinearSVC                      #导入分类器
from sklearn.metrics import classification_report       #用于模型评估
from sklearn.metrics import accuracy_score
# 划分训练集和测试集
from sklearn.model_selection import StratifiedKFold

# 将数据分成10折
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# 定义存储各折ROC和PRC曲线数据的列表
roc_data_svm = []
prc_data_svm = []
acc_list_svm = []  # 存储每次循环得到的准确率
#
# 开始交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 获取当前折的训练集和测试集
    xTrain_svm = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_svm = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_svm  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_svm  = np.array(label_vector_mess[test_idx], dtype="float32")
    # 训练模型
    svm = LinearSVC()
    svm.fit(xTrain_svm,yTrain_svm)
    # 模型评估
    
    yPred_svm = svm.predict(xTest_svm)
    print("支持向量机的精度为",svm.score(xTest_svm,yTest_svm))
    print("更多的评价指标：\n",classification_report(yTest_svm, yPred_svm))
    acc_svm = accuracy_score(yTest_svm, yPred_svm)
    acc_list_svm.append(acc_svm)  # 将准确率加入列表中
    print("第", fold_idx+1, "次循环，支持向量机模型在测试集上的准确率为", acc_svm)
  
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(yTest_svm, yPred_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    precision_svm, recall_svm, thresholds_svm = precision_recall_curve(yTest_svm, yPred_svm)
    prc_auc_svm = auc(recall_svm, precision_svm)

    # 将当前折的ROC和PRC曲线数据存入列表中
    roc_data_svm.append((fpr_svm, tpr_svm, roc_auc_svm))
    prc_data_svm.append((recall_svm, precision_svm, prc_auc_svm))


avg_acc_svm = sum(acc_list_svm) / len(acc_list_svm)  # 计算平均准确率
print("所有循环结束后，支持向量机在测试集上的平均准确率为", avg_acc_svm)
'''
#%% chatgpt的代码
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# 将数据分成10折
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# 定义存储各折ROC和PRC曲线数据的列表
roc_data_svm = []
prc_data_svm = []
acc_list_svm = []  # 存储每次循环得到的准确率

# 开始交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 获取当前折的训练集和测试集
    xTrain_svm = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_svm = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_svm  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_svm  = np.array(label_vector_mess[test_idx], dtype="float32")
    
    # 训练模型
    svm = LinearSVC()
    svm.fit(xTrain_svm, yTrain_svm)
    
    # 模型评估
    yPred_prob_svm = svm.decision_function(xTest_svm)
    yPred_prob_svm_norm = (yPred_prob_svm - yPred_prob_svm.min()) / (yPred_prob_svm.max() - yPred_prob_svm.min())
    
    acc_svm = accuracy_score(yTest_svm, yPred_prob_svm_norm.round())
    acc_list_svm.append(acc_svm)  # 将准确率加入列表中
    print("第", fold_idx+1, "次循环，支持向量机模型在测试集上的准确率为", acc_svm)
  
    fpr_svm, tpr_svm, thresholds_svm = roc_curve(yTest_svm, yPred_prob_svm_norm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    precision_svm, recall_svm, thresholds_svm = precision_recall_curve(yTest_svm, yPred_prob_svm_norm)
    prc_auc_svm = auc(recall_svm, precision_svm)

    # 将当前折的ROC和PRC曲线数据存入列表中
    roc_data_svm.append((fpr_svm, tpr_svm, roc_auc_svm))
    prc_data_svm.append((recall_svm, precision_svm, prc_auc_svm))

avg_acc_svm = sum(acc_list_svm) / len(acc_list_svm)  # 计算平均准确
print("所有循环结束后，支持向量机在测试集上的平均准确率为", avg_acc_svm)



#%%决策树模型
'''
from sklearn.metrics import classification_report       #用于模型评估
from sklearn.metrics import accuracy_score
from sklearn import tree
# 划分训练集和测试集
from sklearn.model_selection import StratifiedKFold

# 将数据分成10折
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# 定义存储各折ROC和PRC曲线数据的列表
roc_data_tree = []
prc_data_tree = []
acc_list_tree = []  # 存储每次循环得到的准确率
#
# 开始交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 获取当前折的训练集和测试集
    xTrain_tree = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_tree = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_tree  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_tree  = np.array(label_vector_mess[test_idx], dtype="float32")
    # 训练模型
    dct = tree.DecisionTreeClassifier(max_depth=8)  #构建决策树
    dct.fit(xTrain_tree,yTrain_tree)     #训练决策树
    # 模型评估
    yPred_tree = dct.predict_proba(xTest_tree)[:, 1]

    #yPred_tree = dct.predict(xTest_tree)
    print("决策树的精度为",dct.score(xTest_tree,yTest_tree))
    print("更多的评价指标：\n",classification_report(yTest_tree, yPred_tree.round()))
    acc_tree = accuracy_score(yTest_tree, yPred_tree.round())

    acc_list_tree.append(acc_tree)  # 将准确率加入列表中
    print("第", fold_idx+1, "次循环，决策树模型在测试集上的准确率为", acc_tree)
  
    
    fpr_tree, tpr_tree, thresholds_tree = roc_curve(yTest_tree, yPred_tree)
    roc_auc_tree = auc(fpr_tree, tpr_tree)
    precision_tree, recall_tree, thresholds_tree = precision_recall_curve(yTest_tree, yPred_tree)
    prc_auc_tree = auc(recall_tree, precision_tree)

    # 将当前折的ROC和PRC曲线数据存入列表中
    roc_data_tree.append((fpr_tree, tpr_tree, roc_auc_tree))
    prc_data_tree.append((recall_tree, precision_tree, prc_auc_tree))


avg_acc_tree = sum(acc_list_tree) / len(acc_list_tree)  # 计算平均准确率
print("所有循环结束后，决策树在测试集上的平均准确率为", avg_acc_tree)
'''
#%%随机森林模型
from sklearn.metrics import classification_report       #用于模型评估
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 划分训练集和测试集
from sklearn.model_selection import StratifiedKFold

# 将数据分成10折
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
# 定义存储各折ROC和PRC曲线数据的列表
roc_data_rf = []
prc_data_rf = []
acc_list_rf = []  # 存储每次循环得到的准确率
#
# 开始交叉验证
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(BP_ratio_data_mess, label_vector_mess)):
    # 获取当前折的训练集和测试集
    xTrain_rf = np.array(BP_ratio_data_mess[train_idx], dtype="float32")
    yTrain_rf = np.array(label_vector_mess[train_idx], dtype="float32")
    xTest_rf  = np.array(BP_ratio_data_mess[test_idx], dtype="float32")
    yTest_rf  = np.array(label_vector_mess[test_idx], dtype="float32")
    # 训练模型
    rf = RandomForestClassifier(n_estimators=100)  #构建随机森林，森林有100个决策树
    rf.fit(xTrain_rf,yTrain_rf)     #训练随机森林
    # 模型评估
    yPred_rf = rf.predict_proba(xTest_rf)[:, 1]
    
    #yPred_rf = rf.predict(xTest_rf)
    print("在测试集上随机森林模型的分类精度为", rf.score(xTest_rf, yTest_rf))
    print("其他评价指标：\n", classification_report(yTest_rf, yPred_rf.round()))
    acc_rf = accuracy_score(yTest_rf, yPred_rf.round())
    acc_list_rf.append(acc_rf)  # 将准确率加入列表中
    print("第", fold_idx+1, "次循环，随机森林模型在测试集上的准确率为", acc_rf)
  
    
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(yTest_rf , yPred_rf )
    roc_auc_rf = auc(fpr_rf , tpr_rf )
    precision_rf , recall_rf , thresholds_rf  = precision_recall_curve(yTest_rf , yPred_rf )
    prc_auc_rf  = auc(recall_rf , precision_rf )

    # 将当前折的ROC和PRC曲线数据存入列表中
    roc_data_rf.append((fpr_rf , tpr_rf , roc_auc_rf ))
    prc_data_rf.append((recall_rf , precision_rf , prc_auc_rf ))


avg_acc_rf  = sum(acc_list_rf ) / len(acc_list_rf )  # 计算平均准确率
print("所有循环结束后，随机森林模型在测试集上的平均准确率为", avg_acc_rf )

#%%
# 绘制平均ROC曲线和PRC曲线
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)
plt.figure(figsize=(16, 6))
#BP神经网络求平均
mean_tpr = 0.0
mean_auc = 0.0
for fpr, tpr, roc_auc in roc_data:
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_auc += roc_auc
mean_tpr /= len(roc_data)
mean_auc /= len(roc_data)

#SVM相应求平均
mean_tpr_svm = 0.0
mean_auc_svm = 0.0
for fpr_svm, tpr_svm, roc_auc_svm in roc_data_svm:
    mean_tpr_svm += np.interp(mean_fpr, fpr_svm, tpr_svm)
    mean_auc_svm += roc_auc_svm
mean_tpr_svm /= len(roc_data_svm)
mean_auc_svm /= len(roc_data_svm)
'''
#Decisiontree相应求平均
mean_tpr_tree = 0.0
mean_auc_tree = 0.0
for fpr_tree, tpr_tree, roc_auc_tree in roc_data_tree:
    mean_tpr_tree += np.interp(mean_fpr, fpr_tree, tpr_tree)
    mean_auc_tree += roc_auc_tree
mean_tpr_tree /= len(roc_data_tree)
mean_auc_tree /= len(roc_data_tree)
'''

#随机森林相应求平均
mean_tpr_rf = 0.0
mean_auc_rf = 0.0
for fpr_rf, tpr_rf, roc_auc_rf in roc_data_rf:
    mean_tpr_rf += np.interp(mean_fpr, fpr_rf, tpr_rf)
    mean_auc_rf += roc_auc_rf
mean_tpr_rf /= len(roc_data_rf)
mean_auc_rf /= len(roc_data_rf)

#绘制平均ROC曲线
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


#绘制平均PRC曲线
#BP神经网络PRC求平均
mean_precision = 0.0
mean_ap = 0.0
for recall, precision, ap in prc_data:
    mean_precision += np.interp(mean_recall[::-1], recall[::-1], precision[::-1])
    mean_ap += ap
mean_precision /= len(prc_data)
mean_ap /= len(prc_data)

#SVM PRC求平均
mean_precision_svm = 0.0
mean_ap_svm = 0.0
for recall_svm, precision_svm, ap_svm in prc_data_svm:
    mean_precision_svm += np.interp(mean_recall[::-1], recall_svm[::-1], precision_svm[::-1])
    mean_ap_svm += ap_svm
mean_precision_svm /= len(prc_data_svm)
mean_ap_svm /= len(prc_data_svm)
'''
#DecisionTree PRC求平均
mean_precision_tree = 0.0
mean_ap_tree = 0.0
for recall_tree, precision_tree, ap_tree in prc_data_tree:
    mean_precision_tree += np.interp(mean_recall[::-1], recall_tree[::-1], precision_tree[::-1])
    mean_ap_tree += ap_tree
mean_precision_tree /= len(prc_data_tree)
mean_ap_tree /= len(prc_data_tree)
'''
#RandomForest PRC求平均
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



