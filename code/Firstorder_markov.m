% close all;
% clear all;
% clc;
% tic
% toc
% disp(['运行时间: ',num2str(toc)]);
%%
% 更改当前工作目录
cd('D:\毕业设计-整个优秀\论文程序相关\2023毕设代码');
%% 
% 导入数据
%data_positive = readmatrix('Positive_try.csv');
%data_negetive_all = readmatrix('Negetive_try.csv');
%random_rows = datasample(1:size(data_negetive_all , 1), 2491, 'Replace', false); % 随机抽取2491行
data_negetive = data_negetive_all(random_rows , : );% 构建新的矩阵
%data_positive    是所有正样本，共2491条数据
%Negetive_try_all 是所有负样本，共431597条数据
%data_negetive    是从所有负样本随机选出的2491条数据
Z0=[0,1,2,3];  % A：0 ； C：1 ；G：2 ； T：3
P0=['00'; '01'; '02'; '03'; '10'; '11'; '12'; '13'; '20'; '21'; '22'; '23'; '30'; '31'; '32'; '33'];
%16种 AA    AC    AG    AT    CA    CC    CG    CT    GA    GC    GG    GT    TA    TC    TG    TT
%%
N=10;
Z1_positive = zeros(16,1999,N);    %正样本，对应一阶马尔科夫，状态转移矩阵为4行4列，共16个数据
Z1_negetive = zeros(16,1999,N);    %负样本

%%
% 将数据集分为10个部分
num_folds = 10;
cv_pos = cvpartition(length(data_positive), 'KFold', num_folds);
%%
%采用十折交叉验证/逐一训练和测试模型
for i = 1:num_folds
    % 将数据集分为训练集和测试集
    train_idx = cv_pos.training(i);
    %test_idx = cv_pos.test(i);
    train_data_pos = data_positive(train_idx,:);
    %test_data_pos = data_positive(test_idx,:);
    train_data_neg = data_negetive(train_idx,:);
    %test_data_neg = data_negetive(test_idx,:);
    
    %%
    %一阶应为16*1999的矩阵
    [row,col] = size(train_data_pos);
    P1_positive = zeros(16,1999);
    %%
    %得到正样本的一阶马尔科夫模型状态转移矩阵Probability transition matrix
    for flag1_positive=1:1999       %此处的flag1代表位置，flag=序列长度-1
        in=1;
        for k1=1:4
            for k2=1:4
                P1_positive(in,flag1_positive)=length(find(train_data_pos(:,flag1_positive)==Z0(k1)&train_data_pos(:,flag1_positive+1)==Z0(k2)))./row;     %这是一阶马尔科夫模型
                in=in+1;
                %个数/总行数(总行数就是训练的序列数量），即表示转移概率
            end
        end
    end
    Z1_positive(:,:,i)=P1_positive;
    
    %%
    %一阶应为16*1999的矩阵
    [row2,col2] = size(train_data_neg);
    P1_negetive = zeros(16,1999);    %负样本的一阶转移矩阵
    %%
    %得到负样本样本的一阶马尔科夫模型状态转移矩阵Probability transition matrix
    for flag1_negetive=1:1999       %此处的flag1代表位置，flag=序列长度-1
        in=1;
        for k1=1:4
            for k2=1:4
                P1_negetive(in,flag1_negetive)=length(find(train_data_neg(:,flag1_negetive)==Z0(k1) & train_data_neg(:,flag1_negetive+1)==Z0(k2)))./row2;     %这是一阶马尔科夫模型
                in=in+1;
                %个数/总行数(总行数就是训练的序列数量），即表示转移概率
            end
        end
    end
    Z1_negetive(:,:,i)=P1_negetive;
end
%save result1.mat Z1_positive Z1_negetive 

%%
firstorder_rate_pos = zeros(N,20);
firstorder_rate_neg = zeros(N,20);
for i =1:N
    %正样本测试-一阶马尔科夫不同位点区域进行判断的准确率分布
    %将数据集分为训练集和测试集
    %train_idx = cv_pos.training(i);
    test_idx = cv_pos.test(i);
    %train_data_pos = data_positive(train_idx,:);
    test_data_pos = data_positive(test_idx,:);
    %train_data_neg = data_negetive(train_idx,:);
    test_data_neg = data_negetive(test_idx,:);    
    for k = 0:19
        count_p = 0;
        count_n = 0;
        for all_test_pos = 1:size(test_data_pos)
            test_pos = test_data_pos(all_test_pos,:);
            ji1=1;
            ji2=1;
            for flag1=(1+100*k):(99+100*k)
                zhi1=find(test_pos(flag1)==Z0);
                zhi2=find(test_pos(flag1+1)==Z0);
                zhi=(zhi1-1)*4+zhi2;
                ji1=ji1*Z1_positive(zhi,flag1,i);
                ji2=ji2*Z1_negetive(zhi,flag1,i);
            end
            if ji1==0 &ji2==0
                aaa=rand(1);
            elseif ji1==0 &ji2~=0
                aaa=100000;
            else
                aaa=ji1/ji2;
            end
            if aaa>=1
                %disp(['G-四链体 ']);
                count_p = count_p +1;
            else
                %disp(['非G-四链体 ']);
            end
        end
        firstorder_rate_pos(i,k+1) = count_p./size(test_data_pos,1) ;
        
        for all_test_neg = 1:size(test_data_neg)
            test_neg = test_data_neg(all_test_neg,:);
            ji1=1;
            ji2=1;
            for flag2=(1+100*k):(99+100*k)
                zhi1=find(test_neg(flag2)==Z0);
                zhi2=find(test_neg(flag2+1)==Z0);
                zhi=(zhi1-1)*4+zhi2;
                ji1=ji1*Z1_positive(zhi,flag2,i);
                ji2=ji2*Z1_negetive(zhi,flag2,i);
            end
            if ji1==0 &ji2==0
                aaa=rand(1);
            elseif ji1==0 &ji2~=0
                aaa=100000;
            else
                aaa=ji1/ji2;
            end
            if aaa>=1
                %disp(['G-四链体 ']);
            else
                %disp(['非G-四链体 ']);
                count_n = count_n +1;
            end
        end
        firstorder_rate_neg(i,k+1) = count_n./size(test_data_neg,1) ;
    end
end
%%
%绘制图形- 正样本- 一阶马尔科夫不同位点预测的准确率1-199,201-399,401-599,......,1801-1999
x_1 = 99:100:1999;
y_1 = mean(firstorder_rate_pos); % 对每一列求均值
y_1 = reshape(y_1, 1, []); % 转换为行向量存储
y_2 = mean(firstorder_rate_neg); % 对每一列求均值
y_2 = reshape(y_2, 1, []); % 转换为行向量存储
y_average = (y_1+y_2)/2;
%%
%不用这一段啦，用下面一段绘图
figure;
plot(x_1,y_1,'ko-','linewidth',2,'Markersize',2);
title("Accuracy rating of different site areas in first-order markov",'FontSize',12);
xlim([0 2000]);xticks(0:100:2000);
set(gca,'fontsize',6); 
grid on;
hold on;
plot(x_1,y_2,'bd-','linewidth',2,'Markersize',2);
plot(x_1,y_average,'rp-','linewidth',2,'Markersize',2);
legend("pos-sample","neg-sample","all-sample");
%%
figure;
% 设定颜色
color1 = [0.25 0.25 1];
color2 = [0.93 0.39 0.28];
color3 = [0.47 0.67 0.19];
% 绘制曲线
plot(x_1,y_1,'o-','linewidth',1.5,'Markersize',4,'color',color1);
grid on;
hold on;
plot(x_1,y_2,'d-','linewidth',1.5,'Markersize',4,'color',color2);
plot(x_1,y_average,'p-','linewidth',1.5,'Markersize',4,'color',color3);
% 设置坐标轴范围和标签
xlim([0 2000]);
xticks(0:200:2000);
xlabel('Regions','FontSize',10,'FontName','Arial');
ylabel('Values','FontSize',10,'FontName','Arial');
% 设置图例
legend({'Sn','Sp','ACC'},'Location','best','FontSize',8,'FontName','Arial');
% 美化背景和边框
set(gca,'box','on','linewidth',1.2,'FontName','Arial','FontSize',8,'XColor','k','YColor','k');
% 添加标题
title('Prediction performance of different position regions in first-order markov','FontSize',10,'FontName','Arial')