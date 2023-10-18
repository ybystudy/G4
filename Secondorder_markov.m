
%% 
% import dataset
data_positive = readmatrix('CATmp.g4seqFirst.F0.1.ex1000.seq.csv');
data_negetive_all = readmatrix('CATmp_v.g4seqFirst.F0.1.ex1000.seq.csv');
random_rows = datasample(1:size(data_negetive_all , 1), 2491, 'Replace', false); 
data_negetive = data_negetive_all(random_rows , : );
Z0=[0,1,2,3];  % A：0 ； C：1 ；G：2 ； T：3
P0=['00'; '01'; '02'; '03'; '10'; '11'; '12'; '13'; '20'; '21'; '22'; '23'; '30'; '31'; '32'; '33'];
%AA    AC    AG    AT    CA    CC    CG    CT    GA    GC    GG    GT    TA    TC    TG    TT
N=10;
Z2_positive = zeros(64,1998,N);    %pos sample 16*4 transition matrix
Z2_negetive = zeros(64,1998,N);    %neg sample

%%
%Divide the dataset into 10 sections
num_folds = 10;
cv_pos = cvpartition(length(data_positive), 'KFold', num_folds);
%%
%Ten fold cross-validation/Model training and testing
for i = 1:num_folds
    
    train_idx = cv_pos.training(i);
    %test_idx = cv_pos.test(i);
    train_data_pos = data_positive(train_idx,:);
    %test_data_pos = data_positive(test_idx,:);
    train_data_neg = data_negetive(train_idx,:);
    %test_data_neg = data_negetive(test_idx,:);
    
    %%
    % Second-order Markov matrix of pos
    [row,col] = size(train_data_pos);
    P2_positive = zeros(64,1998);
    %%
    %%Probability transition matrix of pos samples
    for flag2_positive=1:1998      
        in=1;
        for k1=1:4
            for k2=1:4
                for k3=1:4
                    P2_positive(in,flag2_positive)=length(find(train_data_pos(:,flag2_positive)==Z0(k1)&train_data_pos(:,flag2_positive+1)==Z0(k2)&train_data_pos(:,flag2_positive+2)==Z0(k3)))./row;     %这是正样本二阶马尔科夫模型
                    in=in+1;
                     
                end
            end
        end
    end
    Z2_positive(:,:,i)=P2_positive;
    
    %%
    %Second-order Markov matrix of neg
    [row2,col2] = size(train_data_neg);
    P1_negetive = zeros(64,1998);    
    %%
    
    for flag2_negetive=1:1998       
        in=1;
        for k1=1:4
            for k2=1:4
                for k3=1:4
                    P2_negetive(in,flag2_negetive)=length(find(train_data_neg(:,flag2_negetive)==Z0(k1)&train_data_neg(:,flag2_negetive+1)==Z0(k2)&train_data_neg(:,flag2_negetive+2)==Z0(k3)))./row2;     %这是二阶马尔科夫模型
                    in=in+1;
                    
                end
            end
        end
    end
    Z2_negetive(:,:,i)=P2_negetive;
end
%save result1.mat Z2_positive Z2_negetive 

%%
secondorder_rate_pos = zeros(N,20);
secondorder_rate_neg = zeros(N,20);
for i =1:N
   
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
            for flag1=(1+100*k):(98+100*k)
                zhi1=find(test_pos(flag1)==Z0);
                zhi2=find(test_pos(flag1+1)==Z0);
                zhi3=find(test_pos(flag1+2)==Z0);
                zhi=(zhi1-1)*16+(zhi2-1)*4+zhi3;
                ji1=ji1*Z2_positive(zhi,flag1,i);
                ji2=ji2*Z2_negetive(zhi,flag1,i);
            end
            if ji1==0 &ji2==0
                aaa=rand(1);
            elseif ji1==0 &ji2~=0
                aaa=100000;
            else
                aaa=ji1/ji2;
            end
            if aaa>=1
                %disp(['Non-G4']);
                count_p = count_p +1;
            else
                %disp(['G4']);
            end
        end
        secondorder_rate_pos(i,k+1) = count_p./size(test_data_pos,1) ;
        
        for all_test_neg = 1:size(test_data_neg)
            test_neg = test_data_neg(all_test_neg,:);
            ji1=1;
            ji2=1;
            for flag2=(1+100*k):(98+100*k)
                zhi1=find(test_neg(flag2)==Z0);
                zhi2=find(test_neg(flag2+1)==Z0);
                zhi3=find(test_neg(flag2+2)==Z0);
                zhi=(zhi1-1)*16+(zhi2-1)*4+zhi3;
                ji1=ji1*Z2_positive(zhi,flag2,i);
                ji2=ji2*Z2_negetive(zhi,flag2,i);
            end
            if ji1==0 &ji2==0
                aaa=rand(1);
            elseif ji1==0 &ji2~=0
                aaa=100000;
            else
                aaa=ji1/ji2;
            end
            if aaa>=1
                %disp(['Non-G4 ']);
            else
                %disp(['G4']);
                count_n = count_n +1;
            end
        end
        secondorder_rate_neg(i,k+1) = count_n./size(test_data_neg,1) ;
    end
end
%%
%1-199,201-399,401-599,......,1801-1999
x_1 = 98:100:1998;
y_3 = mean(secondorder_rate_pos); 
y_3 = reshape(y_3, 1, []);
y_4 = mean(secondorder_rate_neg); 
y_4 = reshape(y_4, 1, []); 
y_average2 = (y_3+y_4)/2;

%%
figure;
color1 = [0.25 0.25 1];
color2 = [0.93 0.39 0.28];
color3 = [0.47 0.67 0.19];
plot(x_1,y_3,'o-','linewidth',1.5,'Markersize',4,'color',color1);
grid on;
hold on;
plot(x_1,y_4,'d-','linewidth',1.5,'Markersize',4,'color',color2);
plot(x_1,y_average2,'p-','linewidth',1.5,'Markersize',4,'color',color3);
xlim([0 2000]);
xticks(0:200:2000);
xlabel('Regions','FontSize',10,'FontName','Arial');
ylabel('Values','FontSize',10,'FontName','Arial');
legend({'Sn','Sp','ACC'},'Location','best','FontSize',8,'FontName','Arial');
set(gca,'box','on','linewidth',1.2,'FontName','Arial','FontSize',8,'XColor','k','YColor','k');
title('Prediction performance of different position regions in second-order markov','FontSize',10,'FontName','Arial')