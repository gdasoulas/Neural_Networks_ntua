
clear all
clc
    
%% Bhma 1o 

load dataSet.mat;

[Class,TestData,TestDataTargets,TrainData,TrainDataTargets] = data_init(TestData,...
    TestDataTargets,TrainData,TrainDataTargets);

clear i j ans rand_indices classes_count final_indices population_class

%% Bhma 2 
% Preprocessing 

[TrainData,PS] = removeconstantrows(TrainData);
TestData = removeconstantrows('apply',TestData,PS);

[TrainData,PS] = mapstd(TrainData);     % comment it for no regularization
TestData = mapstd('apply',TestData,PS); % comment it for no regularization

[TrainData,PS] = processpca(TrainData,0.0095);
TestData = processpca('apply',TestData,PS);

clear PS

%% Bhma 3,4,5

for i=5:5:30
    for j=0:5:30
        if j==0
            layers=[i];
        else
            layers=[i j];
        end
        [~,acc,Fsc]=create_NN(TrainData,TrainDataTargets,TestData,TestDataTargets,10,layers,...
                'trainlm','learngdm','purelin');
        accuracy(i/5,j/5+1)=acc;
        F_score(i/5,j/5+1,:)=Fsc;
    end
end

clear recall precision i TestDataOutput mlp_net Fsc acc j k


%% Bhma 6a - Transfer Functions in output Layer
%BEST : TRAINLM -> 2 hidden layers : 20 - 15

TransferFcn={'hardlim','tansig','logsig','purelin'};
for i=1:size(TransferFcn,2)
[~,acc,Fscore]=create_NN(TrainData,TrainDataTargets,TestData,TestDataTargets,10,[20 15],...
    'trainlm','learngdm',TransferFcn{i});
save(strcat('accuracy_',TransferFcn{i},'.mat'),'acc');
save(strcat('F_score_',TransferFcn{i},'.mat'),'Fscore');
end

clear mlp acc Fscore TransferFcn i

%% Bhma 6b - Learning Functions 
%BEST : TRAINLM -> 2 hidden layers : 20 - 15


LearnFcn={'learngdm','learngd'};
for i=1:size(LearnFcn,2)
[~,acc,Fscore]=create_NN(TrainData,TrainDataTargets,TestData,TestDataTargets,20,[20 15],...
    'trainlm',LearnFcn{i},[]);
save(strcat('accuracy_',LearnFcn{i},'.mat'),'acc');
save(strcat('F_score_',LearnFcn{i},'.mat'),'Fscore');
end

clear mlp acc Fscore LearnFcn i

%% Bhma 6c - Validation_Set

[~,acc_val,Fscore_val]=create_NN(TrainData,TrainDataTargets,TestData,TestDataTargets,1,[20 15],...
    'trainlm','learngdm',[]);
% save('accuracy_without_val.mat','acc_val');
% save('F_score_without_val.mat','Fscore_val');


%% PLOTS

bar([5:5:30],acclm.accuracy);
title('Accuracy according to neurons for two hidden layers'); 
xlabel('First layer');
ylabel('Accuracy');
legend('0','5','10','15','20','25','30');

