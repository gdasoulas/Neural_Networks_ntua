%% Bhma 1 

clear all

load dataSet.mat;
classes_count=sum(TrainDataTargets,2);
% bar(classes_count,'stacked');
% title('Population of classes');

rand_indices = randperm(size(TrainDataTargets,2));
TrainDataTargets=TrainDataTargets(:, rand_indices); %randomly shuffle columns
TrainData = TrainData(:,rand_indices);

% we want to keep 107 samples for every class

Class(:,1) = [ 0;0;0;0;1];
Class(:,2) = [ 0;0;0;1;0];
Class(:,3) = [ 0;0;1;0;0];
Class(:,4) = [ 0;1;0;0;0];
Class(:,5) = [ 1;0;0;0;0];

population_class = zeros(5);
final_indices=[];
for j=1:size(TrainData,2)
    for i=1:size(Class,2)
        if Class(:,i)==TrainDataTargets(:,j) & population_class(i)<107
            final_indices = [final_indices j];
            population_class(i)=population_class(i)+1;
        end
    end
end

TrainData = TrainData(:,final_indices);
TrainDataTargets = TrainDataTargets(:,final_indices);


clear i j ans rand_indices classes_count final_indices population_class

%% Bhma 2 
% Preprocessing 

[TrainData,PS] = removeconstantrows(TrainData);
TestData = removeconstantrows('apply',TestData,PS);

[TrainData,PS] = mapstd(TrainData);
TestData = mapstd('apply',TestData,PS);

[TrainData,PS] = processpca(TrainData,0.0097);
TestData = processpca('apply',TestData,PS);

clear PS
%% Bhma 3
% mlp neural network

mlp_net= newff(TrainData,TrainDataTargets,[10 15]);
mlp_net.divideParam.trainRatio=0.8;
mlp_net.divideParam.valRatio=0.2;
mlp_net.divideParam.testRatio=0;
mlp_net.trainParam.epochs=1000;

mlp_net=train(mlp_net,TrainData,TrainDataTargets);

TestDataOutput=sim(mlp_net,TestData);
[accuracy,precision,recall]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

% good measure for combining precise and recall -> F-score - harmonic mean 

F_score = harmmean([precision recall],2);

%% Bhma 4 

F_score = zeros(6,size(Class,1));

for i=5:5:30
    mlp_net= newff(TrainData,TrainDataTargets,i,{},'traingd');
    mlp_net.divideParam.trainRatio=0.8;
    mlp_net.divideParam.valRatio=0.2;
    mlp_net.divideParam.testRatio=0;
    mlp_net.trainParam.epochs=1000;

    mlp_net=train(mlp_net,TrainData,TrainDataTargets);

    TestDataOutput=sim(mlp_net,TestData);
    [accuracy(i/5),precision,recall]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

    % good measure for combining precise and recall -> F-score - harmonic mean 

    F_score(i/5,:) = harmmean([precision recall],2);
end

F_score = F_score';

scatter([5:5:30],accuracy,'filled','red');
title('Accuracy according to neurons for one hidden layer'); 
xlabel('Neurons');
ylabel('Accuracy');
clear recall precision i TestDataOutput mlp_net

%% Trying with 2 hidden layers , and neurons in first layer=30

F_score = zeros(6,size(Class,1));

for i=5:5:30
    for j=5:5:30
        mlp_net= newff(TrainData,TrainDataTargets,[i j]);
        mlp_net.divideParam.trainRatio=0.8;
        mlp_net.divideParam.valRatio=0.2;
        mlp_net.divideParam.testRatio=0;
        mlp_net.trainParam.epochs=1000;

        mlp_net=train(mlp_net,TrainData,TrainDataTargets);

        TestDataOutput=sim(mlp_net,TestData);
        [accuracy(i/5),precision,recall]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

        % good measure for combining precise and recall -> F-score - harmonic mean 

        F_score(i/5,:) = harmmean([precision recall],2);
    end
end

F_score = F_score';

scatter([5:5:30],accuracy,'filled','red');
title('Accuracy according to neurons for two hidden layer r'); 
xlabel('Neurons');
ylabel('Accuracy');
clear recall precision i mlp_net
