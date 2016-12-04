%% Bhma 1o 

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

% [TrainData,PS] = mapstd(TrainData);
% TestData = mapstd('apply',TestData,PS);

[TrainData,PS] = processpca(TrainData,0.00047);
TestData = processpca('apply',TestData,PS);

clear PS

%% Bhma 3 4 

for i=5:5:30
    for j=0:5:30
        for k=1:10
            if j==0
                layers=[i];
            else
                layers=[i j];
            end
            mlp_net= newff(TrainData,TrainDataTargets,layers,{},'trainlm');
            mlp_nt.divideParam.trainRatio=0.8;
            mlp_net.divideParam.valRatio=0.2;
            mlp_net.divideParam.testRatio=0;
            mlp_net.trainParam.epochs=1000;
            mlp_net.trainParam.showWindow=false;

            mlp_net=train(mlp_net,TrainData,TrainDataTargets);

            TestDataOutput=sim(mlp_net,TestData);
            [acc(k),precision,recall]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

            % good measure for combining precise and recall -> F-score - harmonic mean 

            Fsc(k,:) = harmmean([precision recall],2);
        end
        accuracy(i/5,j/5+1)=mean(acc)
        F_score(i/5,j/5+1,:)=mean(Fsc,1);
    end
end

clear recall precision i TestDataOutput mlp_net Fsc acc j k

%% Saving mat files

save('F_score_trainlm_withoutstd.mat','F_score');
save('accuracy_trainlm_withoutstd.mat','accuracy');

%%

bar([5:5:30],acclm.accuracy);
title('Accuracy according to neurons for two hidden layers'); 
xlabel('First layer');
ylabel('Accuracy');
legend('0','5','10','15','20','25','30');



%% BEST : TRAINLM -> 2 hidden layers : 20 - 15


