function [Class,TestData,TestDataTargets,TrainData,TrainDataTargets]=data_init(TestData,...
    TestDataTargets,TrainData,TrainDataTargets)

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

