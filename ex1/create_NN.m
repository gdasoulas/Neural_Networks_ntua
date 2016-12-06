function [mlp_net,accuracy,F_score]=create_NN(TrainData,TrainDataTargets,...
     TestData,TestDataTargets,iters,layers,trainfunc,learnfunc,transfer_fcn)
    for k=1:iters
        mlp_net= newff(TrainData,TrainDataTargets,layers,{},trainfunc);
        mlp_net.divideParam.trainRatio=0.8;
        mlp_net.divideParam.valRatio=0.2;
        mlp_net.divideParam.testRatio=0;
        mlp_net.trainParam.epochs=1000;
        mlp_net.trainParam.showWindow=true;
        mlp_net.layers{2}.transferFcn=transfer_fcn;

        mlp_net=train(mlp_net,TrainData,TrainDataTargets);

        TestDataOutput=sim(mlp_net,TestData);
        [acc(k),precision,recall]=eval_Accuracy_Precision_Recall(TestDataOutput,TestDataTargets);

        % good measure for combining precise and recall -> F-score - harmonic mean 

        Fsc(k,:) = harmmean([precision recall],2);
    end
    accuracy=mean(acc);
    F_score=mean(Fsc,1);
end