clear all;

[imageInputs,imageTargets] = readyImages('Datasets greek/train_high_resolution', 8);

net = feedforwardnet([ 10 ]);

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 100;

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';

% TODOS OS EXEMPLOS DE INPUT SAO USADOS NO TREINO % de usar treino atcho eu
%net.divideFcn = '';


% TREINAR
[net,trainResult] = train(net, imageInputs, imageTargets);
%view(net);
disp(trainResult)
% SIMULAR
output = sim(net, imageInputs);

%VISUALIZAR DESEMPENHO
plotconfusion(imageTargets, output) % Matriz de confusao
%plotperf(trainResult)         % Grafico com o desempenho da rede nos 3 conjuntos  

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)


% SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
testInput = imageInputs(:, trainResult.testInd);
testTargets = imageTargets(:, trainResult.testInd);

testGroupOutput = sim(net, testInput);

accuracy = testNetworkAccuracy(testGroupOutput,testTargets,size(trainResult.testInd,2));
fprintf('Precisao teste %f\n', accuracy)




