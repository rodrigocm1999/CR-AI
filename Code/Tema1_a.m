clear all;

imgsResolution = 11;

[imageInputs,imageTargets] = readyImages('Datasets greek/train_high_resolution', imgsResolution, 'letter_bnw_%d.jpg', 1);

net = feedforwardnet([ 30 24 ]);

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 50;

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';

% TODOS OS EXEMPLOS DE INPUT SAO USADOS NO TREINO % de usar treino atcho eu
net.divideFcn = '';


% TREINAR
[net,trainResult] = train(net, imageInputs, imageTargets);
%view(net);
disp(trainResult)
% SIMULAR
output = sim(net, imageInputs);

%VISUALIZAR DESEMPENHO
%plotconfusion(imageTargets, output) % Matriz de confusao
%plotperf(trainResult)         % Grafico com o desempenho da rede nos 3 conjuntos  

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)


% SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
%testInput = imageInputs(:, trainResult.testInd);
%testTargets = imageTargets(:, trainResult.testInd);

[testInput,testTargets] = readyImages('Datasets greek/test_high_resolution', imgsResolution, 'letter_bnw_test_%d.jpg', 2);

testOutput = sim(net, testInput);

plotconfusion(testTargets, testOutput);

accuracy = testNetworkAccuracy(testOutput,testTargets,size(testTargets,2));
fprintf('Precisao teste %f\n', accuracy)




