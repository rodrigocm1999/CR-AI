clear all;

imgsResolution = 16;

[imageInputs,imageTargets] = readyImages('Datasets/Folder3', imgsResolution, 'letter_bnw_test_%d.jpg', 1);

net = feedforwardnet([ 10 ]);

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 50;

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';

% Usar todas as imagens para treinar
net.divideFcn = '';


% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);
disp(trainResult)
% Simular
output = sim(net, imageInputs);

%VISUALIZAR DESEMPENHO
%plotconfusion(imageTargets, output) % Matriz de confusao
%plotperf(trainResult)         % Grafico com o desempenho da rede nos 3 conjuntos  

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)


% SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
%testInput = imageInputs(:, trainResult.testInd);
%testTargets = imageTargets(:, trainResult.testInd);

[testInput,testTargets] = readyImages('Datasets/Folder3', imgsResolution, 'letter_bnw_test_%d.jpg', 2);

testOutput = sim(net, testInput);

plotconfusion(testTargets, testOutput);

accuracy = testNetworkAccuracy(testOutput,testTargets,size(testTargets,2));
fprintf('Precisao teste %f\n', accuracy)




