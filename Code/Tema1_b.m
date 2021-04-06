clear all;

imgsResolution = 16;

[imageInputs,imageTargets] = readyImages('Datasets/Folder2', imgsResolution, '%d.jpg', 2);

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

plotconfusion(imageTargets, output) % Matriz de confusao
plotperf(trainResult)         % Grafico com o desempenho da rede

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)


