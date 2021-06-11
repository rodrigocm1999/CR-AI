clear all;

imgsResolution = 16;

[imageInputs,imageTargets] = readyImages('Folder2', imgsResolution, 'letter_bnw_%d','jpg', 2);

net = feedforwardnet([ 20 ]);

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 50;

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';

% Usar todas as imagens para treinar
net.divideFcn = '';
% https://www.mathworks.com/help/deeplearning/ug/divide-data-for-optimal-neural-network-training.html

% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);
disp(trainResult)
% Simular
output = sim(net, imageInputs);
plotconfusion(imageTargets, output) % Matriz de confusao

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)

%TODO save neural network- maybe name it the timestamp


