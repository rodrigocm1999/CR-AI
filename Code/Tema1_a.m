clear all;

imgsResolution = 18;
[imageInputs,imageTargets] = readyImages('Folder1', imgsResolution, '%d','jpg');

net = feedforwardnet([10]);
% Usar todas as imagens para treinar
net.divideFcn = '';
%Settings a alterar
%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
%net.trainFcn = 'trainrp';
%net.trainFcn = 'traingdx';

% net.layers{1}.transferFcn = 'logsig';
% net.layers{2}.transferFcn = 'purelin';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'logsig';


net.trainParam.epochs = 100;

% Treinar
[net, trainResult] = train(net, imageInputs, imageTargets);

% Simular
output = sim(net, imageInputs);

% plotconfusion(imageTargets, output) % Matriz de confusao

accuracy =  testNetworkAccuracy(output,imageTargets);
fprintf('Precisao total -> %f\n', accuracy)

