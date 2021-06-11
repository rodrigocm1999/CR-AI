clear all;

imgsResolution = 16;
[imageInputs,imageTargets] = readyImages('Folder1', imgsResolution, '%d','jpg', 1);

net = feedforwardnet(10);
% Usar todas as imagens para treinar
net.divideFcn = '';

%Settings a alterar
net.trainParam.epochs = 15;
%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';

% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);

disp(trainResult)

% Simular
output = sim(net, imageInputs);

plotconfusion(imageTargets, output) % Matriz de confusao

accuracy =  testNetworkAccuracy(output,imageTargets);
fprintf('Precisao total -> %f\n', accuracy)

