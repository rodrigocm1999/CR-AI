clear all;

imgsResolution = 16;
[imageInputs,imageTargets] = readyImages('Folder1', imgsResolution, '%d','jpg', 1);

net = feedforwardnet(10);

%Settings a alterar
net.trainParam.epochs = 15;
%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';

% Usar todas as imagens para treinar
net.divideFcn = '';

% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);

disp(trainResult)

% Simular
output = sim(net, imageInputs);

plotconfusion(imageTargets, output) % Matriz de confusao

numberOfElements = size(trainResult.trainInd,2);
accuracy = testNetworkAccuracy(output,imageTargets,numberOfElements);
fprintf('Precisao total %f\n', accuracy)



