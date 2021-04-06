clear all;

imgsResolution = 16;
[imageInputs,imageTargets] = readyImages('Datasets/Folder1', imgsResolution, '%d.jpg', 1);

net = feedforwardnet([ 10 ]);

net.trainParam.epochs = 50;
% Usar todas as imagens para treinar
net.divideFcn = '';

% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);
disp(trainResult)
% Simular
output = sim(net, imageInputs);

plotconfusion(imageTargets, output) % Matriz de confusao
plotperf(trainResult)               % Grafico com o desempenho da rede

numberOfElements = size(trainResult.trainInd,2);
accuracy = testNetworkAccuracy(output,imageTargets,numberOfElements);
fprintf('Precisao total %f\n', accuracy)



