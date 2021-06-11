clear all;

imgsResolution = 16;

[imageInputs,imageTargets] = readyImages('Folder3', imgsResolution, 'letter_bnw_test_%d','jpg', 1);

% TODO read networks from Tema1_b.m
net = feedforwardnet([ 10 ]);

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

accuracy = testNetworkAccuracy(output,imageTargets);
fprintf('Precisao total %f\n', accuracy)


[testInput,testTargets] = readyImages('Datasets/Folder3', imgsResolution, 'letter_bnw_test_%d.jpg', 2);

testOutput = sim(net, testInput);

plotconfusion(testTargets, testOutput);

accuracy = testNetworkAccuracy(testOutput,testTargets);
fprintf('Precisao teste %f\n', accuracy)




