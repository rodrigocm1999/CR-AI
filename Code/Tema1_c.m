clear variables;

imgsResolution = 24;
[imageInputs,imageTargets] = readyImages('Folder2', imgsResolution, 'letter_bnw_%d','jpg', 1);

% TODO read networks from Tema1_b.m
net = feedforwardnet([ 20 ]);

% Usar todas as imagens para treinar
net.divideFcn = '';

net.trainFcn = 'traingdx';
net.trainParam.epochs = 10000;

% Treinar
[net,trainResult] = train(net, imageInputs, imageTargets);
% Simular
output = sim(net, imageInputs);

%VISUALIZAR DESEMPENHO
% plotconfusion(imageTargets, output) % Matriz de confusao
% plotperf(trainResult)         % Grafico com o desempenho da rede nos 3 conjuntos  

accuracy = testNetworkAccuracy(output,imageTargets);
fprintf('Precisao total %f\n', accuracy)


[testInput,testTargets] = readyImages('Folder3', imgsResolution, 'letter_bnw_test_%d','jpg');
% [testInput,testTargets] = readyImages('Folder1', imgsResolution, '%d','jpg');

testOutput = sim(net, testInput);

% plotconfusion(testTargets, testOutput);

accuracy = testNetworkAccuracy(testOutput,testTargets);
fprintf('Precisao teste %f\n', accuracy)




