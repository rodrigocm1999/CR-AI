
[imageInputs,imageTargets] = readyImages('Datasets greek/train_letters_images');


net = feedforwardnet();

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 500;

net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';



% TREINAR
[net,tr] = train(net, imageInputs, imageTargets);
%view(net);
%disp(tr)
% SIMULAR
output = sim(net, imageInputs);


%VISUALIZAR DESEMPENHO
%plotconfusion(irisTargets, out) % Matriz de confusao
%plotperf(tr)         % Grafico com o desempenho da rede nos 3 conjuntos  

accuracy = testNetworkAccuracy(output,imageTargets,tr.testIndex);
fprintf('Precisao total %f\n', accuracy)


% SIMULAR A REDE APENAS NO CONJUNTO DE TESTE
TInput = imageInputs(:, tr.testInd);
TTargets = imageTargets(:, tr.testInd);

testGroupOutput = sim(net, TInput);

accuracy = testNetworkAccuracy(testGroupOutput,TTargets,tr.testIndex);
fprintf('Precisao teste %f\n', accuracy)




