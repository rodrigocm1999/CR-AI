clear all;

imgsResolution = 16;

[imageInputs,imageTargets] = readyImages('Datasets/Folder2', imgsResolution, 'letter_bnw_%d.jpg', 2);
%[imageInputs,imageTargets] = readyImages('Datasets/Folder1', imgsResolution, '%d.jpg', 1);

net = feedforwardnet([ 30 24 ]);

%net.trainFcn = 'traingd';
%net.trainFcn = 'trainbfg';
net.trainParam.epochs = 500;

%net.layers{1}.transferFcn = 'logsig';
%net.layers{2}.transferFcn = 'purelin';
%net.layers{1}.transferFcn = 'tansig';
%net.layers{2}.transferFcn = 'logsig';

% Usar todas as imagens para treinar
net.divideFcn = '';
% https://www.mathworks.com/help/deeplearning/ug/divide-data-for-optimal-neural-network-training.html


net.input.processFcns = {'mapminmax'};
net.output.processFcns = {'mapminmax'};

net = configure(net,imageInputs,imageTargets);

imageInputsGPU = nndata2gpu(imageInputs);
imageTargetsGPU = nndata2gpu(imageTargets);

% Treinar
[net,trainResult] = train(net, imageInputsGPU, imageTargetsGPU, 'useGPU' , 'only');
disp(trainResult)
% Simular
outputGPU = net(imageInputsGPU);% Execute on GPU             
output = gpu2nndata(outputGPU);    

plotconfusion(imageTargets, output) % Matriz de confusao

accuracy = testNetworkAccuracy(output,imageTargets,size(trainResult.trainInd,2));
fprintf('Precisao total %f\n', accuracy)


