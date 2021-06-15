networkFolder = 'redes guardadas/networks/';
networkId = '13';
load([networkFolder networkId '.mat']);

% Load Images -------------------------------------------------------------
imgsResolution = 12; % Tamanho ideal, pois tem o minimo de informação sem perda de "detalhes"
[folder1Inputs,folder1Targets] = readyImages('Folder1', imgsResolution, '%d','jpg');
[folder2Inputs,folder2Targets] = readyImages('Folder2', imgsResolution, 'letter_bnw_%d','jpg', 1);
[folder3Inputs,folder3Targets] = readyImages('Folder3', imgsResolution, 'letter_bnw_test_%d','jpg');
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
fprintf('Verificar precisão na Folder3 com a rede -> %s\n',networkId)
folder3Output = sim(net, folder3Inputs);
folder3OutputAccuracy = testNetworkAccuracy(folder3Output,folder3Targets);
fprintf('Precisao Folder3 -> %f\n', folder3OutputAccuracy)
% -------------------------------------------------------------------------



% Train -------------------------------------------------------------------
fprintf('\nTreinar com a Folder3\n')
[net,tr] = train(net, folder3Inputs, folder3Targets);
if(strcmp(tr.stop, 'User cancel.'))
    return
end
fprintf('TrainStop: %s\n',tr.stop)

% Simulate each Folder ----------------------------------------------------
fprintf('\nSimular com cada pasta\n')
folder1Output = sim(net, folder1Inputs);
folder1OutputAccuracy = testNetworkAccuracy(folder1Output,folder1Targets);
fprintf('Precisao Folder1 -> %f\n', folder1OutputAccuracy)
folder2Output = sim(net, folder2Inputs);
folder2OutputAccuracy = testNetworkAccuracy(folder2Output,folder2Targets);
fprintf('Precisao Folder2 -> %f\n', folder2OutputAccuracy)
folder3Output = sim(net, folder3Inputs);
folder3OutputAccuracy = testNetworkAccuracy(folder3Output,folder3Targets);
fprintf('Precisao Folder3 -> %f\n', folder3OutputAccuracy)
% -------------------------------------------------------------------------


% Train with all Folders --------------------------------------------------
fprintf('\nTreinar com todas as imagens\n')
allFoldersInputs = [folder1Inputs folder2Inputs folder3Inputs];
allFoldersTargets = [folder1Targets folder2Targets folder3Targets];
[net,tr] = train(net, allFoldersInputs, allFoldersTargets);
if(strcmp(tr.stop, 'User cancel.'))
    return
end
fprintf('TrainStop: %s\n',tr.stop)
% -------------------------------------------------------------------------

% Simulate each Folder ----------------------------------------------------
fprintf('\nSimular com cada pasta novamente\n')

folder1Output = sim(net, folder1Inputs);
folder1OutputAccuracy = testNetworkAccuracy(folder1Output,folder1Targets);
fprintf('Precisao Folder1 -> %f\n', folder1OutputAccuracy)

folder2Output = sim(net, folder2Inputs);
folder2OutputAccuracy = testNetworkAccuracy(folder2Output,folder2Targets);
fprintf('Precisao Folder2 -> %f\n', folder2OutputAccuracy)

folder3Output = sim(net, folder3Inputs);
folder3OutputAccuracy = testNetworkAccuracy(folder3Output,folder3Targets);
fprintf('Precisao Folder3 -> %f\n', folder3OutputAccuracy)
% -------------------------------------------------------------------------

