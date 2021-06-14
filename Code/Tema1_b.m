clear;

imgsResolution = 12;
[imageInputs,imageTargets] = readyImages('Folder2', imgsResolution, 'letter_bnw_%d','jpg', 1);

layers = [5];
net = feedforwardnet(layers);

% net.trainFcn = 'traingdx';
net.trainParam.epochs = 100;

net.divideFcn = 'divideint';
net.divideParam.trainRatio = 0.9;
net.divideParam.testRatio = 0.1;
net.divideParam.valRatio = 0.1;

% Train -------------------------------------------------------------------
[net,tr] = train(net, imageInputs, imageTargets);
disp(tr)
% Simulate ----------------------------------------------------------------
output = sim(net, imageInputs);
% -------------------------------------------------------------------------

% Check network accuracy --------------------------------------------------
trainAccuracy = testNetworkAccuracy(output,imageTargets,tr.trainInd);
fprintf('Precisao treinada treino -> %f\n', trainAccuracy)
validationAccuracy = testNetworkAccuracy(output,imageTargets,tr.valInd);
fprintf('Precisao val treino -> %f\n', validationAccuracy)
testAccuracy = testNetworkAccuracy(output,imageTargets,tr.testInd);
fprintf('Precisao teste treino -> %f\n', testAccuracy)
% -------------------------------------------------------------------------

% Test using different images ---------------------------------------------
[testInput,testTargets] = readyImages('Folder3', imgsResolution, 'letter_bnw_test_%d','jpg');
% [testInput,testTargets] = readyImages('Folder1', imgsResolution, '%d','jpg');
testOutput = sim(net, testInput);
folder3Accuracy = testNetworkAccuracy(testOutput,testTargets);
fprintf('Precisao total Folder3 -> %f\n', folder3Accuracy)
% -------------------------------------------------------------------------

% Create necessary folders ------------------------------------------------
savesFolder = 'savedNetworks';
if not(isfolder(savesFolder))
    mkdir(savesFolder);
end
networksFolder = [savesFolder '/networks'];
if not(isfolder(networksFolder))
    mkdir(networksFolder);
end
trainResultsFolder = [savesFolder '/trainResults'];
if not(isfolder(trainResultsFolder))
    mkdir(trainResultsFolder);
end
% -------------------------------------------------------------------------

% Create and put header if file doesn't exist -----------------------------
scoresFile = [savesFolder '/networksScores.csv'];
if not(isfile(scoresFile))
    fields = {'FileId','Layers','TrainAccuracy','ValidationAccuracy','TestAccuracy','Folder3Accuracy','Iterations','Performance','TrainFunction'};
    writecell(fields,scoresFile,'Delimiter',';');
end
% -------------------------------------------------------------------------

fileId = num2str(now());

results = {fileId, layers, ...
    trainAccuracy ,validationAccuracy ,testAccuracy ,folder3Accuracy,...
    tr.num_epochs , perform(net,imageTargets,output) ,tr.trainFcn};
writecell(results,scoresFile,'WriteMode','append','Delimiter',';');

% Save --------------------------------------------------------------------
networkFile = [networksFolder '/' fileId '.mat'];
save(networkFile, 'net');

trimmedTr.layers 
trimmedTr.gradient =  tr.gradient(tr.num_epochs);
trimmedTr.trainAccuracy = trainAccuracy;
trimmedTr.validationAccuracy = validationAccuracy;
trimmedTr.testAccuracy = testAccuracy;

importantFieldsTr = {'trainFcn','trainParam','performFcn','divideFcn',...
    'divideParam','valInd','testInd','stop','num_epochs','best_epoch',...
    'goal'};
for i=1 : length(importantFieldsTr)
    field = string(importantFieldsTr(i));
    trimmedTr.(field) =  tr.(field);
end

trJson = jsonencode(trimmedTr, 'ConvertInfAndNaN' , true);

trainResultFile = [trainResultsFolder '/' fileId '.json'];
fid = fopen(trainResultFile,'w');
fprintf(fid,'%s\n',trJson(:));
fclose(fid);
