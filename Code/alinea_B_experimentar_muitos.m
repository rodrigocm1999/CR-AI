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
% Create and put header if results file doesn't exist -----------------------------
scoresFile = [savesFolder '/networksScores.csv'];
if not(isfile(scoresFile))
    fields = {'FileId','Layers','TrainAccuracy','ValidationAccuracy','TestAccuracy','Folder3Accuracy','Iterations','Performance','TrainFunction','TrainTime','HiddenLayer','OuputLayer'};
    writecell(fields,scoresFile,'Delimiter',',');
end
% -------------------------------------------------------------------------

% Load Images -------------------------------------------------------------
imgsResolution = 12; % Tamanho ideal, pois tem o minimo de informação sem perda de "detalhes"
[imageInputs,imageTargets] = readyImages('Folder2', imgsResolution, 'letter_bnw_%d','jpg', 1);
[testInput,testTargets] = readyImages('Folder3', imgsResolution, 'letter_bnw_test_%d','jpg');
% -------------------------------------------------------------------------

layersConfigs = [{10},{20}];
activationFncConfigs = [
    {[{'logsig'} {'purelin'}]},...
    {[{'tansig'} {'purelin'}]},...
    {[{'tansig'} {'tansig'}]},...
    {[{'logsig'} {'logsig'}]}];
trainFcnConfigs = [{'trainlm'},{'trainrp'},{'traincgp'}];
divideFcnConfigs = [{'divideint'},{'dividerand'}];
divideParamsConfigs = [{[0.7 0.15 0.15]},{[0.8 0.1 0.1]}]; % train val test

totalRuns = numel(divideParamsConfigs) * numel(layersConfigs) * ...
    numel(trainFcnConfigs) * numel(activationFncConfigs) * numel(divideFcnConfigs);
counter = 0;

for divParamI=1:numel(divideParamsConfigs)
    for layI=1:numel(layersConfigs)
        for divFcnI=1:numel(divideFcnConfigs)
            for actFcnI=1:numel(activationFncConfigs)
                for traFcnI=1:numel(trainFcnConfigs)
                    counter = counter + 1;
                    fprintf('\nRuns -> %d(%d)\n', counter,totalRuns)
                    
                    layers = layersConfigs{layI};
                    net = feedforwardnet(layers);
                    
                    net.trainFcn = trainFcnConfigs{traFcnI};
                    net.layers{1}.transferFcn = activationFncConfigs{actFcnI}{1};
                    net.layers{2}.transferFcn = activationFncConfigs{actFcnI}{2};
                    
                    net.divideFcn = divideFcnConfigs{divFcnI};
                    ratios = divideParamsConfigs{divParamI};
                    net.divideParam.trainRatio = ratios(1);
                    net.divideParam.valRatio = ratios(2);
                    net.divideParam.testRatio = ratios(3);
                    
                    net.trainParam.epochs = 200;
                    
                    fprintf('LayI: %d\tTrainFcn: %s\tLayer1Fcn: %s\tLayer2Fcn: %s\n',...
                        layI,net.trainFcn, net.layers{1}.transferFcn, net.layers{2}.transferFcn);
                    
                    startTime = tic;
                    % Train -------------------------------------------------------------------
                    [net,tr] = train(net, imageInputs, imageTargets);
                    elapsedTime = toc(startTime);
                    fprintf('Tempo: %.2f s\tIterações: %d\n',elapsedTime,tr.num_epochs)
                    % disp(tr)
                    
                    if(strcmp(tr.stop, 'User cancel.'))
                        return
                    end
                    fprintf('TrainStop: %s\n',tr.stop)
                    
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
                    testOutput = sim(net, testInput);
                    folder3Accuracy = testNetworkAccuracy(testOutput,testTargets);
                    fprintf('Precisao total Folder3 -> %f\n', folder3Accuracy)
                    % -------------------------------------------------------------------------
                    
                    %fileId = num2str(now());
                    fileId = num2str(counter);
                    
                    if trainAccuracy < 100 && testAccuracy >= 80
                        fprintf('Accuracy de treino não chegou aos 100 perc. Considerar remover dos testes\n')
                    else
                        networkFile = [networksFolder '/' fileId '.mat'];
                        save(networkFile, 'net', 'imgsResolution');
                    end
                    
                    % Save Results-------------------------------------------------------------
                    
                    results = {fileId, layers, ...
                        trainAccuracy ,validationAccuracy ,testAccuracy,...
                        folder3Accuracy, tr.num_epochs, perform(net,imageTargets,output),...
                        tr.trainFcn, elapsedTime,net.layers{1}.transferFcn,net.layers{2}.transferFcn};
                    
                        
                    writecell(results,scoresFile,'WriteMode','append','Delimiter',',');
                    
                    trimmedTr.layers = layers;
                    trimmedTr.gradient = tr.gradient(tr.num_epochs);
                    trimmedTr.trainAccuracy = trainAccuracy;
                    trimmedTr.validationAccuracy = validationAccuracy;
                    trimmedTr.testAccuracy = testAccuracy;
                    trimmedTr.trainTime = elapsedTime;
                    
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
                end
            end
        end
    end
end
