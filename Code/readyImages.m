function [imageInputs,imageTargets] = readyImages(imagesPath,imgResolution,fileNumberFilter, flag)
%READYIMAGES
% Esta função pega nas imagens de treino e converte para matrizes para
% poderem ser usadas na rede neuronal
amountImageTypes = 10;


% Preparar a lista das imagens a abrir
files = dir(imagesPath); % TODO fix da shit 2 primeiros elementos são saltados

%Sort the files by their real order ---------------------------------------
fileNames = strings(1,length(files));

for i=1:length(files)
    fileNames(i) =  files(i).name;
end

amountImgs = length(files) - 2; % - 200;

filesWithoutDots = strings(1,amountImgs);
filenum = zeros(1,amountImgs);

for i=3:length(files)
    str = fileNames(i);
    
    filenum(i - 2) = sscanf(str, fileNumberFilter);
    filesWithoutDots(i - 2) = str; 
end

[~,sortedIndexes] = sort(filenum);
almostReadyPaths = filesWithoutDots(sortedIndexes);

pathPrePart = strcat(imagesPath, '\');

readyPaths = strings(1,amountImgs);

for i=1:amountImgs
    readyPaths(i) = strcat(pathPrePart, almostReadyPaths(i));
end

if(flag == 2)
    readyPaths = fliplr(readyPaths);
    % garantir que a inversão está certa
end

% Sort End ----------------------------------------------------------------

amountOfEachType = amountImgs / amountImageTypes;

imageInputs = zeros(imgResolution * imgResolution , amountImgs);
imageTargets = zeros(amountImageTypes, amountImgs);

counter = 0;

for i=1:amountImgs
    
    image = imread(readyPaths(i));
    image = imresize(image, [imgResolution imgResolution]);
    
    imageInputs(:,i) = image(:);
    supposedLetter = fix(counter / amountOfEachType) + 1;
    imageTargets(supposedLetter , i) = 1;
    
    counter = counter + 1;
    
end