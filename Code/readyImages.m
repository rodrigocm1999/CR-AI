function [imageInputs,imageTargets] = readyImages(folderName,imgResolution,fileNumberFilter,fileExtension, reverse, isFullPath)
%READYIMAGES
% Esta função pega nas imagens de treino e converte para matrizes para
% poderem ser usadas na rede neuronal
amountImageTypes = 10;

if ~exist('reverse','var')
    reverse = 0;
end
if ~exist('isFullPath','var')
    isFullPath = 0;
end

if ~isFullPath
    folderPrefix = 'Datasets/original/';
    folderPath = [folderPrefix folderName];
else
   folderPath = folderName;
   splitPath = split(folderName,"\");
   folderName = splitPath{end};
end

cacheFolderPrefix = ['Datasets/cached/res_' num2str(imgResolution) '/'];
cacheFolderPath = strcat(cacheFolderPrefix, folderName);

extensionLength = strlength(fileExtension) + 1;

% Preparar a lista das imagens a abrir
files = dir(folderPath);
%Sort the files by their real order ---------------------------------------
fileNames = strings(1,length(files));

for i=1:length(files)
    fileNames(i) = files(i).name;
end

amountImgs = length(files) - 2;

filesWithoutDots = strings(1,amountImgs);
filenum = zeros(1,amountImgs);

for i=3:length(files)
    str = fileNames(i);
    
    temp = regexp(str,'\d*','Match');
%     temp = sscanf(str, fileNumberFilter);
    filenum(i - 2) = temp;
    numba = strlength(str) - extensionLength;
    filesWithoutDots(i - 2) = extractBetween(str,1,numba);
end

[~,sortedIndexes] = sort(filenum);
fileNames = filesWithoutDots(sortedIndexes);

% pathPrePart = strcat(folderPath, '\');

if(reverse == 1)
    fileNames = fliplr(fileNames);
    % garantir que a inversão está certa
end
% Sort End ----------------------------------------------------------------

amountOfEachType = amountImgs / amountImageTypes;

imageInputs = zeros(imgResolution * imgResolution , amountImgs);
imageTargets = zeros(amountImageTypes, amountImgs);

counter = 0;

if not(isfolder(cacheFolderPath))
    mkdir(cacheFolderPath);
end

for i=1:amountImgs
    
    fileName = strcat(fileNames(i),'.');
    fileNameJPG = strcat(fileName,fileExtension);
    
    cachedFilePath = [cacheFolderPath '/' ];
    cachedFilePath = strcat(cachedFilePath,fileName);
    cachedFilePath = strcat(cachedFilePath,'png');
    
    if(isfile(cachedFilePath))
        %Get the cached image
        image = imread(cachedFilePath);
    else
        %Get the image ready and cache it
        filePath = strcat(folderPath,'/');
        filePath = strcat(filePath,fileNameJPG);
        
        image = imread(filePath);
        image = imresize(image, [imgResolution imgResolution]);
        if size(image,3) > 1
            image = rgb2gray(image);
        end
        imwrite(image,cachedFilePath);
    end
    
    image = double(image)/255;
    imageInputs(:,i) = image(:);
    supposedLetter = floor(counter / amountOfEachType) + 1;
    imageTargets(supposedLetter , i) = 1;
    
    counter = counter + 1;
end
