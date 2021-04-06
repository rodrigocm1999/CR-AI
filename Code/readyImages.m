function [imageInputs,imageTargets] = readyImages(imagesPath,imgResolution)
%READYIMAGES
% Esta função pega nas imagens de treino e converte para matrizes para
% poderem ser usadas na rede neuronal

files = dir(imagesPath); % TODO fix da shit 2 primeiros elementos são saltados

amountImgs = length(files) - 2; %- 200;

amountImageTypes = 24;
amountOfEachType = amountImgs / amountImageTypes;

imageInputs = zeros(imgResolution * imgResolution , amountImgs);
imageTargets = zeros(amountImageTypes, amountImgs);

counter = 0;

for i=3:amountImgs + 2
    
    filePath = strcat(files(i).folder , strcat('\',files(i).name))
    image = imread(filePath);
    image = imresize(image, [imgResolution imgResolution]);
    
    imageInputs(:,i - 2) = image(:);
    supposedLetter = fix(counter / amountOfEachType) + 1;
    imageTargets(supposedLetter , i - 2) = 1;
    
    counter = counter + 1;
    
end