networkFolder = 'savedNetworks/networks/';
networkId = '738322.5674';
load([networkFolder networkId '.mat']);


% amountImageTypes = 10;
% imageInputs = zeros(imgResolution * imgResolution , 1);
% imageTargets = zeros(amountImageTypes, 1);
% filePath = 'Datasets\original\Manual\10.jpg';
% image = imread(filePath);
% if size(image,3) > 1
%     image = rgb2gray(image);
% end
% image = imresize(image, [imgResolution imgResolution]);
% image = double(image)/255;
% imageInputs(:,1) = image(:);
% imageTargets(1 , 1) = 1;

imgResolution = 12;
[manualInputs, manualTargets] = readyImages('Manual', imgsResolution, '%d','jpg');
imageOutputs = sim(net, manualInputs);


amountOfElements = size(imageOutputs,2);

correctOutput=0;
for i=1:amountOfElements                         % Para cada classificacao
    [~, obtained] = max(imageOutputs(:, i));           % guarda a linha onde encontrou valor mais alto da saida obtida
    [~, supposed] = max(manualTargets(:, i));           % guarda a linha onde encontrou valor mais alto da saida desejada
    if obtained == supposed                      % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        correctOutput = correctOutput + 1;
    else
        imageOutputs(:,i)
        fprintf('Imagem não deu certo (i) -> %d, invés deu -> %d\n', supposed, obtained)
    end
end

accuracy = correctOutput/amountOfElements * 100;

fprintf('Precisao Manual -> %f\n', accuracy)

