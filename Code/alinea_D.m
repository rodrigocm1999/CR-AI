networkFolder = 'redes guardadas/networks/';
networkId = '13';
load([networkFolder networkId '.mat']);

imgResolution = 12;
[manualInputs, manualTargets] = readyImages('Manual', imgResolution, '%d','jpg');

imageOutputs = sim(net, manualInputs);


correctOutput=0;
amountOfElements = size(imageOutputs,2);
for i=1:amountOfElements
    [~, obtained] = max(imageOutputs(:, i));
    [~, supposed] = max(manualTargets(:, i));
    if obtained == supposed
        correctOutput = correctOutput + 1;
    else
        fprintf('Imagem não deu certo (i) -> %d, invés deu -> %d\n', supposed, obtained)
    end
end
accuracy = correctOutput/amountOfElements * 100;

fprintf('Precisao Manual -> %f\n', accuracy)

