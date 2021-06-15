networkFolder = 'redes guardadas/networks/';
networkId = '13';
load([networkFolder networkId '.mat']);

imgResolution = 12;

[manualInputs, manualTargets] = readyImages('Manual', imgResolution, '%d','jpg');

imageOutputs = sim(net, manualInputs);


amountOfElements = size(imageOutputs,2);
correctOutput=0;
for i=1:amountOfElements                        % Para cada classificacao
    [~, obtained] = max(imageOutputs(:, i));    % guarda a linha onde encontrou valor mais alto da saida obtida
    [~, supposed] = max(manualTargets(:, i));   % guarda a linha onde encontrou valor mais alto da saida desejada
    if obtained == supposed                     % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        correctOutput = correctOutput + 1;
    else
        imageOutputs(:,i)
        fprintf('Imagem não deu certo (i) -> %d, invés deu -> %d\n', supposed, obtained)
    end
end
accuracy = correctOutput/amountOfElements * 100;

fprintf('Precisao Manual -> %f\n', accuracy)

