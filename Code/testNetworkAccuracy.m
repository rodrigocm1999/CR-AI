function [accuracy] = testNetworkAccuracy(output,target,testIndices)
%TESTNETWORKACCURACY
% Esta função serve para calcular a accuracy duma rede comparando os
% valores de output com os supostos valores corretos

if exist('testIndices','var')
    output = output(:, testIndices);
    target = target(:, testIndices);
end

amountOfElements = size(output,2);

correctOutput=0;
for i=1:amountOfElements                         % Para cada classificacao
    [~, obtained] = max(output(:, i));           % guarda a linha onde encontrou valor mais alto da saida obtida
    [~, supposed] = max(target(:, i));           % guarda a linha onde encontrou valor mais alto da saida desejada
    if obtained == supposed                      % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        correctOutput = correctOutput + 1;
    end
end

accuracy = correctOutput/amountOfElements * 100;

end

