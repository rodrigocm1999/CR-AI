function [accuracy] = testNetworkAccuracy(output,target,amountOfElements)
%TESTNETWORKACCURACY
% Esta função serve para calcular a accuracy duma rede comparando os
% valores de output com os supostos valores corretos

correctOutput=0;
for i=1:amountOfElements                         % Para cada classificacao
    [~, obtained] = max(output(:, i));           %b guarda a linha onde encontrou valor mais alto da saida obtida
    [~, supposed] = max(target(:, i));           %d guarda a linha onde encontrou valor mais alto da saida desejada
    if obtained == supposed                      % se estao na mesma linha, a classificacao foi correta (incrementa 1)
        correctOutput = correctOutput + 1;
    end
end

accuracy = correctOutput/amountOfElements *100;

end

