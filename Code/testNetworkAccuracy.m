function [accuracy] = testNetworkAccuracy(output,target,testedIndexes)
%TESTNETWORKACCURACY
% Esta função serve para calcular a accuracy duma rede comparando os
% valores de output com os supostos valores corretos

%Calcula e mostra a percentagem de classificacoes corretas no conjunto de teste
r=0;
for i=1:size(testedIndexes,2)               % Para cada classificacao  
  [a, b] = max(output(:,i));          %b guarda a linha onde encontrou valor mais alto da saida obtida
  [c, d] = max(target(:,i));  %d guarda a linha onde encontrou valor mais alto da saida desejada
  if b == d                       % se estao na mesma linha, a classificacao foi correta (incrementa 1)
      r = r+1;
  end
end

accuracy = r/size(testedIndexes,2)*100;

end

