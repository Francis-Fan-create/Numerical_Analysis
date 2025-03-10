clc
clear
clc
n = 20;
I = zeros(1, n+1);
I(1) = log(1.2);
for i = 1:n
I(i+1) = -5*I(i) + 1/i;
end
for i = 1:numel(I)
fprintf('I[%d]=%f\n', i-1, I(i));
end