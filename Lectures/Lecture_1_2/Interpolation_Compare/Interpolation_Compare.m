%% Lagrange Interpolation with Uniform and Chebyshev Nodes

% Define interpolation orders
orders = [2, 4, 8, 16, 32];

% Define function to generate interpolation values
define_function_values = @(x) arrayfun(@A, x);

% Compute Lagrange interpolants for uniform nodes
for i = 1:length(orders)
    n = orders(i);
    x = linspace(-5, 5, n+1); % Generate n+1 uniform interpolation points
    y = define_function_values(x); % Compute function values at these points
    funcLag{i} = funcLagrange(x, y); % Compute Lagrange interpolation polynomial
end

% Define evaluation points for interpolation
 t = -5:0.01:5;

% Compute polynomial values for each interpolation function
for i = 1:length(orders)
    fx{i} = polyval(funcLag{i}, t);
end

% Plot the interpolated functions
figure;
hold on;
plot_colors = {'r', 'g', 'b', 'm', 'c'};
for i = 1:length(orders)
    plot(t, fx{i}, 'Color', plot_colors{i}, 'DisplayName', sprintf('n=%d', orders(i)));
end
legend;
title('Lagrange Interpolation with Uniform Nodes');
xlabel('x');
ylabel('Interpolated Value');
hold off;

%% Compute Lagrange interpolants for Chebyshev nodes
for i = 1:length(orders)
    n = orders(i);
    xx = 5 * cos((2*(1:n+1)-1) * pi / (2*(n+1))); % Chebyshev nodes
    y = define_function_values(xx); % Compute function values
    funcLagCheb{i} = funcLagrange(xx, y); % Compute Lagrange interpolation polynomial
end

% Compute polynomial values for each Chebyshev interpolation function
for i = 1:length(orders)
    fxCheb{i} = polyval(funcLagCheb{i}, t);
end

% Plot the interpolated functions for Chebyshev nodes
figure;
hold on;
for i = 1:length(orders)
    plot(t, fxCheb{i}, 'Color', plot_colors{i}, 'DisplayName', sprintf('n=%d', orders(i)));
end
legend;
title('Lagrange Interpolation with Chebyshev Nodes');
xlabel('x');
ylabel('Interpolated Value');
hold off;

%% Function Definitions
function [ansFun] = funcLagrange(x, y)
    % Compute Lagrange interpolating polynomial given nodes x and values y
    if length(x) ~= length(y)
        error('The number of x and y values must be the same.');
    end
    n = length(x);
    basfunc = zeros(n, n); % Matrix to store basis polynomials
    for i = 1:n
        p = poly(x([1:i-1, i+1:end])); % Compute Lagrange basis polynomial
        basfunc(i, :) = p / polyval(p, x(i)); % Normalize basis polynomial
    end
    ansFun = sum(y' .* basfunc, 1); % Compute final interpolating polynomial
end

function y = A(x)
    % Function to interpolate: f(x) = 1 / (1 + x^2)
    y = 1 ./ (1 + x.^2);
end
