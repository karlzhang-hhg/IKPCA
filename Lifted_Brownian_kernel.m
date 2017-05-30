function K = Lifted_Brownian_kernel(Z,options)
%% Lifted Brownian kernel based on Prof. Apley's paper: Lifted Brownian kriging models
% Input:
% Z: variation sources
% options: storing parameter
%   options.a: scaling parameter
%   options.beta: power parameter for long range property of the response
% Output:
% K: kernel matrix

N = size(Z,1); %Number of data points
%Apply Gaussian Kernel to distance of two points from Z
K = zeros(N);
Z_sq = repmat(sum(Z.^2,2),[1,N]);
Z_ds = Z_sq+Z_sq'-2*(Z*Z');

psi = (1+Z_sq*options.a).^options.beta;

K = psi+psi'-(1+Z_ds*options.a).^options.beta-1;

end