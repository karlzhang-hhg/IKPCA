function K = Gaussian_Kernel_heuristic(Z,d_samp,sigma_kernel)
% Gaussian Kernel
% Input:
% Z: matrix of input data points with rows as points
% sigma_kernel: parameter for Gaussian Kernel
% Output:
% K: output Gaussian Kernel based on input data points

N = size(Z,1); %Number of data points
%Apply Gaussian Kernel to distance of two points from Z
M = length(d_samp);
K = zeros(N,M);
for j = 1:M
    for i = 1:N
        diff_ij = Z(i,:)-Z(d_samp(j),:);
        K(i,j) = exp(-diff_ij*diff_ij'/2/sigma_kernel^2);
    end
end