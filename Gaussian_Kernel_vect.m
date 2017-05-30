function K = Gaussian_Kernel_vect(Z,options)
% Gaussian Kernel
% Input:
% Z: matrix of input data points with rows as points
% options: store the parameters
%   options.sigma_kernel: parameter for Gaussian Kernel
% Output:
% K: output Gaussian Kernel based on input data points

N = size(Z,1); %Number of data points
%Apply Gaussian Kernel to distance of two points from Z
K = zeros(N);
Z_sq = repmat(sum(Z.^2,2),[1,N]);
Z_ds = Z_sq'+Z_sq-2*(Z*Z');
K = exp(-Z_ds/2/options.sigma_alg^2);

% for i = 1:N
%     for j = i:N
%         diff_ij = Z(i,:)-Z(j,:);
%         %K(i,j) = exp(-diff_ij*diff_ij'/2/sigma_kernel^2);
%         if i~=j
%             K(j,i) = K(i,j);
%         end
%     end
% end