function A_app = updateA(X,K,B,options,lam)
%% Estimate Variation Sources 
[N,n] = size(X);
% Given vector a, find true coefficient matrix A
% sigma_alg = options.sigma_alg;
% a = a*ones(n,1);
% K_true = Gaussian_Kernel(Z,sigma_alg);
% B = X'-a*ones(1,N);

% A_true = B/K_true;
% In calculation, I found that the K_true is close to singular, so that I
% need to use regularized least-square

% Regularized coefficients
lambda = 10.^(-12:-1);

% Error of estimation
err = zeros(1,length(lambda));
% for i = 1:length(lambda)
%     A_MAP = ((lambda(i)*eye(N)+K_true*K_true)\K_true*(X-ones(N,1)*a'))';
%     err(i) = sum(sum((B-A_MAP*K_true).^2))/n/N;
%     i
% end

% There still some round-off error among the first 10 lambda. Choose the
% 11st lambda and use that to approximately calculate true coefficient
% matrix A.
loop_ind = [11,11;10,11;10,10;9,10;9,9];


% ind1 = loop_ind(i,1);
% ind2 = loop_ind(i,2);

%lam = 0.001;
    
% [~,ind] = min(err);
% lambda(ind)
A_app = ((lam*eye(N)+K*K')\K*B')';

end