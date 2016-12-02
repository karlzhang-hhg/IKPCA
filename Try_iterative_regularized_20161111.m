%% Iterative algorithm using regularization
clear();
options = ini_options();
[Z,X,color,PC_X,V,D,PC_std_ind,eig_values_std] = genGaussianDat(options);
N = options.N;
n = (options.l+1)^2;
a = ones(n,1);
B = X'-a*ones(1,N);
max_iter = options.max_iter;
lam1 = 0.01;
lam2 = 0.01;

% %Standardize each coordinates of X; in other words, for each column,
% %substract means and divide by unbiased standard deviation of that column
% std_X = std(X,0,1); 
% X_std = (X-repmat(mean(X,1),N,1))/diag(std_X);
% %X_std = X;
% 
% %In svd (singular value decomposition, the singular values are sorted
% %in non-increasing order.
% [V,D,U] = svd(X_std/sqrt(N-1),'econ');
% %Principle components of X (PCA scores)
% PC_X = V*D(:,1:length(PC_std_ind));
% %The standardized observations doesn't have full column rank, so we
% %should pick out those columns of V that correspond to non-zero
% %singular values.
% rank_X_std = length(PC_std_ind);
% V = V(:,1:rank_X_std);
        
sigma_nois = options.sigma_nois; %Standard Variation of noise
proj_mat = 1/N*ones(N)+V*V'-n*sigma_nois^2*eye(N); %Matrix for projection of K to column space of X

%C = 1.5;

Z_ini =  PC_X(:,[1,2])/diag(sqrt(var(PC_X(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2]))));
K_est = Gaussian_Kernel(Z_ini,options.sigma_alg);
for i=1:max_iter
    A_app = updateA(X,K_est,B,options,lam1);
    K_est = updateK(X,A_app,B,options,lam2);
    %Projection
    K_tilde = proj_mat*K_est*proj_mat; %non-negative definite
    K_tilde = (K_tilde+K_tilde')/2;
    %K_tilde = (K_est+K_est')/2;
    
    %IGaussian_Kernel_old means not using conjugate symmetry of inverted
    %elements of K
    [Z_est,lambda_est] = IGaussian_Kernel_old(K_tilde,options.sigma_alg,options.p);
    %'Symmetric K' means when calculating the inverted elements of K, I
    %didn't use conjugate symmetry, while 'asymmetric K' means I use conjugate symmetry 
    title_text3 = ['(iter=',num2str(i),') Estimated variation source based on regularized OLS',10,'(\sigma_{data}=',...
        num2str(options.sigma_data),', \sigma_{alg}=',num2str(options.sigma_alg),', \lambda_{A}=',num2str(lam1),...
        ', \lambda_{K}=',num2str(lam2),') with symmetric K'];
    scatter_label2d(Z_est,title_text3,options.dd,0,options.psize,color)
%     saveas(gca,[options.cwd,['10-',num2str(i-1)]],'jpg');
%     saveas(gca,[options.cwd,['10-',num2str(i-1)]],'fig');
%     close(gcf);
    K_est = Gaussian_Kernel(Z_est(:,[1,2])/diag(sqrt(var(Z_est(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2])))),options.sigma_alg);
    i
end