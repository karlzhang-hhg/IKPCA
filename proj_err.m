function [err,mag,U_K] = proj_err(K_est,U_X,options)
%Calculate the projection error of column space of X_std onto the column
%space of K_est
% Output:
% err: a projection error of col(X_std) onto col(K_est) (N-by-n matrix)
% mag: the magnitude of the error vectors which is the smallest eigen-value
%       of the K_est passing the options.pct threshold
% Input:
% K_est: estimated K matrix
% U_X: sigular vectors of standardized X corresponding to the singular
%       values accounting for options.pct of variance.
% options: storing options parameters

N = options.N;
% [N,n] = size(X_std);
[eig_vec_K,eig_v_K] = eig(K_est);
lam_K = diag(eig_v_K);
th_K = sing_th_ind(sqrt(lam_K(N:-1:1)),options.pct);
th_K
cond(K_est)
lam_K(N:-1:N-(th_K-1))
colsp_K = eig_vec_K(:,N:-1:N-(th_K-1));
%P_K = colsp_K*diag(lam_K(N:-1:N-(th_K-1)).^(-1))*colsp_K';
U_K = colsp_K;
P_K = colsp_K*colsp_K';
% [U_X,S_X,V_X] = svd(X_std,'econ');
% lam_X = diag(S_X);
% th_X = sing_th_ind(lam_X,options.pct);
% err = (eye(N)-P_K)*U_X(:,1:th_X);
err = (eye(N)-P_K)*U_X;
%err = (eye(N)-P_K)*X_std;
%[err,~] = qr(err,0);
if nargout > 1
    mag = lam_K(N-(th_K-1));
end
end