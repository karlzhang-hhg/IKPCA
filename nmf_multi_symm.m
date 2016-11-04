function [A_est, K_est] = nmf_multi_symm(X,A0,K0,maxiter,tolx,tolfun)
%% Implement Non-negative matrix factorization using multiplicative update algorithm
%% while restricting K to be symmetric. The default model for factorization is X = AK.
% Input:
% X: Data matrix (each column of X is a instance)
% A0: Initial guess for A
% K0: Initial guess for K
% maxiter: Maximum iteration number
% tolx: Tolerance for relative residual (norm(A(t)-A(t-1))/norm(A(t)) or norm(K(t)-K(t-1))/norm(K(t))). Default is 1e-4.
% tolfun: Tolerance for relative residual (norm(X - AK)). Default is 1e-4.

[n,N] = size(X);
sqrteps = sqrt(eps);
count = 0;
A_old = A0;
K_old = K0;

fprintf('    rep\t   iteration\t   rms resid\t  |delta x|\n');
dispfmt = '%7d\t%8d\t%12g\t%12g\n';
repnum = 1;

while (count <= maxiter)
    numer = A_old'*X;
    K_new = max(0,K_old.*(numer./((A_old'*A_old)*K_old+eps(numer))));
    K_new = (K_new+K_new')/2; % Note that there are several ways to keep K_new symmetric
    numer = X*K_new';
    A_new = max(0,A_old.*(numer./(A_old*(K_new*K_new')+eps(numer))));
   
   % Get norm of difference and max change in factors
    D = X - A_new*K_new;
    Dnorm = sqrt(sum(sum(D.^2))/N/n);
    Dw = max(max(abs(A_new-A_old) / (sqrteps+max(max(abs(A_old))))));
    Dh = max(max(abs(K_new-K_old) / (sqrteps+max(max(abs(K_old))))));
    Delta = max(Dw,Dh);

    % Check for convergence
    if count>1
        if Delta <= tolx
            break;
        elseif Dnorm0-Dnorm <= tolfun*max(1,Dnorm0)
            break;
        elseif count == maxiter
            break
        end
    end

    % Print iteration
    fprintf(dispfmt,repnum,count,Dnorm,Delta);
    
    % Remember previous iteration results
    Dnorm0 = Dnorm;
    A_old = A_new;
    K_old = K_new;
    count = count + 1;
end

% Print final
fprintf(dispfmt,repnum,count,Dnorm,Delta);

A_est = A_old;
K_est = K_old;

end