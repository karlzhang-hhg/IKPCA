function [f,g]=cost_one_step_heuristic(Z,X,d_samp,options)
%%Follow the 20170117-weekly-report. Do one-step optimization to get Z
% Z: temporary variation sources (variable of the function)
% X: observations
% options: the options in ini_options.m
% flag: 
%   val: function value
%   grad: gradient vector

%Z = reshape(Z,[options.p,options.N])';

%%Code
lam_A = options.lam_A;
lam_a = options.lam_a;

[N,n] = size(X);
[~,pp] = size(Z);

M = length(d_samp);

K_d_samp = Gaussian_Kernel_heuristic(Z,d_samp,options.sigma_alg);
X_d_samp = X(d_samp,:);

I_N = eye(N);
I_M = eye(M);
one_N = ones(N,N);
one_M = ones(M,M);
temp = I_M-1/(M+lam_a)*one_M;

% Used to test permutation
% res = randi([1 N],1,2);
% i1 = res(1);
% i2 = res(2);
% P = I;
% P(i1,i1)=0;
% P(i2,i2)=0;
% P(i1,i2)=1;
% P(i2,i1)=1;

diff = X_d_samp'*(I_M-temp*K_d_samp'*K_d_samp/(lam_A*I_M+temp*K_d_samp'*K_d_samp))*temp;

f = sum(sum(diff.^2));

if nargout > 1
    Psi = lam_A*I+K*temp*K;
    inv_Psi = I/Psi;
    J_K_deri = -2*((temp*X*diff*temp*K*inv_Psi)'+inv_Psi*K*temp*X*diff*temp);

    for p=1:N
        for q=1:N
            temp2 = inv_Psi(:,p)*temp(q,:)*K*inv_Psi;
            inv_Psi_K = -(temp2'+temp2);
            J_K_deri(p,q)=J_K_deri(p,q)-2*sum(sum(diff.*(X'*temp*K*inv_Psi_K*K*temp)));
        end
    end
    grad = zeros(N*pp,1);

    for i=1:N
        K_z = -diag(K(i,:)/options.sigma_alg^2)*(repmat(Z(i,:),N,1)-Z);
        term1=J_K_deri(i,:)*K_z;
        term2=J_K_deri(:,i)'*K_z;
        grad((i-1)*pp+1:i*pp) = (term1+term2)'; 
    end
    g = grad;
end

%         out = finite_diff(@cost_one_step,Z,X,options,1e-6);
        
%% Use tensor notation to calculate is very space-consuming.
%         Psi = lam_A*I+K*temp*K;
%         inv_Psi = I/Psi;
%         inv_Psi_ten = tensor(inv_Psi);
%         %The index is spqr as show in weekly report 20170117
%         %inv_Psi_deri = -sptensor(ttt(inv_Psi_ten,tensor(temp*K*inv_Psi)))-permute(sptensor(ttt(tensor(inv_Psi*K*temp),inv_Psi_ten),[1 3 2 4]));
%         inv_Psi_deri = -(ttt(inv_Psi_ten,tensor(temp*K*inv_Psi)))-permute(ttt(tensor(inv_Psi*K*temp),inv_Psi_ten),[1 3 2 4]);
% 
%         t0=tensor(-X'*temp);
%         t1=tensor(inv_Psi*K*temp);
%         t2=ttm(ttm(inv_Psi_deri,K,1),(K*temp)',4);
%         t3=ttt(tensor(K*inv_Psi),tensor(temp));
%         part1 = permute(ttt(t0,t1),[1 3 2 4]);
%         part2 = ttt(t0,t2,2,1);
%         part3 = ttt(t0,t3,2,1);
% 
%         J_K_deri = double(2*ttt(tensor(diff),part1+part2+part3,[1 2],[1 4]));
% 
%         grad = zeros(N*p,1);
% 
%         for i=1:N
%             K_z = -diag(K(i,:)/options.sigma_alg^2)*(repmat(Z(i,:),N,1)-Z);
%             term1=J_K_deri(i,:)*K_z;
%             term2=J_K_deri(:,i)'*K_z;
%             grad((i-1)*pp+1:i*pp) = (term1+term2)'; 
%         end
%         out = grad;
end