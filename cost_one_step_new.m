function [f,grad]=cost_one_step_new(Z,X,options,k_func)
%%Follow the 20170522-weekly-report. Do one-step optimization to get Z with
%%new cost function
% Z: temporary variation sources (variable of the function)
% X: observations
% options: the options in ini_options.m
% flag: 
%   val: function value
%   grad: gradient vector

%Z = reshape(Z,[options.p,options.N])';

%%Code
lam_AK = options.lam_AK;

K = k_func(Z,options);

[N,n] = size(X);
[~,pp] = size(Z);

I = eye(N);
one_N = ones(N,N);
Psi = lam_AK*I+K;
Psi_inv = I/Psi;
temp1 = (I-Psi_inv*K);
temp11 = X'*Psi_inv;
temp2 = 0;
for i=1:n
    temp2 = temp2 + temp11(i,:)*K*temp11(i,:)';
end

f = sum(sum((X'*temp1).^2))+lam_AK*temp2;

if nargout > 1
    %Derivative of the approximated cost w.r.t. K
    temp3 = X*X';
    temp4 = Psi_inv*temp3*Psi_inv;
    temp5 = temp4*K*Psi_inv;
    J_K_deri = -2*Psi_inv*temp3*(temp1*temp1')+lam_AK*(temp4-temp5-temp5');
    grad = zeros(N,pp);
    for i=1:N
        K_z_deri = -diag(K(i,:)/options.sigma_alg^2)*(repmat(Z(i,:),N,1)-Z);
        grad(i,:) = (J_K_deri(i,:)+J_K_deri(:,i)')*K_z_deri;
    end
end

end