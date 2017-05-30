function [f,grad]=cost_one_step_approx(Z,X,options,k_func)
%%Follow the 20170411-weekly-report. Do approximation to one-step optimization to get Z
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

K = k_func(Z,options);

[N,n] = size(X);
[~,pp] = size(Z);

I = eye(N);
one_N = ones(N,N);
temp = I-1/(N)*one_N;
temp1 = K*K;

% Used to test permutation
% res = randi([1 N],1,2);
% i1 = res(1);
% i2 = res(2);
% P = I;
% P(i1,i1)=0;
% P(i2,i2)=0;
% P(i1,i2)=1;
% P(i2,i1)=1;

diff = X'*(I-1/lam_A*temp*temp1)*temp;

f = sum(sum(diff.^2));

if nargout > 1
    %Derivative of the approximated cost w.r.t. K
    temp2 = X*X';
    temp3 = (I-1/lam_A*K*K*temp)*temp2*temp;
    J_K_deri = -2/lam_A*K*temp*(temp3+temp3');
    grad = zeros(N,pp);
    for i=1:N
        K_z_deri = -diag(K(i,:)/options.sigma_alg^2)*(repmat(Z(i,:),N,1)-Z);
        grad(i,:) = (J_K_deri(i,:)+J_K_deri(:,i)')*K_z_deri;
    end
end

end