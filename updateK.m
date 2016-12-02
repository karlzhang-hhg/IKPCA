function K_est = updateK(X,A,B,options,lam)

[N,n] = size(X);
% B = X'-a*ones(1,N);

%lam = 0.001;
K_est = (lam*eye(N)+A'*A)\A'*B;
end