function K_est = solve_K(A,B,lambda)
%Solve the SK+KS=R problem in 20161104 weekly report by assembling the
%coefficient matrix.
[n,N] = size(A);    
S = lambda*eye(N)+A'*A;
R_temp = B'*A;
R = R_temp+R_temp';
%%Without implicitly apply symmetric constraints on K
Const_M = kron(sparse(eye(N)),S)+kron(S,sparse(eye(N)));
size(Const_M)
%rank(Const_M)
K_est = reshape(Const_M\reshape(R,N^2,1),N,N);
end