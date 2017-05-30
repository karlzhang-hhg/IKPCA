function rl_err = finite_diff_ck(objfun,Z,X,options,e_step)
grad = objfun(Z,X,options,'grad');
val = objfunc(Z,X,options,'val');
[N,p] = size(Z);
est_grad = zeros(p*N,1);
for i=1:p*N
    Z_temp = Z;
    Z_temp(int64((i-mod(i,2))/2)+1,mod(i-1,2)+1) = Z_temp(int64((i-mod(i,2))/2)+1,mod(i-1,2)+1) + e_step;
    est_grad(i) = (objfun(Z_temp,X,options,'val')-val)/e_step;
end

rl_err = norm(est_grad-grad)/norm(est_grad);

end