function est_grad = finite_diff(objfun,Z,X,options,e_step)
%% Use the finite difference method to calculate gradient of the objective function
%grad = objfun(Z,X,options,'grad');
val = objfun(Z,X,options);
[N,p] = size(Z);
est_grad = zeros(p*N,1);
for i=1:p*N
    Z_temp = Z;
    Z_temp(int64((i-mod(i-1,2)-1)/2)+1,mod(i-1,2)+1) = Z_temp(int64((i-mod(i-1,2)-1)/2)+1,mod(i-1,2)+1) + e_step;
    val_temp = objfun(Z_temp,X,options);
    est_grad(i) = (val_temp-val)/e_step;
end

end