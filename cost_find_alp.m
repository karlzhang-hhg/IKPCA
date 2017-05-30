function f = cost_find_alp(alp,err,K_est,U_X,options)
%% cost function combining the error of projecting col(X) onto row(K_est) and the error of projecting col(X) onto row(K_est+alp*err*err')

K_est_new = K_est+alp*(err*err');
% K_est_new = (K_est_new+K_est_new')/2;
K_est_new = max(K_est_new,options.delta);
err1 = proj_err(K_est_new,U_X,options);
Z_est_new = IGaussian_Kernel_vect(K_est_new,options);
K_est_new_new = Gaussian_Kernel_vect(Z_est_new,options);
err2 = proj_err(K_est_new_new,U_X,options);

f = sum(sum(err1.^2+options.wei*err2.^2));

end