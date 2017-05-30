function [alp_est, K_est_new, Z_est_new] = find_alp(err,mag_err,K_est,U_X,options)


%% The optimization in continuous manner doesn't work
% options_optim = optimoptions(@fminunc,'Display','iter-detailed',...
%     'Algorithm','quasi-newton','SpecifyObjectiveGradient',false,...
%     'MaxIterations',1000,'MaxFunctionEvaluations',5e3);
% fun = @(alp)cost_find_alp(alp,err,K_est,U_X,options);
% alp0 = mag_err+0.5;
% [alp_est,fval,exitflag,output] = fminunc(fun,alp0,options_optim);
% K_est_new = K_est + alp_est*(err*err');
% [Z_est_new,lambda_est] = IGaussian_Kernel_vect(K_est_new,options);

%% Do it in non-continuous manner
alp_set = 0:5:300;
err_tot = zeros(length(alp_set),1);
for i=1:length(alp_set)
    err_tot(i) = cost_find_alp(alp_set(i),err,K_est,U_X,options);
end
[~,ind] = min(err_tot);
alp_est = alp_set(ind(1));
K_est_new = K_est + alp_est*(err*err');
K_est_new = max(K_est_new,options.delta);
[Z_est_new,lambda_est] = IGaussian_Kernel_vect(K_est_new,options);

end