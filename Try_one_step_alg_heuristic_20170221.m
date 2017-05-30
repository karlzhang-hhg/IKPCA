%% Try optimization function in one step by first getting the OLS sol to A and a and then optimize w.r.t Z
%1. Generate the observations of Gaussian profiles, X, using variation sources Z. 
%2. Then, calculated the optimal solution of the one-step algorithm. 


%% Generate Data
clear();
% Initialize parameters
options = ini_options();

% File prefix
pr = 1;
name = '4-';

% Specify the seed of random number generator
rngn = 2;
rng(rngn); %Set seed for random generator

% Generate random (p-dimensional) z's which are sources of variation
N = options.N; %Number of data points 
p = options.p; %Dimension of variation sources
dd = options.dd;
psize = options.psize;

I = eye(N);
One = ones(N);
Z = rand(N,p); %N of p-dimenional points z

title_text1 = 'Originally generated 2-d data as variation source';
scat_func_tag = 1;
scatter_label2d_func = @scatter_label2d;
if scat_func_tag == 1
    tag = 0; %tag for showing index of data in 2d scatter plot (0: not showing)
    color = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];
    scatter_label2d_func(Z,title_text1,dd,tag,psize,color) %Plot scatter plot of Z
elseif scat_func_tag == 2
    tag = 1;
    color = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),0.3*ones(N,1)];
    a = int16(Z(:,2)*100); b = num2str(a); c = cellstr(b); %Labels
    scatter_label2d_func(Z,title_text1,dd,tag,psize,color,c) %Plot scatter plot of Z
end
if pr == 1
    saveas(gca,[options.cwd,[name,'0']],'jpg');
    saveas(gca,[options.cwd,[name,'0']],'fig');
end
% Generated data set 1
% %Generate n-D embeded data at higher-dimensional space
% n = 10; %Dimension of observed data
% sigma_kernel = 0.3;
% [X,K] = generate_GK_nd(Z,n,sigma_kernel);

% Generated data set 2
%Generate N Gaussian profile images (2-D) as high-dimensional observations
%sigma_data: the sigma of those Gaussian profiles
%l: the largest index of pixel in one side of images of those Gaussian
sigma_data = options.sigma_data;
l = options.l;
n = (l+1)^2;
X = generate_Gaussian_profile(Z*l,N,sigma_data,l);


%========================================================================================
%Plot PCA score to see if there is any nonlinear pattern in the generated
%observed data
pct = options.pct; %the percentage of threhold eigen-values
pc1 = options.pc1; %The index of the first component to be plotted
pc2 = options.pc2;
pc3 = options.pc3;
az = options.az;
el = options.el;
title_text2 = 'Principle components of';
color_3D = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];

%Standardize each coordinates of X; in other words, for each column,
%substract means and divide by unbiased standard deviation of that column
std_X = std(X,0,1); 
X_std = (X-repmat(mean(X,1),N,1))/diag(std_X);
%X_std = X;

%PCA on standardized observations
title_text3 = 'Standardized version: Principle components of';
[PC_std_ind,eig_values_std] = scatter_PCA_3d(X_std,pc1,pc2,pc3,pct,title_text3,color_3D,psize,az,el);
if pr == 1
    saveas(gca,[options.cwd,[name,'1']],'jpg');
    saveas(gca,[options.cwd,[name,'1']],'fig');
end

%%Inverse KPCA
%In svd (singular value decomposition, the singular values are sorted
%in non-increasing order.
[V,D,U] = svd(X_std/sqrt(N-1),'econ');
%Principle components of X (PCA scores)
PC_X = V*D(:,1:length(PC_std_ind));
%The standardized observations doesn't have full column rank, so we
%should pick out those columns of V that correspond to non-zero
%singular values.

%Plot 2-D scattering plot of first two coordinates obtained by PCA
title_text_PCA = 'Principle components of observations X';
scatter_label2d_func(PC_X(:,[1,2])/diag(sqrt(var(PC_X(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2])))),title_text_PCA,dd,tag,psize,color)
if pr == 1
    saveas(gca,[options.cwd,[name,'2']],'jpg');
    saveas(gca,[options.cwd,[name,'2']],'fig');
end


%% Test the the cost_one_step function by numerical finite difference method
% tic;
% [val,grad] = cost_one_step(Z,X,options);
% toc;
% tic;
% est_grad = finite_diff(@cost_one_step,Z,X,options,1e-6);
% toc;

%% Optimize w.r.t Z
tic;
d_samp = (randsample(N,options.downsmp*N))'; %Random sample index of observations
options_optim = optimoptions(@fminunc,'Display','iter-detailed',...
    'Algorithm','quasi-newton','SpecifyObjectiveGradient',false,...
    'MaxIterations',1000,'MaxFunctionEvaluations',1e6);
    fun = @(Z)cost_one_step_heuristic(Z,X,d_samp,options);
%     Z0 = reshape(PC_X(:,[1,2])',[1,N*p]);
    Z0 = PC_X(:,[1,2]);
%     Z0 = Z;
    %Z0 = reshape(Z',[1,N*p]);
    [Z_est,fval,exitflag,output] = fminunc(fun,Z0,options_optim);
    Z_est_m = Z_est;
    %Z_est_m = reshape(Z_est,[p,N])';
    title_text2 = 'Estimate variaton sources, Z, by quasi-netwon, Z_0 = X_{PCA} (iter=1000000)';
toc;
if scat_func_tag == 1
    tag = 0; %tag for showing index of data in 2d scatter plot (0: not showing)
    scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color) %Plot scatter plot of Z
elseif scat_func_tag == 2
    tag = 1;
    scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color,c) %Plot scatter plot of Z
end

if pr == 1
    saveas(gca,[options.cwd,[name,'3']],'jpg');
    saveas(gca,[options.cwd,[name,'3']],'fig');
end

