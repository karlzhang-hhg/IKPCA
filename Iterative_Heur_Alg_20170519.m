%% Try optimization function in one step by first getting the OLS sol to A and a and then optimize w.r.t Z
%1. Generate the observations of Gaussian profiles, X, using variation sources Z. 
%2. Then, iteratively update A and latent variable Z, with Z_ini = X_PCA. 
%3. Use nonlinear equality constrains to fix the variance of each
%   coordinates of Z.


%% Generate Data
clear();
% Initialize parameters
options = ini_options();

% File prefix
pr = 1;
name = '1-';

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
% Z = Z_gen(options.d,Z);
% options.N = size(Z,1);
% N = options.N;

title_text1 = 'Originally generated 2-d data as variation source';
scat_func_tag = 1;
scatter_label2d_func = @scatter_label2d;
if scat_func_tag == 1
    tag = 0; %tag for showing index of data in 2d scatter plot (0: not showing)
    color = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];
    mark = [127,92,398,440,268,356,157]';% b = num2str(a); c = cellstr(b); %Labels
    scatter_label2d_func(Z,title_text1,dd,tag,psize,color,mark) %Plot scatter plot of Z
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
scatter_label2d_func(PC_X(:,[1,2])/diag(sqrt(var(PC_X(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2])))),title_text_PCA,dd,tag,psize,color,mark)
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
profile on;
options_optim = optimoptions(@fmincon,'Display','iter-detailed',...
    'Algorithm','interior-point','SpecifyObjectiveGradient',false,...
    'MaxIterations',1000,'MaxFunctionEvaluations',2e4);

max_iter = options.max_iter;
I = eye(N);
Z_ini = PC_X;
K_ini = Gaussian_Kernel_vect(Z_ini,options);
A_ini = X'/(options.lam_A*I+K_ini);

Z0 = PC_X(:,[1,2])/diag(sqrt(var(PC_X(:,[1,2]))));
A0 = A_ini;

A = [];
b = [];
Aeq = [];
beq = [];
lb = [];
ub = [];
nonlcon = @fixvarc;
for i=1:max_iter
    display(i);
    fun = @(Z)cost_iter(Z0,X,A0,options,@Gaussian_Kernel_vect);
    [Z_est,fval,exitflag,output] = fmincon(fun,Z0,A,b,Aeq,beq,lb,ub,nonlcon,options_optim);
    
    title_text3 = ['(iter=',num2str(i),') Estimated variation source based on iterative and heuristic alg-20170519',10,'(\sigma_{data}=',...
        num2str(options.sigma_data),', \sigma_{alg}=',num2str(options.sigma_alg),', \lambda_{A}=',num2str(options.lam_A),...
        ', \lambda_{Z}=',num2str(options.lam_Z),') with symmetric K'];
    scatter_label2d_func(Z_est,title_text3,options.dd,0,options.psize,color,mark)
    if pr == 1
        saveas(gca,[options.cwd,[name,num2str(i-1)]],'jpg');
        saveas(gca,[options.cwd,[name,num2str(i-1)]],'fig');
        close(gcf);
    end
    
    K_est = Gaussian_Kernel_vect(Z_est,options);
    A0 = X'/(options.lam_A*I+K_est);
    norm(Z_est-Z0)/norm(Z0)
    Z0 = Z_est;
end

    %fun = @(Z)cost_one_step_approx(Z,X,options,@Gaussian_Kernel_vect);
    %fun = @(Z)cost_one_step(Z,X,options,@Gaussian_Kernel_vect);
    %fun = @(Z)cost_one_step(Z,X,options,@Lifted_Brownian_kernel);
%     Z0 = reshape(PC_X(:,[1,2])',[1,N*p]);
%     Z0 = Z;
profile viewer;
toc;

%     title_text2 = ['Estimate variaton sources (approx-cost), Z, by interior-point, Z_0 = X_{PCA}, fix variance',10,...
%         'Vectorized Gaussian kernel calculation (t=',num2str(toc),',iter=5e4',',\lambda_{A}=',num2str(options.lam_A),',\lambda_{a}=',num2str(options.lam_a),')'];
% if scat_func_tag == 1
%     tag = 0; %tag for showing index of data in 2d scatter plot (0: not showing)
%     scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color,mark) %Plot scatter plot of Z
% elseif scat_func_tag == 2
%     tag = 1;
%     scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color,c) %Plot scatter plot of Z
% end
% 
% if pr == 1
%     saveas(gca,[options.cwd,[name,'3']],'jpg');
%     saveas(gca,[options.cwd,[name,'3']],'fig');
% end

