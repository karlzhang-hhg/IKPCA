%% Try optimization function to obtain variation sources Z 
%Generate the observations of Gaussian profiles, X, using variation sources
%Z. Then, use the basic model proposed by Prof. Apley to get the constant
%matrix A. Starting with A and X, use optimization function of MATLAB to
%find out the variation sources to see if it can return the correct
%answers.

%% Generate Data
% Initialize parameters
options = ini_options();

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
scatter_label2d_func(PC_X(:,[1,2]),title_text_PCA,dd,tag,psize,color)
% saveas(gca,[options.cwd,['2-3']],'jpg');
% saveas(gca,[options.cwd,['2-3']],'fig');

%% Estimate Variation Sources 
% Given vector a, find true coefficient matrix A
sigma_alg = options.sigma_alg;
a = ones(n,1);
K_true = Gaussian_Kernel(Z,sigma_alg);
B = X'-a*ones(1,N);

% A_true = B/K_true;
% In calculation, I found that the K_true is close to singular, so that I
% need to use regularized least-square

% Regularized coefficients
lambda = 10.^(-20:-1);

% Error of estimation
err = zeros(1,length(lambda));
for i = 1:length(lambda)
    A_MAP = ((lambda(i)*eye(N)+K_true*K_true)\K_true*(X-ones(N,1)*a'))';
    err(i) = sum(sum((B-A_MAP*K_true).^2))/n/N;
    i
end

% There still some round-off error among the first 10 lambda. Choose the
% 11st lambda and use that to approximately calculate true coefficient
% matrix A.

A_app = ((lambda(11)*eye(N)+K_true*K_true)\K_true*B')';

% options_optim = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
% problem.options = options_optim;
% problem.x0 = rand(N,p);
% problem.objective = @(Z)cost_func(Z,A_app,B,sigma_alg);
% problem.solver = 'fminunc';

% flag == 2: trust-region
% flag == 1: quasi-newton

flag = 1;
if flag == 2
    options_optim = optimoptions(@fminunc,'Display','iter-detailed','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    fun = @(Z)cost_func(Z,A_app,B,sigma_alg);
%     Z0 = reshape(PC_X(:,[1,2])',[1,N*p]);
    Z0 = rand(1,N*p);
    %Z0 = reshape(Z',[1,N*p]); % Reshape the initial N-by-p matrix Z into a row vector;
    [Z_est,fval,exitflag,output] = fminunc(fun,Z0,options_optim);
    Z_est_m = reshape(Z_est,[p,N])'; % Reshape the row vector Z_est_m into a N-by-p matrix
    title_text2 = 'Estimate variaton sources, Z, by trust-region, Z_0 = random';
else
    options_optim = optimoptions(@fminunc,'Display','iter-detailed','Algorithm','quasi-newton','SpecifyObjectiveGradient',true,'MaxIterations',1000);
    fun = @(Z)cost_func(Z,A_app,B,sigma_alg);
%     Z0 = reshape(PC_X(:,[1,2])',[1,N*p]);
    Z0 = rand(1,N*p);
    %Z0 = reshape(Z',[1,N*p]);
    [Z_est,fval,exitflag,output] = fminunc(fun,Z0,options_optim);
    Z_est_m = reshape(Z_est,[p,N])';
    title_text2 = 'Estimate variaton sources, Z, by quasi-netwon, Z_0 = random';
end


if scat_func_tag == 1
    tag = 0; %tag for showing index of data in 2d scatter plot (0: not showing)
    scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color) %Plot scatter plot of Z
elseif scat_func_tag == 2
    tag = 1;
    scatter_label2d_func(Z_est_m,title_text2,dd,tag,psize,color,c) %Plot scatter plot of Z
end
saveas(gca,[options.cwd,['1-2']],'jpg');
saveas(gca,[options.cwd,['1-2']],'fig');