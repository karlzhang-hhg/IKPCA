%% Heuristic algorithm based on the optimization results from the weekly report on 20170117
%1. Move the current K towards XX^T, with a small step. 
%2. Re-solve the variation-source problem of Z using inverse Gaussian and MDS.

%% Generate Data
clear();
% Initialize parameters
options = ini_options();

% File prefix
pr = 1;
name = '2-';

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
    saveas(gca,[options.cwd,[name,'Z_true']],'jpg');
    saveas(gca,[options.cwd,[name,'Z_true']],'fig');
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
    saveas(gca,[options.cwd,[name,'3d_PCA']],'jpg');
    saveas(gca,[options.cwd,[name,'3d_PCA']],'fig');
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
    saveas(gca,[options.cwd,[name,'2d_PCA']],'jpg');
    saveas(gca,[options.cwd,[name,'2d_PCA']],'fig');
end

%% 
max_iter = options.max_iter;

Z0 = PC_X(:,[1,2]);

%Z_ini =  PC_X(:,[1,2])/diag(sqrt(var(PC_X(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2]))));
Z_ini =PC_X(:,[1,2]);
K_est = Gaussian_Kernel(Z_ini,options.sigma_alg);
for i=1:max_iter
    %IGaussian_Kernel_old means not using conjugate symmetry of inverted
    %elements of K
    [Z_est,lambda_est] = IGaussian_Kernel_old(K_est,options.sigma_alg,options.p);
    %[Z_est,lambda_est] = IGaussian_Kernel_iter(K_est,options.sigma_alg,options.pct_pca);
    %'Symmetric K' means when calculating the inverted elements of K, I
    %didn't use conjugate symmetry, while 'asymmetric K' means I use conjugate symmetry 
    title_text3 = ['(iter=',num2str(i),') Estimated variation source based on heuristic alg-20170214',10,'(\sigma_{data}=',...
        num2str(options.sigma_data),', \sigma_{alg}=',num2str(options.sigma_alg),', \alpha_{1}=',num2str(options.alp1),...
        ', \alpha_{2}=',num2str(options.alp2),') with symmetric K'];
    scatter_label2d_func(Z_est,title_text3,options.dd,0,options.psize,color)
    if pr == 1
        saveas(gca,[options.cwd,[name,num2str(i-1)]],'jpg');
        saveas(gca,[options.cwd,[name,num2str(i-1)]],'fig');
        close(gcf);
    end
    
    %K_est = Gaussian_Kernel(Z_est(:,[1,2])/diag(sqrt(var(Z_est(:,[1,2]))))*diag(sqrt(var(Z(:,[1,2])))),options.sigma_alg);
    K_est = Gaussian_Kernel(Z_est(:,[1,2])/diag(sqrt(var(Z_est(:,[1,2])))),options.sigma_alg);%Just let the variance of each Z_est to be unit
    K_est = K_est + options.alp1*(options.alp2*(X_std*X_std')-K_est);
    i
end


