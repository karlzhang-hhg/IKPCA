%% Try Non-negative matrix factorization (nnmf()) in matlab to factorize the
%% observation X matrix

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

% %% Call matlab function nnmf() on X;
% %initialize the K matrix with PCA score:
% 
% %Plot PCA score to see if there is any nonlinear pattern in the generated
% %observed data
% pct = options.pct; %the percentage of threhold eigen-values
% pc1 = options.pc1; %The index of the first component to be plotted
% pc2 = options.pc2;
% pc3 = options.pc3;
% az = options.az;
% el = options.el;
% title_text2 = 'Principle components of';
% color_3D = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];
% 
% %Standardize each coordinates of X; in other words, for each column,
% %substract means and divide by unbiased standard deviation of that column
% std_X = std(X,0,1); 
% X_std = (X-repmat(mean(X,1),N,1))/diag(std_X);
% %X_std = X;
% 
% %PCA on standardized observations
% title_text3 = 'Standardized version: Principle components of';
% [PC_std_ind,eig_values_std] = scatter_PCA_3d(X_std,pc1,pc2,pc3,pct,title_text3,color_3D,psize,az,el);
% 
% %%Inverse KPCA
% %In svd (singular value decomposition, the singular values are sorted
% %in non-increasing order.
% [V,D,U] = svd(X_std/sqrt(N-1),'econ');
% %Principle components of X (PCA scores)
% PC_X = V*D(:,1:length(PC_std_ind));
% %The standardized observations doesn't have full column rank, so we
% %should pick out those columns of V that correspond to non-zero
% %singular values.
% 
% %Set options for nnmf;
% opt = statset('Maxiter',1000,'Display','iter');
% 
% 
% [W,H] = nnmf(X',N,'w0',W0,'h0',PC_X*PC_X',...
%     'options',opt,...
%     'algorithm','als');
% 
% % Try both "als" and "mult" algorithm, but output following error:
% % Error using nnmf>checkmatrices (line 360)
% % H must be a matrix of non-negative values


%% Call nnmf() function on X;
%% Random initialization
W0 = rand(n,N);
H0 = rand(N,N);
[W,H] = nnmf(X',N,'w0',W0,'h0',H0,...
    'options',opt,...
    'algorithm','mult');
% 'als' doesn't give me good result, but 'mult' works;


