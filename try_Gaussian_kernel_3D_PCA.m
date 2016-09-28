%%
clear();
rngn = 2;
rng(rngn); %Set seed for random generator

%Initialize options:
options = ini_options();
%Generate random (p-dimensional) z's which are sources of variation
N = options.N; %Number of data points 
p = options.p; %Dimension of variation sources
dd = options.dd;
psize = options.psize;

I = eye(N);
One = ones(N);
Z = rand(N,p); %N of p-dimenional points z

%Scatter plot of variation sources
color = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];
title_text1 = 'Variation sources';
tag = 0;
scatter_label2d(Z,title_text1,dd,tag,psize,color) %Plot scatter plot of Z

%Some parameters and color for 3D scatter plot
pct = options.pct; %the percentage of threhold eigen-values
pc1 = options.pc1; %The index of the first component to be plotted
pc2 = options.pc2;
pc3 = options.pc3;
az = options.az;
el = options.el;
color_3D = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];

%Generate kernel matrix (Gaussian Kernel)
sigma_alg = options.sigma_alg; 
K_Z = Gaussian_Kernel(Z,sigma_alg);

%Just try a linear case of PCA without centering
[V,D,U] = svd(Z,'econ');
title_text = 'PCA of Z without centering';
scatter_label2d(V*D,title_text,dd,tag,psize,color);

%Centering matrix;
M_cent = eye(N)-1/N*ones(N);
%PCA scores for centered feature vectors;
title_text = ['PCA scores of feature vectors of variation sources Z (\sigma_{alg}=',num2str(sigma_alg),') '];
[PC_ind,eig_values] = scatter_GK_PCA_3d(M_cent*K_Z*M_cent,pc1,pc2,pc3,pct,title_text,color_3D,psize,az,el);

%PCA scores for uncentered feature vectors;
title_text = ['Uncentered: PCA scores of feature vectors of variation sources Z (\sigma_{alg}=',num2str(sigma_alg),') '];
[PC_ind,eig_values] = scatter_GK_PCA_3d(K_Z,pc1,pc2,pc3,pct,title_text,color_3D,psize,az,el);

% saveas(gca,[options.cwd,['2-3']],'jpg');
% saveas(gca,[options.cwd,['2-3']],'fig');