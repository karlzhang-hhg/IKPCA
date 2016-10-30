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
%%
n_fig = 9;%Used to save figures
%%

color = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];
title_text1 = 'Variation sources';
tag = 0;
scatter_label2d(Z,title_text1,dd,tag,psize,color) %Plot scatter plot of Z
if (n_fig > 0)
    saveas(gca,[options.cwd,[num2str(n_fig),'-0']],'jpg');
    saveas(gca,[options.cwd,[num2str(n_fig),'-0']],'fig');
end
% Generated data set 2
%Generate N Gaussian profile images (2-D) as high-dimensional observations
%sigma_data: the sigma of those Gaussian profiles
%l: the largest index of pixel in one side of images of those Gaussian
sigma_data = options.sigma_data;
l = options.l;
n = (l+1)^2;
X = generate_Gaussian_profile(Z*l,N,sigma_data,l);

%Some parameters and color for 3D scatter plot
pct = options.pct; %the percentage of threhold eigen-values
pc1 = options.pc1; %The index of the first component to be plotted
pc2 = options.pc2;
pc3 = options.pc3;
az = options.az;
el = options.el;
color_3D = [0.7*ones(N,1),Z(:,1)/max(Z(:,1)),Z(:,2)/max(Z(:,2))];

%% Iterative algorithm:
%Standardize each coordinates of X; in other words, for each column,
%substract means and divide by unbiased standard deviation of that column
std_X = std(X,0,1); 
X_std = (X-repmat(mean(X,1),N,1))/diag(std_X);
%X_std = X;

%%Inverse KPCA
%In svd (singular value decomposition, the singular values are sorted
%in non-increasing order.
[V_X_std,D_X_std,U_X_std] = svd(X_std/sqrt(N-1),'econ');
PC_std_ind = (1:sing_th_ind(diag(D_X_std),pct));
%Principle components of X (PCA scores)
PC_X_std = V_X_std*D_X_std(:,1:length(PC_std_ind));
%The standardized observations doesn't have full column rank, so we
%should pick out those columns of V that correspond to non-zero
%singular values.
rank_X_std = length(PC_std_ind);
%V_X_std = V_X_std(:,1:rank_X_std);

%PCA on standardized observations
title_text3 = 'Standardized version: Principle components of';
[PC_std_ind,eig_values_std] = scatter_PCA_3d(PC_X_std,pc1,pc2,pc3,pct,title_text3,color_3D,psize,az,el);
if (n_fig > 0)
    saveas(gca,[options.cwd,[num2str(n_fig),'-1']],'jpg');
    saveas(gca,[options.cwd,[num2str(n_fig),'-1']],'fig');
end

%% The heuristic method of data embedding
% %Initialization of the algorithm
% q = length(PC_std_ind);
% norm_PC_X = sqrt(sum(PC_X.^2,2));
% a = mean(diag(1./norm_PC_X)*PC_X,1)';
% a = a/norm(a);
% gamma = rand(1);
% trans_PC_X = PC_X + repmat(gamma*a',N,1);
% norm_PC_X = sqrt(sum(trans_PC_X.^2,2));
% scal = max(norm_PC_X)./norm_PC_X;
% trans_scal_PC_X = diag(scal)*trans_PC_X;
% k =0;
% l = 0;
% 
% 
% k = k+1;
% display(['k=',num2str(k),', ',num2str(gamma)]);
% norm_sq = sum(trans_scal_PC_X.^2,2); 
% b = sum(norm_sq)/sum(norm_sq.^2);
% est_K = trans_scal_PC_X*trans_scal_PC_X';
% l = 0;
% gamma_cur = gamma;
% while 1
%     l = l+1;
%     min_est_K = min(min(est_K));
%     max_est_K = max(max(est_K));
%     if (min_est_K > 0)
%         break;
%     else
%         if (min_est_K <= 0)
%             [i_min,j_min] = find(est_K == min_est_K,1);
%             gamma_next = proj_gamma(trans_scal_PC_X,a,b,i_min,j_min,1,options);
%             trans_PC_X = trans_scal_PC_X + repmat(gamma_next*a',N,1);
%             norm_PC_X = sqrt(sum(trans_PC_X.^2,2));
%             scal = max(norm_PC_X)./norm_PC_X;
%             trans_scal_PC_X = diag(scal)*trans_PC_X;
%             est_K = trans_scal_PC_X*trans_scal_PC_X';
%             if (l>10) 
%                 break;
%             end
%         end
% %             max_est_K = max(max(est_K));
% %             if (max_est_K > 1/b)
% %                 [i_max,j_max] = find(est_K == max_est_K,1);
% %                 a_next = proj_a(PC_X,q,a_cur,b,i_max,j_max,2,options);
% %                 trans_PC_X = PC_X - repmat(a_next,N,q);
% %                 est_K = trans_PC_X*trans_PC_X';
% %             end
%     end
%     gamma_cur = gamma_next;
%     display(['l=',num2str(l),', ',num2str(gamma_cur)]);
% end

%% Improve the accuracy of embedding
sigma_alg = options.sigma_alg;
rng(rngn+1);
Z_em = rand(N,p);
scatter_label2d(Z_em,['Z_{em}'],dd,tag,psize,color);

K_Z_em = Gaussian_Kernel(Z_em,sigma_alg);
[V_K_Z_em,D_K_Z_em,U_K_Z_em] = svd(K_Z_em,'econ');
% rank_cent_K_Z_em = sing_th_ind(diag(D_cent_K_Z_em),pct);
% PC_K_Z_em = V_K_Z_em*sqrt(D_K_Z_em(:,1:rank_X_std));
% mean_PC_K_Z = mean(PC_K_Z,1);
%Centering matrix;
M_cent = eye(N)-1/N*ones(N);
[V_cent_K_Z_em,D_cent_K_Z_em,U_cent_K_Z_em] = svd(M_cent*V_K_Z_em*D_K_Z_em,'econ');
rank_cent_K_Z_em = sing_th_ind(diag(D_cent_K_Z_em),pct);
rank_em = max(rank_X_std,rank_cent_K_Z_em);
PC_cent_K_Z_em = V_cent_K_Z_em*D_cent_K_Z_em(:,1:rank_em);
% cent_K_Z_em = M_cent*K_Z_em*M_cent;
% [V_cent_K_Z_em,D_cent_K_Z_em,U_cent_K_Z_em] = svd(cent_K_Z_em,'econ');
% rank_cent_K_Z_em = sing_th_ind(diag(sqrt(D_cent_K_Z_em)),pct);

%Plot PCA score for Gaussian kernel matrix (feature vectors)
title_text=['Centered:PCA scores of feature vectors of Kernel generated by random Z_{em} (\sigma_{alg}=',num2str(sigma_alg),') '];
%[PC_K_Z_ind,eig_values] = scatter_GK_PCA_3d(cent_K_Z_em,pc1,pc2,pc3,pct,title_text,color_3D,psize,az,el);
[PC_K_Z_ind,eig_values] = scatter_PCA_3d(PC_cent_K_Z_em,pc1,pc2,pc3,pct,title_text,color_3D,psize,az,el);
if (n_fig > 0)
    saveas(gca,[options.cwd,[num2str(n_fig),'-2']],'jpg');
    saveas(gca,[options.cwd,[num2str(n_fig),'-2']],'fig');
end

%Align the coordinates of observations and feature vectors
%Actually, we don't need to do rotation, because the PCA scores are
%coordinates of data in their principle directions with decreasing order of
%eigen-values, so that two sets of data (observations and feature vectors) 
%are in their own local coordinates, no matther what bases they are using
%(bases are just U_X_std and some unknow bases for centered kernel)

W = embedding_GK(PC_cent_K_Z_em,V_X_std*D_X_std(:,1:rank_em),options.k_neig);
Z_est = W*Z_em;
title_text3 = ['Final estimated variation source based on observational data embedding (\sigma_{alg}=',num2str(sigma_alg),')'];
scatter_label2d(Z_est,title_text3,dd,tag,psize,color) %Plot scatter plot of Z
if (n_fig > 0)
    saveas(gca,[options.cwd,[num2str(n_fig),'-3']],'jpg');
    saveas(gca,[options.cwd,[num2str(n_fig),'-3']],'fig');
end



% title_text3 = 'Translated version: Principle components of X';
% figure()
% scatter_3d(trans_scal_PC_X(:,[pc1,pc2,pc3]),title_text3,color_3D,psize);
% view(az,el);
% 
% K_est= b*est_K;
% [Z_est,lambda_est] = IGaussian_Kernel(K_est,sigma_alg,p);
% title_text3 = ['Final estimated variation source based on data in feature space'];
% scatter_label2d(Z_est,title_text3,dd,tag,psize,color) %Plot scatter plot of Z

% saveas(gca,[options.cwd,['3-1']],'jpg');
% saveas(gca,[options.cwd,['2-1']],'fig');