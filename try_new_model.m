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
rank_X_std = length(PC_std_ind);
V = V(:,1:rank_X_std);

%Initialization of the algorithm
a = rand(1);
q = length(PC_std_ind);
trans_PC_X = PC_X - repmat(a,N,q);
k =0;
l = 0;
while 1
    k = k+1;
    display(['k=',num2str(k),', ',num2str(a)]);
    norm_sq = sum(trans_PC_X.^2,2); 
    b = sum(norm_sq)/sum(norm_sq.^2);
    est_K = trans_PC_X*trans_PC_X';
    l = 0;
    a_cur = a;
    while 1
        l = l+1;
        min_est_K = min(min(est_K));
        max_est_K = max(max(est_K));
        if (min_est_K > 0)
            break;
        else
            if (min_est_K <= 0)
                [i_min,j_min] = find(est_K == min_est_K,1);
                a_next = proj_a(PC_X,q,a_cur,b,i_min,j_min,1,options);
                trans_PC_X = PC_X - repmat(a_next,N,q);
                est_K = trans_PC_X*trans_PC_X';
                if (l>10) 
                    break;
                end
            end
%             max_est_K = max(max(est_K));
%             if (max_est_K > 1/b)
%                 [i_max,j_max] = find(est_K == max_est_K,1);
%                 a_next = proj_a(PC_X,q,a_cur,b,i_max,j_max,2,options);
%                 trans_PC_X = PC_X - repmat(a_next,N,q);
%                 est_K = trans_PC_X*trans_PC_X';
%             end
        end
        a_cur = a_next;
        display(['l=',num2str(l),', ',num2str(a_cur)]);
    end
    if (abs(a_next-a)<options.esp)
        display(abs(a_next-a));
        a = a_next;
        break;
    else
        a = a_next;
    end
end


K_est= b*est_K;
sigma_alg = options.sigma_alg;
[Z_est,lambda_est] = IGaussian_Kernel(K_est,sigma_alg,p);
title_text3 = ['Final estimated variation source based on data in feature space'];
scatter_label2d(Z_est,title_text3,dd,tag,psize,color) %Plot scatter plot of Z