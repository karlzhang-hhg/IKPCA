%% Iterative algorithm using old models
clear();
options = ini_options();
% Generate Gaussian profiles and plot the variation sources and PCA (2d and 3d)
text = '10-';
[Z,X,color,PC_X,V,D,PC_std_ind,eig_values_std] = genGaussianDat(options,text);
N = options.N;
n = (options.l+1)^2;
a = ones(n,1);
X_til_est = (X'-a*ones(1,N))';
max_iter = options.max_iter;
% Regularization constant in estimating Psi
lam1 = 1;

% Iterative algorithm
Z_ini =  PC_X/diag(sqrt(var(PC_X)))*sqrt(var(Z(:,1)));
K_est = Gaussian_Kernel(Z_ini,options.sigma_alg);
Psi_est = zeros(n);

num = 4;
for i=1:max_iter
    %% Given a and K, estimate Psi
    temp = (lam1*eye(n)+X_til_est'*X_til_est);
    Psi_est = temp\(X_til_est'*K_est*X_til_est+lam1^2*Psi_est)/temp';%Prof. Apley proposed regularization
    %Psi_est = temp\(X_til_est'*K_est*X_til_est)/temp';%My way of regularization
    Psi_est = (Psi_est+Psi_est')/2;
    
    %% Given Psi and a, estimate Z
    K_est = X_til_est*Psi_est*X_til_est';
    K_est = (K_est+K_est')/2;
    %IGaussian_Kernel_old means not using conjugate symmetry of inverted
    %elements of K
    %[Z_est,lambda_est] = IGaussian_Kernel_new(K_est,options.sigma_alg,options.p,options.pct,num);
    %num = num - 1;%Upper bound of number of components.
%     [Z_est,lambda_est] = IGaussian_Kernel_iter(K_est,options.sigma_alg,options.pct);
%     size(Z_est,2)
    [Z_est,lambda_est] = IGaussian_Kernel_old(K_est,options.sigma_alg,options.p);
    %'Symmetric K' means when calculating the inverted elements of K, I
    %didn't use conjugate symmetry, while 'asymmetric K' means I use conjugate symmetry 
    title_text3 = ['(iter=',num2str(i),') Estimated variation source based on old model-20161209',10,'(\sigma_{data}=',...
        num2str(options.sigma_data),', \sigma_{alg}=',num2str(options.sigma_alg),', \lambda_{\Psi}=',num2str(lam1),', pct=',num2str(options.pct),...
        ') with symmetric K'];
    scatter_label2d(Z_est,title_text3,options.dd,0,options.psize,color)
        saveas(gca,[options.cwd,[text,'-',num2str(i-1)]],'jpg');
        saveas(gca,[options.cwd,[text,'-',num2str(i-1)]],'fig');
        close(gcf);

    %% Given Z and Psi, estimate a
    K_est = Gaussian_Kernel(Z_est,options.sigma_alg);
    [R,Lam,U1] = svd(K_est);%Lam is N-by-N
    n
    thd_K = sing_th_ind(diag(Lam),options.pct)
    [Q,Sig,U2] = svd(Psi_est);%Sig is n-by-n
    N
    thd_Psi = sing_th_ind(diag(Sig),options.pct)
    thd = max(thd_K,thd_Psi);
    a_est = mean(X'-Q(:,1:thd)*diag(diag(Sig(1:thd,1:thd)).^(-0.5))*diag(diag(Lam(1:thd,1:thd)).^(0.5))*R(:,1:thd)',2);
    X_til_est = (X'-a_est*ones(1,N))';
    i
end