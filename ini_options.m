function [options] = ini_options()
%%
% Initialize parameters in the run_alg.m
% options.N: sample size
% options.p: dimension of variation sources
% options.dd: the translation of text besides a data point in scatter plot
% options.psize: the size of marker in scatter plot
% options.sigma_alg: the sigma in algorithm (Gaussian Kernel)
% options.sigma_nois: the sigma of nois in algorithm when projecting kernel
%   matrix
% options.pct: the threshold percentage of variation in PCA on observations
%   X
% options.pc1,pc2,pc3: the index of three principle components
% options.az: azthmuthal angle of the view
% options.el: elevation angle of the view
% options.sigma_data: the sigma used to generate data set (images of
%   Gaussian profiles)
% options.l: length of images
% options.esp: the tolerance of norm of difference in Z between two 
%   consecutive iteration
% options.max_iter: maximum iteration number
%%
% Parameters for generating data
    options.N = 1000; %Number of data points 
    options.l = 20;
    options.p = 2; %Dimension of variation sources
    options.sigma_data = 5;
    options.sigma_nois = 0.0; %Standard Variation of noise
    options.rngn = 2; %Seed for random generator
    
% Parameters for running the algorithm    
    options.sigma_alg = 2;
    options.max_iter = 10;
    options.esp = 1e-6; %Tolerance for convergence

% Parameters for storing and plotting data
    options.cwd = '/Users/kungangzhang/Documents/OneDrive/Northwestern/Study/Courses/Independent Study/20170523-RKHS-Implementation-Alternating-Alg/figures/';
    options.pct = 0.99; %the percentage of threshold eigen-values
    %options.pct_pca = 0.99; %the percentage of threshold eigen-values in PCA
    options.pc1 = 1; %The index of the first component to be plotted
    options.pc2 = 2;
    options.pc3 = 3;
    options.az = 30;
    options.el = 25;
    options.dd = 0.02;
    options.psize = 10;
    
% Parameters for the one-step algorithm
    options.lam_A = 6e3;
    options.lam_a = 0;
    options.lam_Z = 0;
    options.lam_AK = 0.8e3;
    
% Parameters during iteration in new model
    options.delta = 1e-4;
    options.norm = 100;
    
% Parameters of embedding data
    options.k_neig = 3;
    
% Temporary parameters
    options.a = 0;
    
% Parameters for heuristic alg after 20170207
    options.alp1=0.1;
    options.alp2=0.2;
    
% Parameters for heuristic alg after 20170221
    options.downsmp = 0.1;
    
% Parameters for lifted brownian kernel
    options.a = 1;
    options.beta = 0.5;

% Parameters for finding alpha of the factor for projection error of col(X)
% on row(K)
    options.wei = 1;
    
    
% Parameter of hole in variation sources
    options.d = 0.3;
end