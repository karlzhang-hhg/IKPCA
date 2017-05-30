function [c,ceq] = fixvarc(Z)
%%The nonlinear equality constrains to fix the variance of each coordinates
%%of Z
c=[];%The empty constrain still needs to be specified
ceq = var(Z,1)'-1;