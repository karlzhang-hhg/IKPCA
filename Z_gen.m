function Z_new=Z_gen(d,Z)
% Generate random variation sources with a hole in the center
% Input:
%   d: the edge length of the hole
%   Z: origianl generated variation sources
% Output:
%   Z_new: the variation sources (each row is a sample) after thinning

Z_new=Z(~((abs(Z(:,1)-0.5)<=d/2)&(abs(Z(:,2)-0.5)<=d/2)),:);

end