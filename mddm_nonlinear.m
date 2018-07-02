function [C lambda] = mddm_nonlinear(Q, L, projtype, mu, dim_para)
% mddm_nonlinear tackles nonlinear dimensionality reduction of multi-label problem through the method proposed in [1,2].
%
%    Syntax
%
%       [C lambda] = mddm_nonlinear(Q, L, projtype, mu, dim_para)
%
%    Description
%
%       mddm_nonlinear takes,
%           Q                - A NxN kernel matrix for features, where N is the number of data. 
%                              For uncorrelated subspace dimensionality reduction, Q must be centered beforehand.
%           L                - A NxN kernel matrix for labels.
%           projtype         - The parameter for the projection type, can takes
%                                'proj'        for uncorrelated projection dimensionality reduction
%                                'spc'         for uncorrelated subspace dimensionality reduction
%           mu               - The regularization parameter for uncorrelated subspace dimensionality reduction, in [0, 1]
%           dim_para         - The parameter for the final dimension, can takes:
%                                0:            keep the original dimension
%                                (0, 1):       dim_para is thr [1]
%                                [1, +\inf):   dim_para is d [1]
%
%      and returns,
%           C                - The obtained coefficients of projection
%           lambda           - The corresponding eigenvalues
%
% [1] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. ACM Transactions on Knowledge 
%     Discovery from Data.
% [2] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. In: AAAI'08, Chicago, IL, 2008, 
%     pp.1503-1505.


N = size(L,1);
tmpL = L - repmat(mean(L,1),N,1);
HLH = tmpL - repmat(mean(tmpL,2),1,N);

S = Q * HLH * Q;

if strcmp(projtype,'proj')
    B = Q + 1e-6 * eye(N);
else
    B = mu * Q * Q + (1 - mu) * Q;
end

clear Q L;

[tmp_C tmp_lambda] = eig(S, B);
tmp_C = real(tmp_C);
tmp_lambda = real(diag(tmp_lambda));
[lambda order] = sort(tmp_lambda, 'descend');
C = tmp_C(:,order);
clear tmp_C;

proper_dim = getProperDim(lambda, dim_para);
C = C(:,1:proper_dim);
lambda = lambda(1:proper_dim);