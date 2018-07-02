function [P lambda] = mddm_linear(train, L, projtype, mu, dim_para)
% mddm_linear tackles linear dimensionality reduction of multi-label problem through the method proposed in [1,2].
%
%    Syntax
%
%       [P lambda] = mddm_linear(X, L, projtype, mu, dim_para)
%
%    Description
%
%       mddm_linear takes,
%           train                - A NxD matrix, where D is the dimension of data and N is the number of data. 
%                                  Each column is a sample. For uncorrelated subspace dimensionality reduction,
%                                  X must be centered beforehand.
%           L                - A NxN matrix, the kernel matrix for label
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
%           P                - The obtained projection
%           lambda           - The corresponding eigenvalues
%
% [1] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. ACM Transactions on Knowledge 
%     Discovery from Data.
% [2] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. In: AAAI'08, Chicago, IL, 2008, 
%     pp.1503-1505.


X = train';
[D N] = size(X);
tmpL = L - repmat(mean(L,1),N,1);
HLH = tmpL - repmat(mean(tmpL,2),1,N);

S = X * HLH * X';

if strcmp(projtype,'proj')
    B = eye(D);
else
    B = mu * X * X' + (1 - mu) * eye(D);
end

clear X L;


[tmp_P tmp_lambda] = eig(S, B);
tmp_P = real(tmp_P);
tmp_lambda = real(diag(tmp_lambda));
[lambda order] = sort(tmp_lambda, 'descend');
P = tmp_P(:,order);

proper_dim = getProperDim(lambda, dim_para);
P = P(:,1:proper_dim);
lambda = lambda(1:proper_dim);