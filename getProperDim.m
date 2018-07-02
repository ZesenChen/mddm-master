function proper_dim = getProperDim(lambda, dim_para)
% getProperDim transform the dim_para to the required dimension.
%
%    Syntax
%
%       proper_dim = getProperDim(lambda, dim_para)
%
%    Description
%
%       getProperDim takes,
%           lambda           - The predicted label of testing data
%           dim_para         - The parameter for the final dimension, can takes:
%                                0:            keep the original dimension
%                                (0, 1):       dim_para is thr [1]
%                                [1, +\inf):   dim_para is d [1]
%
%      and returns,
%           proper_dim       - The required dimension
%
% [1] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. ACM Transactions on Knowledge 
%     Discovery from Data.
% [2] Y. Zhang and Z.-H. Zhou. Multi-label dimensionality reduction via dependency maximization. In: AAAI'08, Chicago, IL, 2008, 
%     pp.1503-1505.

if dim_para == 0
    proper_dim = length(lambda);
    return;
end

if dim_para < 1 % use thr
    thr = dim_para;
    sum_lambda = sum(lambda);
    lambda_num = length(lambda);
    tmp_lambda = 0;                
    for lind = 1 : lambda_num
        tmp_lambda = tmp_lambda + lambda(lind);
        if tmp_lambda >= thr * sum_lambda
            proper_dim = lind;
            break;
        end
    end
else % use d
    proper_dim = dim_para;
end