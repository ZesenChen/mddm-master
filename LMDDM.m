function [Outputs,Pre_Labels] = LMDDM(train_data,train_target,test_data, test_target, projtype, mu, ratio)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
    
[num_train, dim] = size(train_data);
[num_class, num_test] = size(test_target);

P = cell(num_class,1);
Models=cell(num_class,1);
for i = 1:num_class
    disp(['Performing mddm features reduction for the ',num2str(i),'-th class']);
    p_idx=find(train_target(i,:)==1);
    n_idx=setdiff([1:num_train],p_idx);
        
    p_data=train_data(p_idx,:);
    n_data=train_data(n_idx,:);
        
    k=min(ceil(length(p_idx)*ratio),ceil(length(n_idx)*ratio));
    L=train_target(i,:)'*train_target(i,:);
    [p, lambda] = mddm_linear(train_data, L, projtype, mu, k);
    P{i,1} = p;
    new_train_data = train_data*p;
    Models{i,1}=svmtrain(train_target(i,:)',new_train_data,'-t 0 -b 1 -q');   
end
Pre_Labels=[];
Outputs=[];

for i = 1:num_class
    disp(['Predicting the ',num2str(i),'-th class']);
    new_test_data = test_data*P{i,1};
    [predicted_label,accuracy,prob_estimates]=svmpredict(test_target(i,:)',new_test_data,Models{i,1},'-b 1');
    if(isempty(predicted_label))
        predicted_label=train_target(i,1)*ones(num_test,1);
        if(train_target(i,1)==1)
            Prob_pos=ones(num_test,1);
        else
            Prob_pos=zeros(num_test,1);
        end
        Outputs=[Outputs;Prob_pos'];
        Pre_Labels=[Pre_Labels;predicted_label'];
        else
            pos_index=find(Models{i,1}.Label==1);
            Prob_pos=prob_estimates(:,pos_index);
            Outputs=[Outputs;Prob_pos'];
            Pre_Labels=[Pre_Labels;predicted_label'];
        end
end
end

