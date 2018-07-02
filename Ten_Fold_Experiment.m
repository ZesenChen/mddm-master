base_path = 'data\';
save_path = '';
file_name = 'yeast';%'rcv1subset1_top944';
file_format = '.mat';
set = [file_name,file_format];
finaldata = [];
for jj = 1:1
    path = [base_path,set];
    dataset = load(path);
    if(isfield(dataset,'train_data'))
        data = [dataset.train_data; dataset.test_data];
        target = [dataset.train_target, dataset.test_target];
    else
        data = zscore(dataset.data);
        target = dataset.target;
    end
    %data = zscore(data);
    target(target==0)=-1;
    [data_num,tmp] = size(data);
    for i = data_num:-1:1
        swap_num = randint(1,1,[1,i]);
        tmpd = data(i,:);
        tmpt = target(:,i);
        data(i,:) = data(swap_num,:);
        target(:,i) = target(:,swap_num);
        data(swap_num,:) = tmpd;
        target(:,swap_num) = tmpt;
    end
    %交叉验证次数
    fold_num = 10;
    test_num = round(data_num/fold_num);
    test_instance = cell(fold_num,1);
    for i = 1:fold_num-1
        test_instance{i,1} = (i-1)*test_num+1:i*test_num;
    end
    test_instance{fold_num,1} = (fold_num-1)*test_num+1:data_num;

    ratio = 0.1;
    mu = 0.1;
    projtype = 'proj';
    result = zeros(fold_num,5);
    for i = 1:fold_num
        disp(['The ',num2str(i),'-th fold is going on...']);
        train_data = data;
        train_target = target;
        test_data = data(test_instance{i,1},:);
        test_target = target(:,test_instance{i,1});
        train_data(test_instance{i,1},:) = [];
        train_target(:,test_instance{i,1}) = [];
        [Outputs,Pre_Labels] = LMDDM(train_data,train_target,test_data, test_target, projtype, mu, ratio);
        HammingLoss=Hamming_loss(Pre_Labels,test_target);
        RankingLoss=Ranking_loss(Outputs,test_target);
        OneError=One_error(Outputs,test_target);
        Coverage=coverage(Outputs,test_target);
        Average_Precision=Average_precision(Outputs,test_target);
        result(i,:) = [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision];
    end 
    disp('Hamming Loss,Ranking Loss,One Error,Coverage,Average Precision');
    disp(mean(result,1));
end