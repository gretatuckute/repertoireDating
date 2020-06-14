clear all 
close all 
home
%
if 1
    fprintf('adding tools to path \n');
    addpath('~/MyCodes/repertoireDating//');
end 
%
train_sess='train_Apr27_19-14-13';
dat_dir='/Users/eghbalhosseini/MyData/LearningDynamics/train_results_matlab/';
train_files=dir(strcat(dat_dir,train_sess,'/*.mat'));
d_train=arrayfun(@(x) {strcat(train_files(x).folder,'/',train_files(x).name)}, 1:length(train_files));
%subsample and construct a data matrix 
nSamples=500;
nEpochs=length(d_train);
data=[];
tars=[];
subEpoch=[];
epoch=[];
for i=1:length(d_train)
    t=load(d_train{i});
    unique_cell=mat2cell(unique(t.batch),1,ones(1,length(unique(t.batch))));
    batch_idx_cell=cellfun(@(x) find(t.batch==x),unique_cell,'uni',false);
    batchs=1:length(batch_idx_cell);
    batchs_=repmat(batchs,nSamples,1);
    batchs=reshape(batchs_,[],1);
    batch_subsample=cell2mat(cellfun(@(x) randperm(length(x),nSamples)+x(1)-1,batch_idx_cell,'uni',false));
    data_subsample=double(t.fc(batch_subsample,:));
    tar_subsample=double(t.target(batch_subsample))';
    batch_sub=double(t.batch(batch_subsample))';
    temp=unique(batch_sub);
    bath_sub_idx=sum(cell2mat(arrayfun(@(x) x*(batch_sub==temp(x)),1:length(temp),'UniformOutput',false)),2);
    data=[data;data_subsample];
    subEpoch=[subEpoch;bath_sub_idx];
    tars=[tars;tar_subsample];
    epoch=[epoch;i+0*tar_subsample];
end 
productionTime = (1:length(epoch))'; 
%% 
NNids = knnsearch(data, data, 'K', 50); 
NNids = NNids(:, 2:end); 
figure;
imagesc(NNids)
%% 
[RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids, epoch, subEpoch);
repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, 1:length(d_train));

repertoireDating.renditionPercentiles(NNids, epoch,  'doPlot', true);
repertoireDating.renditionPercentiles(NNids, epoch, 'valid', epoch == 5, 'doPlot', true,'percentiles',[5,50,95]);
%%
MM = repertoireDating.mixingMatrix(NNids, epoch, 'doPlot', true);
RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);
stratMM = repertoireDating.stratifiedMixingMatrices(data, epoch, subEpoch, RP);
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);
%% 
repertoireDating.visualizeStratifiedMixingMatrix(avgStratMM, stratMM);


