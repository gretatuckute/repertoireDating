clear all 
close all 
home

%% Figure specs

set(groot, ...
'DefaultFigureColor', 'w', ...
'DefaultAxesLineWidth', 0.5, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 8, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultLineLineWidth', 1, ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 8, ...
'DefaultTextFontName', 'Helvetica', ...
'DefaultAxesBox', 'off', ...
'DefaultAxesTickLength', [0.01 0.015]);
set(groot, 'DefaultAxesTickDir', 'out');
set(groot, 'DefaultAxesTickDirMode', 'manual');

%% Directories
data_dir = '/om/user/ehoseini/MyData/neural_manifolds/train_network_on_synthetic';

% Load the generated mat files, session of interest: (input, the model identifier)
model_identifier = 
KNN_files = dir(strcat(data_dir, model_identifier, '/extracted*.mat'));



%%
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder,'/',KNN_files(x).name)}, 1:length(KNN_files));
[~,idx]=sort([KNN_files.datenum]);
KNN_data = KNN_data(idx);
KNN_data'

% Subsample and construct a data matrix 
nSamples = 50; % Per point in time, i.e. per batch idx
nEpochs = length(KNN_data);

data = [];
targets = [];
subEpoch = [];
epoch = [];
%% 
for i=1:length(KNN_data)
    t=load(KNN_data{i});
    unique_cell=mat2cell(unique(t.batch),1,ones(1,length(unique(t.batch))));
    batch_idx_cell=cellfun(@(x) find(t.batch==x),unique_cell,'uni',false);
    batchs=1:length(batch_idx_cell);
    batchs_=repmat(batchs,nSamples,1);
    batchs=reshape(batchs_,[],1);
    batch_subsample=cell2mat(cellfun(@(x) randperm(length(x),nSamples)+x(1)-1,batch_idx_cell,'uni',false));
    data_subsample=double(t.fc(batch_subsample,:));
    target_subsample=double(t.target(batch_subsample))';
    batch_sub=double(t.batch(batch_subsample))';
    temp=unique(batch_sub);
    bath_sub_idx=sum(cell2mat(arrayfun(@(x) x*(batch_sub==temp(x)),1:length(temp),'UniformOutput',false)),2);
    data=[data;data_subsample];
    subEpoch=[subEpoch;bath_sub_idx];
    targets=[targets;target_subsample];
    epoch=[epoch;i+0*target_subsample];
end 
productionTime = (1:length(epoch))'; 

%% %%%%%%%%%% ANALYSES %%%%%%%%%%%%%%
%% 
norms = vecnorm(data');
figure;plot(productionTime,norms,'r.')

%%
data1=data(1:(length(data)/2),:);
data2=data((length(data)/2)+1:length(data),:);
%% 
NNids = knnsearch(data, data, 'K', 300,'Distance','euclidean'); 
%NNids = NNids(:, 2:end); 
figure;
imagesc(NNids)

%%
NNids1 = knnsearch(data1, data1, 'K', 50,'Distance','euclidean'); 
NNids1 = NNids1(:, 2:end); 

NNids2 = knnsearch(data2, data2, 'K', 50,'Distance','euclidean'); 
NNids2 = NNids2(:, 2:end); 

%%
figure;imagesc(NNids)
figure;plot(NNids(1:20,:)')

%%
s=zeros(299,length(epoch));
for i=2:300
    s(i,:)=(abs(NNids(:,1)-NNids(:,i)));%.^2;
end

m=mean(s,1);
v=std(s,1);

%%
colors=magma(size(s,1)-225);
% norms = vecnorm(data');
figure;
ax=axes()
% figure('Color',[1,1,1],'position',[1000,509,1462,829]);
% ax=axes('Position',[.05,.5,.9,.45])
epoch_loc=arrayfun(@(x) find(epoch==x),[1:max(epoch)],'uniformoutput',false);
hold on 
arrayfun(@(i) scatter(productionTime(epoch_loc{i})',m(epoch_loc{i})',10,colors(i,:),'filled','o'),[1:max(epoch)])
axis tight
set(gca,'XTick',cellfun(@min,epoch_vec(1:2:end)),'XTickLabel',[1:2:size(accuracy,1)])
axis tight
ax.YAxis.LineWidth=2;
ax.YAxis.TickLength=[.005,0.007];
ax.XAxis.LineWidth=2;
ax.XAxis.TickLength=[.005,0.007];
ax.XLabel.String='Epoch';

ax.YAxis.LineWidth=2;
ax.YAxis.TickLength=[.005,0.007];
ax.FontSize=20
ax.FontWeight='bold'
ax.YLabel.String='Neighborhod time distance'
h=gcf;
set(h,'PaperOrientation','landscape');
set(h,'PaperPosition', [1 1 28 19]);
%print(gcf, '-depsc', strcat(pwd,'/disttime',train_sess,'.png'));

%%

m=mean(s,1);
v=std(s,1);

%figure;plot(m);hold on;plot(v,'Color','red')
figure;scatter(1:length(m),m,1);hold on;scatter(1:length(v),v,1,'red');hold on;...
    legend('Mean','Std');xticks(linspace(0,18000,7));xticklabels({0,10,20,30,40,50,60});xlabel('Epoch');ylabel('Neighborhood time distance')

%%
colors=magma(size(s,1)-225);
figure;
ax=axes()
epoch_loc=arrayfun(@(x) find(epoch==x),[1:max(epoch)],'uniformoutput',false);
hold on 
arrayfun(@(i) scatter(productionTime(epoch_loc{i})',m(epoch_loc{i})',4,colors(i,:),'filled'),[1:max(epoch)])
axis tight
xticks(linspace(0,18000,7));xticklabels({0,10,20,30,40,50,60});xlabel('Epoch');ylabel('Neighborhood time distance')
title('Mean')
%%

ax.YAxis.LineWidth=2;
ax.YAxis.TickLength=[.005,0.007];
ax.XAxis.LineWidth=2;
ax.XAxis.TickLength=[.005,0.007];
ax.XLabel.String='Epoch';

ax.YAxis.LineWidth=2;
ax.YAxis.TickLength=[.005,0.007];
ax.FontSize=20
ax.FontWeight='bold'
ax.YLabel.String='Neighborhod time distance'
h=gcf;
set(h,'PaperOrientation','landscape');
set(h,'PaperPosition', [1 1 28 19]);
%print(gcf, '-depsc', strcat(pwd,'/disttime',train_sess,'.png'));



%% 
squareform(pdist(data));
figure;
imagesc(squareform(pdist(data,'cosine')));
%% 
[RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids, epoch, subEpoch);
repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, 1:length(KNN_data));

repertoireDating.renditionPercentiles(NNids, epoch,  'doPlot', true);
repertoireDating.renditionPercentiles(NNids, epoch, 'valid', epoch == 50, 'doPlot', true,'percentiles',[5,50,95]);

%%
[RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids1, epoch(1:(length(data)/2)), subEpoch(1:(length(data)/2)));
repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, 1:length(KNN_data)/2);

repertoireDating.renditionPercentiles(NNids1, epoch(1:(length(data)/2)),  'doPlot', true);
repertoireDating.renditionPercentiles(NNids1, epoch(1:(length(data)/2)), 'valid', epoch == 15, 'doPlot', true,'percentiles',[5,50,95]);

%% 
[RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids2, epoch((length(data)/2)+1:length(data)), subEpoch((length(data)/2)+1:length(data)));
repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, (length(KNN_data)/2)+1:length(KNN_data));

repertoireDating.renditionPercentiles(NNids2, epoch((length(data)/2)+1:length(data)),  'doPlot', true);
repertoireDating.renditionPercentiles(NNids2, epoch((length(data)/2)+1:length(data)), 'valid', epoch == 15, 'doPlot', true,'percentiles',[5,50,95]);
%%
MM = repertoireDating.mixingMatrix(NNids, epoch, 'doPlot', true);
RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);
stratMM = repertoireDating.stratifiedMixingMatrices(data, epoch, subEpoch, RP);
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);

%%
MM = repertoireDating.mixingMatrix(NNids1, epoch(1:(length(data)/2)), 'doPlot', true);
RP = repertoireDating.renditionPercentiles(NNids1, epoch(1:(length(data)/2)), 'percentiles', 50);
stratMM = repertoireDating.stratifiedMixingMatrices(data1, epoch(1:(length(data)/2)), subEpoch(1:(length(data)/2)), RP);
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);

%%
MM = repertoireDating.mixingMatrix(NNids2, epoch((length(data)/2)+1:length(data)), 'doPlot', true);
RP = repertoireDating.renditionPercentiles(NNids2, epoch((length(data)/2)+1:length(data)), 'percentiles', 50);
stratMM = repertoireDating.stratifiedMixingMatrices(data2, epoch((length(data)/2)+1:length(data)), subEpoch((length(data)/2)+1:length(data)), RP);
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);

%% 
repertoireDating.visualizeStratifiedMixingMatrix(avgStratMM, stratMM);


