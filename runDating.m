function params_out=runDating(varargin)
p=inputParser();
addParameter(p, 'data_dir', '/om/group/evlab/Greta_Eghbal_manifolds/extracted');
addParameter(p, 'model_identifier', 'NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance-epoch=1-batchidx=600');
addParameter(p, 'layer', 'layer_3_Linear');
addParameter(p, 'k', 50);
addParameter(p, 'dist_metric', 'euclidean');
addParameter(p, 'num_subsamples', 60);



parse(p, varargin{:});
params = p.Results;
params_out = params;

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
data_dir = '/om/evlab';

% Load the generated mat files, session of interest: (input, the model identifier)
model_identifier = d
KNN_files = dir(strcat(data_dir, model_identifier, layer, '/*_extracted.mat'));

%% test
data=load('test.mat')
e=data.data.projection_results{1,1}.layer_0_Input;
% num_subs
%%
data_size = size(e)
num_classes = data_size(1)
num_features = data_size(2)
num_examples = data_size(3)


%%
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder,'/',KNN_files(x).name)}, 1:length(KNN_files));
[~,idx]=sort([KNN_files.datenum]);
KNN_data = KNN_data(idx);
KNN_data'

% Subsample and construct a data matrix 
num_subsamples = 120; % Per point in time, i.e. per batch idx
num_epochs = length(KNN_data);

data = [];
targets = [];
subEpoch = [];
epoch = [];

% Assert that subsampling across time is possible 
assert(num_subsamples <= num_classes * num_examples, 'Too many subsamples specified')
assert(num_subsamples >= num_classes, 'Number of subsamples has to be equal to or larger than number of classes')

%% 
% Make sure at least one sample per class per time point
num_remaining_sub = num_subsamples - num_classes
fix_sub = randsample(num_classes, num_classes) % Randomize the fixed samples 

% Don't sample the same data points 

sub = randsample(num_classes, num_subsamples, true)
if length(unique(sub)) ~= num_classes
    d=[]
end



%%
for i=1:length(KNN_data)
    t=load(KNN_data{i});
    unique_cell=mat2cell(unique(t.batch),1,ones(1,length(unique(t.batch))));
    batch_idx_cell=cellfun(@(x) find(t.batch==x),unique_cell,'uni',false);
    batchs=1:length(batch_idx_cell);
    batchs_=repmat(batchs,num_subsamples,1);
    batchs=reshape(batchs_,[],1);
    batch_subsample=cell2mat(cellfun(@(x) randperm(length(x),num_subsamples)+x(1)-1,batch_idx_cell,'uni',false));
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
NNids = knnsearch(data, data, 'K', k , 'Distance', dist_metric); 
%NNids = NNids(:, 2:end); 
figure;
imagesc(NNids)

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
MM = repertoireDating.mixingMatrix(NNids, epoch, 'doPlot', true);
RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);
stratMM = repertoireDating.stratifiedMixingMatrices(data, epoch, subEpoch, RP);
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);

%% 
repertoireDating.visualizeStratifiedMixingMatrix(avgStratMM, stratMM);


