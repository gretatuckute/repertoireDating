function params_out=runDating_with_checks(varargin)
p=inputParser();
addParameter(p, 'data_dir', '/om/group/evlab/Greta_Eghbal_manifolds/extracted');
addParameter(p, 'model_identifier', 'NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance-epoch=1-batchidx=600');
addParameter(p, 'layer', 'layer_3_Linear');
addParameter(p, 'hier_level', 1);

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
data_dir = 'C:\Users\greta\Documents\GitHub\neural_manifolds\local\';
% Load the generated mat files, session of interest: (input, the model identifier)
model_identifier = 'NN-partition_nclass=50_nobj=50000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed'
layer = 'layer_1_Linear'
KNN_files = dir(strcat(data_dir, model_identifier, '\*', layer, '_extracted.mat'))

%% 
order = cellfun(@(x) str2num(x(1:4)), {KNN_files.name}, 'UniformOutput', false)
assert(issorted(cell2mat(order)), 'Files not ordered correctly!')

%% test
num_subsamples = 120; % Per point in time, i.e. per batch idx
hier_level = 1;
k = 50;
dist_metric = 'euclidean'
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder,'/',KNN_files(x).name)}, 1:length(KNN_files));

file = load(KNN_data{1}).activation;
e = file.projection_results{1, hier_level}.( layer );

data_size = size(e)
num_classes = data_size(1)
num_features = data_size(2)
num_examples = data_size(3)

%% Assert that subsampling across time is possible 
assert(num_subsamples <= num_classes * num_examples, 'Too many subsamples specified')
assert(num_subsamples >= num_classes, 'Number of subsamples has to be equal to or larger than number of classes')

%% Iterate over all files

data = [];
targets = [];
epoch = [];
subEpoch = [];

for i = 1:51%length(KNN_data)
    file = load(KNN_data{i}).activation;
    f = file.projection_results{1,hier_level}.( layer );
    
    
    % Subsample and construct a data matrix 
    f_perm = permute(f, [3 1 2]);
    % check
    % c1=f(2,:,3) % 2nd class, 3rd example
    % c2=squeeze(f_perm(3,2,:))' % 3rd example, 2nd class, 
    % isequal(c1,c2)
    
    % figure;imshow(cov(f_res'))

    f_res = reshape(f_perm, [num_classes*num_examples, num_features]);
    % I.e. concatenated according to: rows = num samples per class, e.g. if
    % num_examples = 20, then the first 20 rows correspond to all samples from
    % class 1
    
    % Generate targets
    targets = repelem([1:num_classes], num_examples)';
    
%     c1=squeeze(f_perm(1,2,:)); % first sample, 2nd class
%     c2=squeeze(f_perm(num_examples,2,:)); % last sample, 2nd class 

    % test that it matches with the reshaped version
%     r1=squeeze(f_res(num_examples+1,:))'; % first sample, 2nd class
%     isequal(c1,r1)
% 
%     r2=squeeze(f_res(num_examples*2,:))'; % last sample, 2nd class
%     isequal(c2,r2)


    % Make sure at least one sample per class per time point
    sub = randsample(num_examples, num_classes, true);
    % Add indices according to where to sample from, taking num_examples
    % into account:
    add = linspace(0, num_examples*num_classes, num_classes+1);
    add_array = add(1:num_classes);

    idx_cat = sub' + add_array; % Indices for subsampling per category

    % Add more indices
    if num_subsamples ~= num_classes
        num_remaining_sub = num_subsamples - num_classes;
        draw = [1:num_examples*num_classes]; % possible indices to draw from
        draw_array = setdiff(draw, idx_cat); % subtract the ones already used for the category requirement

        % Now draw from this list, to avoid sampling the same data points 
        more_idx = randsample(draw_array, num_remaining_sub, false); % without replacement 

    end

    final_idx = horzcat(idx_cat, more_idx)'

    assert(length(final_idx) == num_subsamples, 'Subsampling index does not match')

    % Subsample
    subsampled_data = f_res(final_idx,:); % Checked that it correspond to the first idx in the sub list
    subsampled_target = targets(final_idx);
    
    % get batch/epoch 
    batchidx_cell = file.batchidx
    batchs = repmat(batchidx_cell, num_subsamples, 1);
    
    epoch_cell = file.epoch
    epochs = repmat(epoch_cell, num_subsamples, 1);
    
    
    % Append
    data = [data; subsampled_data];
    epoch = [epoch; epochs];
    subEpoch = [subEpoch; batchs];
    targets = [targets; subsampled_target];
    

end

%% %%%%%%%%%% ANALYSES %%%%%%%%%%%%%%
%% 
norms = vecnorm(data');
productionTime = (1:length(epoch))'; 
figure;plot(productionTime,norms,'r.')

%% 
NNids = knnsearch(data, data, 'K', k , 'Distance', dist_metric); 
NNids = NNids(:, 2:end); 
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


