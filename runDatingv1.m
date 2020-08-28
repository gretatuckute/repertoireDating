function params_out=runDatingv1(varargin)
p=inputParser();
addParameter(p, 'data_dir', '/om/group/evlab/Greta_Eghbal_manifolds/extracted/');
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
'DefaultAxesFontSize', 10, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultLineLineWidth', 1, ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 10, ...
'DefaultTextFontName', 'Helvetica', ...
'DefaultAxesBox', 'off', ...
'DefaultAxesTickLength', [0.01 0.015]);
set(groot, 'DefaultAxesTickDir', 'out');
set(groot, 'DefaultAxesTickDirMode', 'manual');

%% Directories
data_dir = 'C:\Users\greta\Documents\GitHub\neural_manifolds\local\';
% Load the generated mat files, session of interest: (input, the model identifier)
model_identifier = 'NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed'
%model_identifier = 'NN-partition_nclass=50_nobj=50000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed'
layer = 'layer_3_Linear'
KNN_files = dir(strcat(data_dir, model_identifier, '\*', layer, '_extracted.mat'))

%% 
order = cellfun(@(x) str2num(x(1:4)), {KNN_files.name}, 'UniformOutput', false);
assert(issorted(cell2mat(order)), 'Files not ordered correctly!')

%% Manual params for testing
num_subsamples = 96*2; % Per point in time, i.e. per batch idx
hier_level = 1;
k = 50;
dist_metric = 'euclidean'

%% Load files
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder,'/',KNN_files(x).name)}, 1:length(KNN_files));
file = load(KNN_data{1}).activation;
e = file.projection_results{1, hier_level}.( layer );

data_size = size(e);
num_classes = data_size(1);
num_features = data_size(2);
num_examples = data_size(3);

%% Assert that subsampling across time is possible 
assert(num_subsamples <= num_classes * num_examples, 'Too many subsamples specified')
assert(num_subsamples >= num_classes, 'Number of subsamples has to be equal to or larger than number of classes')

%% Iterate over all files

data = [];
targets = [];
epoch = [];
subEpoch = [];
testAcc = [];
trainAcc = [];

for i = 1:length(KNN_data)
    file = load(KNN_data{i}).activation;
    f = file.projection_results{1, hier_level}.( layer );
    
    % Subsample and construct a data matrix 
    f_perm = permute(f, [3 1 2]);
    f_res = reshape(f_perm, [num_classes*num_examples, num_features]);
    % I.e. concatenated according to: rows = num samples per class, e.g. if
    % num_examples = 20, then the first 20 rows correspond to all samples from
    % class 1
    
    % Cov matrix:
    % figure;imshow(cov(f_res'))
    
    % Generate targets
    target = repelem([1:num_classes], num_examples)';

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

    final_idx = horzcat(idx_cat, more_idx)';

    assert(length(final_idx) == num_subsamples, 'Subsampling index does not match')

    % Subsample
    subsampled_data = f_res(final_idx,:); % Checked that it correspond to the first idx in the sub list
    subsampled_target = target(final_idx);
    
    % get batch/epoch 
    batchidx_cell = file.batchidx;
    batchs = repmat(batchidx_cell, num_subsamples, 1);
    
    epoch_cell = file.epoch
    epochs = repmat(epoch_cell, num_subsamples, 1);
    
    % Get accuracies
    test_a = file.test_acc;
    train_a = file.train_acc;

    
    % Append
    data = [data; subsampled_data];
    epoch = double([epoch; epochs]);
    subEpoch = double([subEpoch; batchs]);
    targets = [targets; subsampled_target];
    testAcc = [testAcc; test_a];
    trainAcc = [trainAcc; train_a];

end

% Find log interval:
log_int = abs(subEpoch(num_subsamples) - subEpoch(num_subsamples*2));


productionTime = (1:length(epoch))'; 

%% %%%%%%%%%% ANALYSES %%%%%%%%%%%%%%

%% Test acc
figure;plot([1:length(testAcc)], testAcc)
xlabel('Time')
ylabel('Test accuracy')

% Test acc for each subsample - if overlay on the other plots
testAccSubsample = repmat(testAcc, 1, num_subsamples);
testAccSubsamples = reshape(testAccSubsample', [], 1)

figure;plot([1:length(testAccSubsamples)], testAccSubsamples)
xlabel('Time')
ylabel('Test accuracy')

%% Make axes labels
g = diff(subEpoch);
g = [0;g];

subepoch_loc = arrayfun(@(x) find(subEpoch==x), unique(subEpoch), 'uniformoutput',false); % finds the unique subEpochs
epoch_loc = arrayfun(@(x) find(epoch==x), [1:max(epoch)],'uniformoutput',false); % Locates the epochs
epochSwitch = (g==(min(g)));

p = g.*double(epochSwitch) - max(g).*double(epochSwitch); % account for that epochSwitch only contains 1, and not log interval
pp = [cumsum(-p)]; 
ppp = subEpoch + pp;

su = arrayfun(@(x) find(ppp==x), unique(ppp), 'uniformoutput',false); % finds unique number of subEpoch batches
incrSub = cell2mat(cellfun(@(x) min(x), su, 'uniformoutput',false));
h = subEpoch((incrSub));

%%
figure;plot([1:length(testAccSubsamples)], testAccSubsamples)
hold on
set(gca,'XTick',downsample(incrSub,15))
set(gca,'XTickLabel',downsample(h,15))
xlabel('Subepoch')
ylabel('Test accuracy')

%%
colors = magma(max(epoch)); % 3 colors
figure;
ax=axes()
hold on
arrayfun(@(i) plot(productionTime(epoch_loc{i})', testAccSubsamples(epoch_loc{i})', 'Linewidth', 3, 'Color', colors(i,:)), [1:max(epoch)])
hold on
set(gca,'XTick',downsample(incrSub,20))
set(gca,'XTickLabel',downsample(h,20))
xlabel('Subepoch')
ylabel('Test accuracy')
axis tight
saveas(gcf, strcat(pwd,filesep,'figures',filesep,'test_acc_',model_identifier,'_',layer,'_',num2str(hier_level),'.pdf'));

%% epoch legend 
figure;plot([1:length(testAccSubsamples)], testAccSubsamples)
hold on
set(gca,'XTick',downsample(productionTime,45504/3))
%set(gca,'XTick',epochSwitch)
set(gca,'XTickLabel',downsample(epoch,45504/3))

%%
figure;plot([1:length(testAccSubsamples)], testAccSubsamples)
a1Pos = get(gca,'Position');
%// Place axis 2 below the 1st.
ax2 = axes('Position',[a1Pos(1) a1Pos(2)-.05 a1Pos(3) a1Pos(4)],'Color','none','YTick',[],'YTickLabel',[]);
%// Adjust limits
xlim([min(x2(:)) max(x2(:))])
hold on
set(gca,'XTick',downsample(productionTime,1000))
%set(gca,'XTick',epochSwitch)
set(gca,'XTickLabel',epochSwitch)



%% 
norm = vecnorm(data');

figure;
ax=axes()
hold on
arrayfun(@(i) scatter(productionTime(epoch_loc{i})', norm(epoch_loc{i})', 10, colors(i,:), 'filled', 'o'), [1:max(epoch)])
hold on
set(gca,'XTick',downsample(incrSub,20))
set(gca,'XTickLabel',downsample(h,20))
xlabel('Subepoch')
ylabel('Test accuracy')
axis tight
saveas(gcf, strcat(pwd,filesep,'figures',filesep,'norm_',model_identifier,'_',layer,'_',num2str(hier_level),'.pdf'));

% figure;plot(productionTime,norm,'r.')
% hold on
% xlabel('Subepoch')
% hold on
% set(gca,'YTick',downsample(incrSub,20))
% set(gca,'YTickLabel',downsample(h,20))
% ylabel('Norm')


%% 
%k=300
NNids_self = knnsearch(data, data, 'K', k , 'Distance', dist_metric); 
NNids = NNids_self(:, 2:end); 
figure;
imagesc(NNids)
% ylabel('Time')
xlabel('K nearest neighbors')
hold on
set(gca,'YTick',downsample(incrSub,20))
set(gca,'YTickLabel',downsample(h,20))
ylabel('Subepoch')
axis tight
saveas(gcf, strcat(pwd,filesep,'figures',filesep,'KNN_',model_identifier,'_',layer,'_',num2str(hier_level),'.pdf'));

%%
figure;imagesc(NNids)
figure;plot(NNids(1:20,:)')

%% Compute norm
norms = zeros(k - 1, length(epoch));

for i=2:k
    norms(i-1, :)=(abs(NNids_self(:,1) - NNids_self(:,i)));%.^2;
end

mean_norm = mean(norms,1);
std_norm = std(norms,1);

%%
% Simple figure test
figure;scatter([1:size(mean_norm,2)], mean_norm)

%% Mean vector norms over samples at the same time 
y = reshape(mean_norm, num_subsamples, length(KNN_files));
mean_time_norm = mean(y, 1);


%% Plot vector norm - according to epochs colors
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
colors = magma(max(epoch)); % 3 colors
figure;
ax=axes()
epoch_loc = arrayfun(@(x) find(epoch==x), [1:max(epoch)],'uniformoutput',false); % Locates the epochs
hold on 
arrayfun(@(i) plot(productionTime(epoch_loc{i})', testAccSubsamples(epoch_loc{i})', 'Linewidth', 3, 'Color', colors(i,:)), [1:max(epoch)])
hold on
%arrayfun(@(i) scatter(productionTime(epoch_loc{i})', mean_norm(epoch_loc{i})', 10, colors(i,:), 'filled', 'o'), [1:max(epoch)])
ax.YLabel.String='Neighborhod time distance'
ax.XLabel.String='Subepoch'
hold on
set(gca,'XTick',downsample(incrSub,20))
set(gca,'XTickLabel',downsample(h,20))
axis tight
saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meannorm_KNN_',model_identifier,'_',layer,'_',num2str(hier_level),'.pdf'));


% %% Plot vector norm - according to subepochs colors
% % Create unique subEpoch vector that adds the values
% 
% %subEpoch_incr = subEpoch + [1:log_int:]log_int;
% 
% % Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
% colors = magma(length(unique(subEpoch))*3); 
% figure;
% ax=axes()
% subEpoch_loc = arrayfun(@(x) find(subEpoch==x), [(unique(subEpoch))],'uniformoutput',false); % Locates the subepochs
% hold on 
% arrayfun(@(i) scatter(productionTime(subEpoch_loc{i})', mean_norm(subEpoch_loc{i})', 10, colors(i,:), 'filled', 'o'), [(unique(subEpoch))])
% ax.YLabel.String='Neighborhod time distance'
% ax.XLabel.String='Time'
% 
% %% Plot vector norm - according to data point colors
% % Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
% colors = magma(length(mean_norm)); 
% figure;
% ax=axes()
% hold on 
% arrayfun(@(i) scatter(productionTime(i)', mean_norm(i)', 10, colors(i,:), 'filled', 'o'), [1:length(mean_norm)])
% 
% %% Plot vector norm - according to data point colors - meaned over time
% % Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
% time = [1:length(KNN_files)]
% colors = magma(length(mean_time_norm)); 
% figure;
% ax=axes()
% hold on 
% arrayfun(@(i) scatter(time(i)', mean_time_norm(i)', 30, colors(i,:), 'filled', 'o'), [1:length(mean_time_norm)])
% ax.YLabel.String='Neighborhod time distance'
% ax.XLabel.String='Time'
% %ax.Title.String='
% %% Plot vector norm.
% % Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
% 
% colors = magma(size(mean_norm,2));%-255);%-225);
% %colors = magma(max(epoch)); % 3 colors
% 
% figure;
% ax=axes()
% % figure('Color',[1,1,1],'position',[1000,509,1462,829]);
% % ax=axes('Position',[.05,.5,.9,.45])
% epoch_loc = arrayfun(@(x) find(epoch==x), [1:max(epoch)],'uniformoutput',false); % Locates the epochs
% hold on 
% %arrayfun(@(i) scatter(productionTime(epoch_loc{i})', mean_norm(epoch_loc{i})', 10, colors(i,:), 'filled', 'o'), [1:max(epoch)])
% arrayfun(@(i) scatter(productionTime(i)', mean_norm(i)', 10, colors(i,:), 'filled', 'o'), [1:max(mean_norm)])
% %%
% axis tight
% %set(gca,'XTick', subEpochs) %'XTickLabel', [1:size(mean_norm,2)])%[1:size(norms,1)])
% 
% 
% %set(gca,'XTick', cellfun(@min, epoch_loc),'XTickLabel', [1:size(mean_norm,2)])%[1:size(norms,1)])
% axis tight
% % ax.YAxis.LineWidth=2;
% % ax.YAxis.TickLength=[.005,0.007];
% % ax.XAxis.LineWidth=2;
% % ax.XAxis.TickLength=[.005,0.007];
% ax.XLabel.String='Epoch';
% 
% % ax.YAxis.LineWidth=2;
% % ax.YAxis.TickLength=[.005,0.007];
% % ax.FontSize=20
% % ax.FontWeight='bold'
% ax.YLabel.String='Neighborhod time distance'
% h=gcf;
% set(h,'PaperOrientation','landscape');
% set(h,'PaperPosition', [1 1 28 19]);
% %print(gcf, '-depsc', strcat(pwd,'/disttime',train_sess,'.png'));
% 
% %%
% 
% mean_norm=mean(norms,1);
% v=std(norms,1);
% 
% %figure;plot(m);hold on;plot(v,'Color','red')
% figure;scatter(1:length(mean_norm),mean_norm,1);hold on;scatter(1:length(v),v,1,'red');hold on;...
%     legend('Mean','Std');xticks(linspace(0,18000,7));xticklabels({0,10,20,30,40,50,60});xlabel('Epoch');ylabel('Neighborhood time distance')
% 
% %%
% colors=magma(size(norms,1)-225);
% figure;
% ax=axes()
% epoch_loc=arrayfun(@(x) find(epoch==x),[1:max(epoch)],'uniformoutput',false);
% hold on 
% arrayfun(@(i) scatter(productionTime(epoch_loc{i})',mean_norm(epoch_loc{i})',4,colors(i,:),'filled'),[1:max(epoch)])
% axis tight
% xticks(linspace(0,18000,7));xticklabels({0,10,20,30,40,50,60});xlabel('Epoch');ylabel('Neighborhood time distance')
% title('Mean')
% %%
% 
% ax.YAxis.LineWidth=2;
% ax.YAxis.TickLength=[.005,0.007];
% ax.XAxis.LineWidth=2;
% ax.XAxis.TickLength=[.005,0.007];
% ax.XLabel.String='Epoch';
% 
% ax.YAxis.LineWidth=2;
% ax.YAxis.TickLength=[.005,0.007];
% ax.FontSize=20
% ax.FontWeight='bold'
% ax.YLabel.String='Neighborhod time distance'
% h=gcf;
% set(h,'PaperOrientation','landscape');
% set(h,'PaperPosition', [1 1 28 19]);
% %print(gcf, '-depsc', strcat(pwd,'/disttime',train_sess,'.png'));
% 
% 
% 
% %% 
% squareform(pdist(data));
% figure;
% imagesc(squareform(pdist(data,'cosine')));
% %% 
% [RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids, epoch, subEpoch);
% repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, 1:length(KNN_data));
% 
% repertoireDating.renditionPercentiles(NNids, epoch,  'doPlot', true);
% repertoireDating.renditionPercentiles(NNids, epoch, 'valid', epoch == 50, 'doPlot', true,'percentiles',[5,50,95]);
% 
% 
% %%
% MM = repertoireDating.mixingMatrix(NNids, epoch, 'doPlot', true);
% RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);
% stratMM = repertoireDating.stratifiedMixingMatrices(data, epoch, subEpoch, RP);
% C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
% avgStratMM = nanmean(cat(3, C{:}), 3);
% 
% %% 
% repertoireDating.visualizeStratifiedMixingMatrix(avgStratMM, stratMM);
% 

