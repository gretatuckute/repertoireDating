function params_out = runKNN(varargin)
p=inputParser();
addParameter(p, 'data_dir', '/om/group/evlab/Greta_Eghbal_manifolds/extracted/');
addParameter(p, 'model_identifier', 'NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance-epoch=1-batchidx=600');
addParameter(p, 'layer', 'layer_3_Linear');
addParameter(p, 'hier_level', 1);
addParameter(p, 'k', 50);
addParameter(p, 'dist_metric', 'euclidean');
addParameter(p, 'num_subsamples', 60);
addParameter(p, 'save_fig', true);

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
set(gcf,'color','w');

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

%% Load files
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder,'/',KNN_files(x).name)}, 1:length(KNN_files));
file = load(KNN_data{1}).activation;
e = file.projection_results{1, hier_level}.( layer );

data_size = size(e);
num_classes = data_size(1);
num_features = data_size(2);
num_examples = data_size(3);

%% Manual params for testing
num_subsamples = num_classes*2; % Per point in time, i.e. per batch idx
hier_level = 1;
k = num_classes*2;
dist_metric = 'euclidean';


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
    
    epoch_cell = file.epoch;
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
logInt = abs(subEpoch(num_subsamples) - subEpoch(num_subsamples*2));
productionTime = (1:length(epoch))'; 
time = [1:length(KNN_files)];
relTime = productionTime./max(productionTime);

%% %%%%%%%%%% ANALYSES %%%%%%%%%%%%%%

saveStr = strcat(model_identifier,'_',layer,'_hier_',num2str(hier_level),'_numSubsamples_',num2str(num_subsamples),'_k_',num2str(k),'.pdf');
colorsEpoch = magma(max(epoch))% +1); % 3 colors
% colorsEpoch = colorsEpoch(2:end, :);
colorsTarget = magma(max(num_classes));
colorsTargetViridis = viridis(max(num_classes));
targetLoc = arrayfun(@(x) find(targets==x), [1:max(target)],'uniformoutput',false); % Locates the targets

%% Make axes labels
subEpochSwitch = diff(subEpoch);
subEpochSwitch = [0;subEpochSwitch];

subepochLoc = arrayfun(@(x) find(subEpoch==x), unique(subEpoch), 'uniformoutput',false); % finds the unique subEpochs
epochLoc = arrayfun(@(x) find(epoch==x), [1:max(epoch)],'uniformoutput',false); % Locates the epochs
epochSwitch = (subEpochSwitch==(min(subEpochSwitch)));

p = subEpochSwitch.*double(epochSwitch) - max(subEpochSwitch).*double(epochSwitch); % account for that epochSwitch only contains 1, and not log interval
pp = [cumsum(-p)]; 
incrTrial = subEpoch + pp;

uniqueSubEpochBatches = arrayfun(@(x) find(incrTrial==x), unique(incrTrial), 'uniformoutput',false); % finds unique number of subEpoch batches
incrSubEpochTrial = cell2mat(cellfun(@(x) min(x), uniqueSubEpochBatches, 'uniformoutput',false));
incrSubEpoch = subEpoch((incrSubEpochTrial));

%% Test/train accuracy

% Test acc for each subsample - if overlay on the other plots
testAccSubsample = repmat(testAcc, 1, num_subsamples);
testAccSubsamples = reshape(testAccSubsample', [], 1);

% Train acc for each subsample - if overlay on the other plots
trainAccSubsample = repmat(trainAcc, 1, num_subsamples);
trainAccSubsamples = reshape(trainAccSubsample', [], 1);

if save_fig
    % Plot test accuracy
    figure;
    ax=axes()
    hold on
    arrayfun(@(i) plot(productionTime(epochLoc{i})', testAccSubsamples(epochLoc{i})', 'Linewidth', 1, 'Color', colorsEpoch(i,:)), [1:max(epoch)]);
    hold on
    ylabel('Test accuracy')
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'test_acc_',saveStr));
    
    % If label using subepoch indexing:
%     set(gca,'XTick',downsample(incrSubEpochTrial,15))
%     set(gca,'XTickLabel',downsample(incrSubEpoch,15))
%     xlabel('Subepoch')

    % If label using epoch indexing:
%     set(gca,'XTick',downsample(productionTime, round(size(productionTime,1)/max(epoch))))
%     set(gca,'XTickLabel',downsample(epoch, round(size(productionTime,1)/max(epoch))))
end 

%% Vector norm
dataNorm = vecnorm(data');

% Epoch coloring
if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(epochLoc{i})', dataNorm(epochLoc{i})', 2, colorsEpoch(i,:), 'filled', 'o'), [1:max(epoch)])
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    ylabel('Norm')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'norm_epochColor_',saveStr));
end 

% Target coloring
if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', dataNorm(targetLoc{i})', 8, colorsTarget(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    ylabel('Norm')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'norm_targetColor_',saveStr));
end 

%% Vector norm - meaned across samples
y = reshape(dataNorm, num_subsamples, length(KNN_files));
meanTimeDataNorm = mean(y, 1);

% figure;scatter([1:237],meanTimeDataNorm)

%% Nearest neighbors
NNids_self = knnsearch(data, data, 'K', k, 'Distance', dist_metric); 
NNids_self = NNids_self./max(NNids_self(:,1)); % normalized NNids
NNids = NNids_self(:, 2:end); 

if save_fig
    figure;
    imagesc(NNids)
    hold on
    xlabel('K nearest neighbors')
    set(gca,'YTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'YTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    ylabel('Relative time in training')
    colorbar()
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'NearestNeighbors_',saveStr));
end  

y1 = reshape(NNids, num_subsamples, length(KNN_files), k-1); % num subsamples x num points in time x num k-1
meanTimeNNids = squeeze(mean(y1, 1));
% figure;imagesc(meanTimeNNids)

%% Compute norm of neighbors
normNN = zeros(k - 1, length(epoch));

for i=2:k
    normNN(i-1, :)=(abs(NNids_self(:,1) - NNids_self(:,i)));%.^2;
end

meanNormNN = mean(normNN,1);
stdNormNN = std(normNN,1);

%% Mean vector norms over samples at the same time 
y2 = reshape(meanNormNN, num_subsamples, length(KNN_files));
meanTimeNormNN = mean(y2, 1);

y3 = reshape(meanStdNN, num_subsamples, length(KNN_files));
meanTimeStdNormNN = mean(y3, 1);

%% Plot vector norm - according to epochs colors
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.

if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(epochLoc{i})', meanNormNN(epochLoc{i})', 1, colorsEpoch(i,:), 'filled', 'o'), [1:max(epoch)])
    ylabel('Neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanNormNN_',saveStr));
end


%% Plot vector norm - according to data point colors
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
colorsNormNN = magma(length(meanNormNN)); 

if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(i)', meanNormNN(i)', 10, colorsNormNN(i,:), 'filled', 'o'), [1:length(meanNormNN)])
    ylabel('Neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanNormNN_indColors_',saveStr));
end

%% Plot vector norm - according to data point colors - meaned over time
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
colorsMeanTimeNormNN = magma(length(meanTimeNormNN)); 

if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(time(i)', meanTimeNormNN(i)', 30, colorsMeanTimeNormNN(i,:), 'filled', 'o'), [1:length(meanTimeNormNN)])
    ylabel('Mean neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanTimeNormNN_',saveStr));
end


%% KNN Vector norm and std in same plot

if save_fig
    figure;
    scatter(productionTime', meanNormNN', 1, colorsEpoch(1,:), 'filled', 'o')
    hold on
    scatter(productionTime', stdNormNN', 1, colorsEpoch(2,:), 'filled', 'o')
    hold on
    legend('Mean neighbor distance','Std neighbor distance')
    ylabel('Neighbor distance') 
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanStdTimeNormNN_',saveStr));
end

%% Target coloring

if save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', meanNormNN(targetLoc{i})', 1, colorsTarget(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', stdNormNN(targetLoc{i})', 1, colorsTargetViridis(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    %legend('Mean neighbor distance','Std neighbor distance')
    ylabel('Neighbor distance') 
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanStdTimeNormNN_targetColor_',saveStr));
end

%% Save variables

%% First across all subsamples, then averaged - pairwise
keySet = {'params', 'targets', 'testAccSubsamples', 'testAcc', 'trainAccSubsamples', 'trainAcc', ...
    'dataNorm', 'meanTimeDataNorm', 'NNids', 'meanTimeNNids', 'meanNormNN', 'meanTimeNormNN', ... 
    'meanStdNN', 'meanTimeStdNormNN'};

valueSet = {params_out, targets, testAccSubsamples, testAcc, trainAccSubsamples, trainAcc, ...
    dataNorm, meanTimeDataNorm, NNids, meanTimeNNids, meanNormNN, meanTimeNormNN, ...
     meanStdNN, meanTimeStdNormNN};
M = containers.Map(keySet,valueSet,'UniformValues',false)

end

%% Repetoire Dating
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


