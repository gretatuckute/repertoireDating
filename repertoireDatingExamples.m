    
    % This script creates random example data and computes/plots 
    % repertoire dating statistics and mixing matrices.
    %
    % 
    % ---
    % Copyright (C) 2020 University Zurich, Sepp Kollmorgen
    % 
    % Reference (please cite):
    % Kollmorgen, S., Hahnloser, R.H.R. & Mante, V. Nearest neighbours reveal
    % fast and slow components of motor learning. Nature 577, 526-530 (2020).
    % https://doi.org/10.1038/s41586-019-1892-x
    % 
    % This program is free software: you can redistribute it and/or modify
    % it under the terms of the GNU Affero General Public License as published by
    % the Free Software Foundation, either version 3 of the License, or
    % (at your option) any later version.
    %
    % This program is distributed in the hope that it will be useful,
    % but WITHOUT ANY WARRANTY; without even the implied warranty of
    % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    % GNU Affero General Public License for more details.
    %
    % You should have received a copy of the GNU Affero General Public License
    % along with this program (see LICENSE file).  If not, see <https://www.gnu.org/licenses/>.
    %
    % Repertoire-Dating on Github: <a href="matlab:web('https://github.com/skollmor/repertoireDating', '-browser')">https://github.com/skollmor/repertoireDating</a>
    % Dataspace on Github: <a href="matlab:web('https://github.com/skollmor/dspace', '-browser')">https://github.com/skollmor/dspace</a>


%% Try to load real KNN data
data_super = load('/Users/gt/Documents/GitHub/neural_manifolds/local/result/feb26/NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed_layer_2_Linear_numSubsamples_100_k_100.mat')

%%
hier_num = 6;
data = data_super.super_struct.hier_6;
NNids_data = data.NNids;

% fix normalization 
NNids_t = size(NNids,1)*NNids;
NNids = ceil(NNids_t);

figure;imagesc(NNids)

%% Get params
nSamples = 100;                                        % samples per subEpoch
nEpochs = 10;                                          % number of epochs (e.g. days)
nSubEpochs = length(NNids)/nSamples/10;                                        % number of sub epochs  (e.g. periods in day)
N = nEpochs * nSubEpochs * nSamples;                   % total number of datapoints
productionTime = (1:N)';                               % production time for each datapoint
epoch = ceil(productionTime/(nSubEpochs*nSamples));
subEpoch = floor(mod(productionTime-1, (nSubEpochs*nSamples))/nSamples) + 1;
permute = False

%% Create random data
% nEpochs = 40;                                          % number of epochs (e.g. days)
% nSubEpochs = 5;                                        % number of sub epochs  (e.g. periods in day)
% nSamples = 250;                                        % samples per subEpoch
% dim = 25;                                              % data dimension
% N = nEpochs * nSubEpochs * nSamples;                   % total number of datapoints
% productionTime = (1:N)';                               % production time for each datapoint
% epoch = ceil(productionTime/(nSubEpochs*nSamples));
% subEpoch = floor(mod(productionTime-1, (nSubEpochs*nSamples))/nSamples) + 1;
% data = randn(N, dim) + (1:N)'/N * 2;
% permute = false

%% Compute k-NN graph

if permute
    data1 = data;
    rand_idx = randperm(length(data1));
    data = data1(rand_idx, :);
end 

NNids_self = knnsearch(data, data, 'K', 100);               % or use your preferred nearest neighbor searcher 
NNids = NNids_self(:, 2:end);                               % points should not be their own neighbors

%% GT norm computation (prob not relevant)
% normNN = zeros(100 - 1, length(epoch));
% 
% for i=2:100
%     normNN(i-1, :)=(abs(NNids_self(:,1) - NNids_self(:,i)));%.^2;
% end
% 
% meanNormNN = mean(normNN,1);
% stdNormNN = std(normNN,1)

%% Compute and plot repertoire dating pecentiles 
[RPD, RPD_epoch, RPD_subEpoch] = repertoireDating.percentiles(NNids, epoch, subEpoch);

%%
repertoireDating.plotPercentiles(RPD, RPD_epoch, RPD_subEpoch, 3:5);

%%
RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);

%% Compute and plot rendition percentiles for all datapoints from epoch 20
repertoireDating.renditionPercentiles(NNids, epoch, 'valid', epoch == 3, 'doPlot', true);
    
%% Compute mixing matrix and plot it
MM = repertoireDating.mixingMatrix(NNids, epoch, 'doPlot', true);

%% Plot percentiles
NNids=ceil(data_super.super_struct.hier_1.NNids*N);
figure;
plot(prctile(NNids,[5,50,95],2))

figure;
plot(prctile(NNids,[5,50,95],2))


figure;plot(movmean(prctile(NNids,[5,50,95],2),10));

%% Across different hierarchies

NNids = ceil(data_super.super_struct.hier_1.NNids*N); 
figure;plot(movmean(prctile(NNids,[5,50,95],2),10));title('Hierarchy 1')

NNids = ceil(data_super.super_struct.hier_2.NNids*N); 
figure;plot(movmean(prctile(NNids,[5,50,95],2),10));title('Hierarchy 2')

NNids = ceil(data_super.super_struct.hier_3.NNids*N); 
figure;plot(movmean(prctile(NNids,[5,50,95],2),10));title('Hierarchy 3')

NNids = ceil(data_super.super_struct.hier_4.NNids*N); 
figure;plot(movmean(prctile(NNids,[5,50,95],2),10));title('Hierarchy 4')

NNids = ceil(data_super.super_struct.hier_6.NNids*N); 
figure;plot(movmean(prctile(NNids,[5,50,95],2),10));title('Hierarchy 6')

%% Compute stratified mixing matrix and plot it
RP = repertoireDating.renditionPercentiles(NNids, epoch, 'percentiles', 50);
% RP is the repertoire time (50th percentile)

%% MDS
MM = repertoireDating.mixingMatrix(NNids, subEpoch, 'doPlot', true, 'doMDS', true);



% fix this
stratMM = repertoireDating.stratifiedMixingMatrices(data, epoch, subEpoch, RP, 'uEpochs', 1:5);
% Avg the stratified mixing matrices
C = arrayfun(@(i) stratMM.allMMs{i}.log2CountRatio, 1:numel(stratMM.allMMs), 'uni', false);
avgStratMM = nanmean(cat(3, C{:}), 3);
% Visualize through MDS
repertoireDating.visualizeStratifiedMixingMatrix(avgStratMM, stratMM);

