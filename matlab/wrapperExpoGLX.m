%%
baseDir = pwd;
baseDir = strrep(baseDir, '/matlab', '/');
disp(baseDir);
expDir  = 'V1/recordings/';
expName = 'm676p3l11#16/';

msFolder = [baseDir expDir 'sorted/' expName];
filename = 'sfMixHalfInt.dat';
expoName = [baseDir expDir strrep(expName, '/', '[sfMixHalfInt].xml')];
firingsName = 'firings_355_384.curated.mda';
metricsName = 'cluster_metrics_355_384.json';

%%
addpath(genpath([baseDir 'ExpoAnalysisTools/']));
addpath(genpath([baseDir expDir]));

%%
DIS = ReadExpoXML([loadPath, '/', fileName], 0, 0); % 0, 0 for (doSpikeWaveforms, beVerbose)
[clusid,spikes_by_clus,classification] = alignExpoGLX(DIS, msFolder, [msFolder filename], expoName, firingsName, metricsName);

