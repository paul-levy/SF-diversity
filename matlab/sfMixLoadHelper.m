%% pass in a spike time offset if analyzing files run with neuropixel probe/spikeGLX
spkOffset = 0.135; % from Manu, offset is 135 ms, roughly...

%% set up the glx structure, if grabbing sorted spikes
baseDir = pwd;
baseDir = strrep(baseDir, '/matlab', '/');
disp(baseDir);
expDir  = 'V1/recordings';
expName = 'm676l01#109/';
progName = 'sfMixLGN';

msFolder = [baseDir expDir '/sorted/' expName];
filename = 'sfMixHalfInt.dat';
expoName = [baseDir expDir strrep(expName, '/', '[sfMixHalfInt].xml')];
firingsName = 'firings_180_209.curated.mda';
metricsName = 'cluster_metrics_180_209.json';

glx.msFolder = msFolder;
glx.filename = filename;
glx.expoName = expoName;
glx.firingsName = firingsName;
glx.metricsName = metricsName;

%%
sfMixLoad(strrep(expName, '/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structuresTest')], 'prog', progName);
%sfMixLoad(strrep(expName, '#95/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structuresTest')], 'prog', progName);
% sfMixLoad(strrep(expName, '#13/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structures')], 'spkOffset', spkOffset, 'glxStruct', glx, 'prog', progName);
