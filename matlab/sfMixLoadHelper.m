%% pass in a spike time offset if analyzing files run with neuropixel probe/spikeGLX
spkOffset = 0.135; % from Manu, offset is 135 ms, roughly...

%% set up the glx structure, if grabbing sorted spikes
baseDir = pwd;
baseDir = strrep(baseDir, '/matlab', '/');
disp(baseDir);
expDir  = 'V1/recordings/';
expName = 'm676p3l11#16/';
progName = 'sfMixHalfInt';

msFolder = [baseDir expDir 'sorted/' expName];
filename = 'sfMixHalfInt.dat';
expoName = [baseDir expDir strrep(expName, '/', '[sfMixHalfInt].xml')];
firingsName = 'firings_355_384.curated.mda';
metricsName = 'cluster_metrics_355_384.json';

glx.msFolder = msFolder;
glx.filename = filename;
glx.expoName = expoName;
glx.firingsName = firingsName;
glx.metricsName = metricsName;

%%
sfMixLoad(strrep(expName, '#16/', ''), 'loadPath', [baseDir expDir], 'savePath', [baseDir strrep(expDir, 'recordings', 'structures')], 'spkOffset', spkOffset, 'glxStruct', glx, 'prog', progName);
