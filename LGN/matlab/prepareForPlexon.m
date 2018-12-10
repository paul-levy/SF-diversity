% this script will grab all .xml files in a given directory and convert all
% for use in the Plexon Offline Spike Sorter

% add the folders/paths we need
addpath(genpath('/u/vnl/matlab/ExpoMatlab/'));


%% get the files
basePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/LGN/recordings/';
xmlFiles = dir([basePath, '*xml']);

for i = 1 : length(xmlFiles)
   
    XMLtoPlexon([basePath, xmlFiles(i).name]);
end