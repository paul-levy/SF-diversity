% Modified from Robbe Goris - 17.12.7

% Begin with clean slate
clear all
close all
clc

% overall
nDispMix = 4; % 4 dispersions in sfMix block
nConsMix = 4; % 4 total cons in sfMix block
nGrats = 7; % [1, 3, 5, 7] gratings for dispersions [1, 2, 3, 4]
centGrat = mean(1:nGrats);

% frequency series
nSfsTot = 11;
sfMin = 0.3;
sfMax = 10;
freqSeries = logspace(log10(sfMin), log10(sfMax), nSfsTot);

% contrast series
nCons = 9;
conMin = 0.05;
conMax = 1;
logCons = logspace(log10(conMin), log10(conMax), nCons);

%% calculate contrast values for each dispersion X total contrast cond.
conProfile = zeros(nGrats, nDispMix, nConsMix);

% dispersions 1 & 2 appear for all total contrasts
% dispersion 3 appears for all but lowest total contrast (first 3)
% dispersion 4 appears for all but last two total contrasts (first 2)
d1start = length(logCons);
d2start = 8;
d3start = 7;
d4start = 6;
for c = 1 : nConsMix
    % disp = 1
    conProfile(centGrat, 1, c) = logCons(d1start-(c-1));
    
    % disp = 2; the flanking gratings are 4 steps lower than the center on
    % our logCons steps
    % i.e. [n-4 n n-4] or...
    % i.e. [4 8 4] --> [3 7 3] --> etc
    conProfile(centGrat, 2, c) = logCons(d2start-(c-1));
    conProfile(centGrat + [-1 1], 2, c) = logCons(d2start-4-(c-1)); 

    % disp 3;
    % base profile is [3, 4, 7, 4, 3] --> sub 1 --> sub 1
    if c < 4
        conProfile(centGrat, 3, c) = logCons(d3start-(c-1));
        conProfile(centGrat + [-1 1], 3, c) = logCons(d3start-3-(c-1));
        conProfile(centGrat + [-2 2], 3, c) = logCons(d3start-4-(c-1));
    end
    
    % disp 4;
    % base profile is [2, 3, 4, 6, 4, 3, 2] --> sub 1
    if c < 3
        conProfile(centGrat, 4, c) = logCons(d4start-(c-1));
        conProfile(centGrat + [-1 1], 4, c) = logCons(d4start-2-(c-1));
        conProfile(centGrat + [-2 2], 4, c) = logCons(d4start-3-(c-1));
        conProfile(centGrat + [-3 3], 4, c) = logCons(d4start-4-(c-1));
    end 
end
    
%% calculate spatial frequency centers for each dispersion
sfsLost = [0, 2, 4, 6];
sfCenters = {};
for sfs = 1 : length(sfsLost)
    sfCenters{sfs} = freqSeries(ceil((1+sfsLost(sfs))/2) : (nSfsTot - sfsLost(sfs)/2)); 
end

meanSf = mean(1:nSfsTot);
multFactors = sfCenters{1}((meanSf+1):end)./sfCenters{1}(meanSf);
% use multFactors for both center sfs and for grating sfs around center 

%% now compute the opacity
% NOTE: These opacities apply if you change only the opacity and not the
% contrast of each grating. If you want to follow Robbe's approach in
% sfMix, then you can safely use only the opacity relationships described
% with O(:, :, 1), i.e. full contrast; to then get the contrast values for
% a particular total contrast, use the same opacities prescribed there but
% with each grating at the new total contrast (e.g. use same opacity but
% have contrast be 0.33 instead of 1.0)
conSort = sort(conProfile, 1, 'descend');
 
O = zeros(size(conProfile));
 
for c = 1 : nConsMix
    for d = 1 : nDispMix
        O(7, d, c) = conSort(7, d, c);
        O(6, d, c) = conSort(6, d, c)./((1 - O(7, d, c)));
        O(5, d, c) = conSort(5, d, c)./((1 - O(7, d, c)).*(1 - O(6, d, c)));
        O(4, d, c) = conSort(4, d, c)./((1 - O(7, d, c)).*(1 - O(6, d, c)).*(1 - O(5, d, c)));
        O(3, d, c) = conSort(3, d, c)./((1 - O(7, d, c)).*(1 - O(6, d, c)).*(1 - O(5, d, c)).*(1 - O(4, d, c)));
        O(2, d, c) = conSort(2, d, c)./((1 - O(7, d, c)).*(1 - O(6, d, c)).*(1 - O(5, d, c)).*(1 - O(4, d, c)).*(1 - O(3, d, c)));
        O(1, d, c) = conSort(1, d, c)./((1 - O(7, d, c)).*(1 - O(6, d, c)).*(1 - O(5, d, c)).*(1 - O(4, d, c)).*(1 - O(3, d, c)).*(1 - O(2, d, c)));
    end
end

%% print opacities nicely by contrast level:
for c = 1 : nConsMix
    reshape(O(:, :, c), nGrats, nConsMix)
end

%% inverse test - i.e. from con&opacity to contrast

temp.con{1} = 0.68;
temp.con{2} = 0.68;
temp.con{3} = 0.68;
temp.con{4} = 0.68;
temp.con{5} = 0.68;
temp.con{6} = 0.68;
temp.con{7} = 0.68;


temp.opa{1} = 0.9690;
temp.opa{2} = 0.3142;
temp.opa{3} = 0.2391;
temp.opa{4} = 0.1412;
temp.opa{5} = 0.1237;
temp.opa{6} = 0.0784;
temp.opa{7} = 0.0727;


%% Code snippet for reversing opacity back into contrast

% going from most superficial to "deepest"/base grating
trial.con{7} = temp.con{7}.*temp.opa{7};
trial.con{6} = temp.con{6}.*temp.opa{6}.*(1-temp.opa{7});
trial.con{5} = temp.con{5}.*temp.opa{5}.*(1-temp.opa{7}).*(1-temp.opa{6});
trial.con{4} = temp.con{4}.*temp.opa{4}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5});
trial.con{3} = temp.con{3}.*temp.opa{3}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4});
trial.con{2} = temp.con{2}.*temp.opa{2}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3});
trial.con{1} = temp.con{1}.*temp.opa{1}.*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3}).*(1-temp.opa{2});
% trial.con{2} = temp.con{2}.*temp.opa{2}.*(1-temp.opa{9}).*(1-temp.opa{8}).*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3});
% trial.con{1} = temp.con{1}.*temp.opa{1}.*(1-temp.opa{9}).*(1-temp.opa{8}).*(1-temp.opa{7}).*(1-temp.opa{6}).*(1-temp.opa{5}).*(1-temp.opa{4}).*(1-temp.opa{3}).*(1-temp.opa{2});



