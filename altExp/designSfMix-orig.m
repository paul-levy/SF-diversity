% Begin with clean slate
clear all
close all
clc
%%
% Set constants
freqSeries = logspace(log10(0.3), log10(10), 11);
freqCent   = freqSeries(end);

spreadVec = logspace(log10(.125), log10(1.25), 5);
octSeries  = linspace(1.5, -1.5, 9);
freqComp   = exp(-((octSeries * log(2)) - log(freqCent)));

for iC = 1:5
    
    spread     = spreadVec(iC);
    profTemp   = normpdf(octSeries, 0, spread);
    profile    = profTemp/sum(profTemp);
    
    conProfile(iC,:) = profile;
    
    figure(1)
    semilogx(freqComp, profile, 'o-', 'color', rand(1,3), 'linewidth', 2)
    hold on, box off, axis square
    axis([1 100 0 1])
end



conSort = fliplr(sort(conProfile, 2));
 
O = zeros(5,9);
 
for W = 1:5
    O(W,9) = conSort(W,9);
    O(W,8) = conSort(W,8)./((1 - O(W,9)));
    O(W,7) = conSort(W,7)./((1 - O(W,9)).*(1 - O(W,8)));
    O(W,6) = conSort(W,6)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)));
    O(W,5) = conSort(W,5)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)).*(1 - O(W,6)));
    O(W,4) = conSort(W,4)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)).*(1 - O(W,6)).*(1 - O(W,5)));
    O(W,3) = conSort(W,3)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)).*(1 - O(W,6)).*(1 - O(W,5)).*(1 - O(W,4)));
    O(W,2) = conSort(W,2)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)).*(1 - O(W,6)).*(1 - O(W,5)).*(1 - O(W,4)).*(1 - O(W,3)));
    O(W,1) = conSort(W,1)./((1 - O(W,9)).*(1 - O(W,8)).*(1 - O(W,7)).*(1 - O(W,6)).*(1 - O(W,5)).*(1 - O(W,4)).*(1 - O(W,3)).*(1 - O(W,2)));
end

opacity = O