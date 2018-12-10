%% makePsth
function [psth, times] = makePsth(spikeTimes, binWidth, makePlot, tfs)
% spikeTimes - in seconds
% binWidth - in seconds
% makePlot - yes (1) or no (0)
% tfs - at what TFs are the stimulus components [in Hz]?

binWidth=1e-2; % bin in s

stimDur = 1; % in seconds
times = linspace(0, stimDur, 1+stimDur./binWidth);
% times = linspace(0, stimDur-binWidth, stimDur./binWidth);

%% analyze - get PSTH, Fourier transform
psth = histcounts(spikeTimes, times);

maxTf = max(tfs); % in Hertz
power = abs(fft(psth));

nyquist = length(power)/2;

%% make plot

if makePlot
    
    subplot(1, 2, 1);
    plot(times(1:end-1), psth);
%     plot(times, psth./binWidth);
    title('PSTH');
    xlabel('time (s)');
    ylabel('spike count (sps)');
    xlim([-stimDur/5.0 6*stimDur./5.0]);
%     ylim([-max(
    
    subplot(1, 2, 2);

%     plot(power);
    stem(power(1:nyquist));
    xlabel('Frequency');
    ylabel('Power');
    xlim([-nyquist/5 6*nyquist/5]);
%     
%     power = abs((fft(psth)));
%     relPower = power(1:10*maxTf);
%     plot(0:length(relPower)-1, relPower);
%    
%     xlim([-length(relPower)/10, 1.1*length(relPower)]);
%     ylabel('Power');
%     xlabel('Temporal frequency (Hz)');
%     
%     plot([periodDur periodDur], [0 max(onlyPos)], '--')
end
