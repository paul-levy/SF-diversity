from os import chdir as cd
from model_responses import SFMGiveBof

def data(cellNum):
    # Given a cell number, plots just the data with descriptive model fits for that cell
    
    # Set paths - pick one
    base = '/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/'; # CNS
    # base = '/home/pl1465/modelCluster/'; # cluster
    # base = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/'; # local
   
    currentPath  = base + 'sfDiv-python/Analysis/Functions';
    loadPath     = base + 'sfDiv-python/Analysis/Structures';
    functionPath = base + 'sfDiv-python/Analysis/Functions';

    # load data, set constants
    cd(loadPath)
    dataList = numpy.load('dataList').item();
    S = numpy.load(N['unitName'][cellNum-1] + '_sfm.npy').item();
    cd(currentPath)
    
    plot_steps = 100; # how many (log) steps to take between 0.3 and 10 for plotting descriptive curves?
    
    params = S['sfm']['mod']['fit']['params'];

    # Get model prediction
    cd(functionPath);
    modResp = SFMGiveBof(params, S);
    modResp = modResp['respModel']; # unpack the dictionary
    
    
'''

%% Get the mean model predictions, split out by stimulus condition
fprintf('\n \n')

% Analyze the stimulus-driven responses for the orientation tuning curve
oriBlockIDs = [131:2:155, 132:2:136];
iC = 0;

for iB = oriBlockIDs
    indCond = find(S.sfm(1).exp.trial.blockID == iB);    
    if ~isempty(indCond)
        iC         = iC+1;
        rateOr(iC) = mean(respModel{1}(indCond));
    end
end


% Analyze the stimulus-driven responses for the contrast response function
conBlockIDs = [138:2:156];
iC = 0;

for iB = conBlockIDs
    indCond = find(S.sfm(1).exp.trial.blockID == iB);    
    if ~isempty(indCond)
        iC         = iC+1;
        rateCo(iC) = mean(respModel{1}(indCond));
    end
end


% Analyze the stimulus-driven responses for the spatial frequency mixtures
for iE = 1:2
    for iW = 1:5

        StimBlockIDs  = ((iW-1)*(13*2)+1)+(iE-1) : 2 : ((iW)*(13*2)-5)+(iE-1);
        nStimBlockIDs = length(StimBlockIDs);
        
        % Initialize Variables        
        rate{iW}{iE} = nan(1, nStimBlockIDs);
        iC = 0;

        for iB = StimBlockIDs
            indCond = find(S.sfm(1).exp.trial.blockID == iB);
            if ~isempty(indCond)
                iC                  = iC+1;
                rateSfm{iW}{iE}(iC) = mean(respModel{1}(indCond));
            end
        end
    end
end


%% Make figure
% get spatial filter
cd(functionPath)
imSizeDeg = S.sfm(1).exp.size(1);
pixSize   = 0.0028;
prefSf    = params(2);
prefOri   = pi/180 * params(1);
dOrder    = params(4);
aRatio    = params(3);
filtTemp  = giveOriFilt(imSizeDeg, pixSize, prefSf, prefOri, dOrder, aRatio);
filt      = (filtTemp - filtTemp(1))/max(abs(filtTemp(:) - filtTemp(1)));
cd(currentPath)

% Get spatial frequency tuning filter
ds    = params(5);
theta = pi/180*linspace(0, 360, 100);
o     = (cos(theta).^2 .* exp(((aRatio^2)-1) * cos(theta).^2)).^(dOrder/2);
oMax  = exp(((aRatio^2)-1)).^(dOrder/2);
oNl   = o/oMax;
e     = 1 + (ds*.5*(-1+(square(theta + pi/2))));
f     = oNl.*e;


% Compute spatial frequency tuning
omega = logspace(-2, 2, 1000);
sfRel = omega./params(2);
s     = omega.^dOrder .* exp(-dOrder/2 * sfRel.^2);
sMax  = params(2).^dOrder .* exp(-dOrder/2);
sfExc = s/sMax;


% Get spatial frequency tuning suppressive signal
% Broadly tuned pool
peakSf{1}    = S.sfm(1).mod.normalization.pref.sf{1}; 
n{1}         = 0.75;   
gain{1}      = .57;  

% Narrowly tuned pool
peakSf{2}    = S.sfm(1).mod.normalization.pref.sf{2}; 
n{2}         = 1.5;   
gain{2}      = .614;  

% Asymmetry in pool composition
asym = params(13);    

% Weight vector
weight{1} = 1 + asym*(log(peakSf{1}) - mean(log(peakSf{1})));
weight{2} = 1 + asym*(log(peakSf{2}) - mean(log(peakSf{2})));

for iP = 1:2
    for iU = 1:numel(peakSf{iP})
        sfRel = omega./peakSf{iP}(iU);
        s     = omega.^n{iP} .* exp(-n{iP}/2 * sfRel.^2);
        sMax  = peakSf{iP}(iU).^n{iP} .* exp(-n{iP}/2);
        
        sfFilt{iP}(:,iU) = gain{iP} * s/sMax;
    end
end
sfInh  = params(6)*.5*(weight{1}*sfFilt{1}'.^2 + weight{2}*sfFilt{2}'.^2);
sfNorm = -.5*(weight{1}*sfFilt{1}'.^2 + weight{2}*sfFilt{2}'.^2);
sfNorm = sfNorm/max(abs(sfNorm));


% Set colors
col{1} = [1.0 0.0 0.0];
col{2} = [1.0 0.5 0.0];
col{3} = [0.3 1.0 0.0];
col{4} = [0.2 0.5 1.0];
col{5} = [0.7 0.0 0.7];
col{6} = [1.0 1.0 0.0];
col{7} = [0.5 0.5 0.5];

% Set axes
stimSf  = S.sfm(1).exp.sf{1}{1};
stimOri = S.sfm(1).exp.ori;
stimCon = S.sfm(1).exp.con;
maxPlot = 10*ceil(.1*1.75*max(S.sfm(1).exp.oriRateMean));
yAx     = linspace(0, maxPlot, 2);
mu      = logspace(-1, 2, 100);

plotSf = logspace(log10(0.3), log10(10), plot_steps);

% Top row: spatial filter, suppressive signal, and nonlinearity
set(figure(1), 'OuterPosition', [200 200 2000 1200])
subplot(4,5,3)
imagesc(filt)
hold on, box off, axis square, axis off
colormap('gray'), caxis([-1 1])
title(sprintf('%0.3g deg', imSizeDeg))

subplot(4,5,4)
semilogx([omega(1) omega(end)], [0 0], 'k--')
hold on, box off, axis square, axis off
semilogx([.01 .01], [-1 1], 'k--')
semilogx([.1 .1], [-1 1], 'k--')
semilogx([1 1], [-1 1], 'k--')
semilogx([10 10], [-1 1], 'k--')
semilogx([100 100], [-1 1], 'k--')
semilogx(omega, sfExc, 'k-')
semilogx(omega, sfInh, 'r-', 'linewidth', 2)
semilogx(omega, sfNorm, 'r-', 'linewidth', 1)
axis([omega(1) omega(end) -1 1])
title('Suppressive signal')

subplot(4,5,5)
plot([-1 1], [0 0], 'k--')
hold on, box off, axis square, axis off
plot([0 0], [-.1 1], 'k--')
plot(linspace(-1,1,100), max(0, linspace(-1,1,100)).^params(8), 'r-', 'linewidth', 2)
axis([-1 1 -.1 1])
title('Nonlinearity')

% Top row: variance to mean relation
subplot(4,5,1)
loglog([.01 1000], [.01 1000], 'r--')
hold on, box off, axis square
loglog(S.sfm(1).exp.oriRateMean, S.sfm(1).exp.oriRateVar, 'ko', 'markerfacecolor', col{6}, 'markersize', 6)
loglog(S.sfm(1).exp.conRateMean, S.sfm(1).exp.conRateVar, 'ko', 'markerfacecolor', col{7}, 'markersize', 6)
axis([.01 1000 .01 1000])
xlabel('Mean (spikes)')
ylabel('Variance (spikes^2)')

% Second row: orientation tuning and contrast response function
subplot(4,5,6)
plot(stimOri, rateOr, '-', 'linewidth', 2, 'color', col{6})
hold on, box off, axis square
plot([0 360], [S.sfm(1).exp.sponRateMean S.sfm(1).exp.sponRateMean], 'k--', 'linewidth', 2)
plot(stimOri, S.sfm(1).exp.oriRateMean, 'ko-', 'markerfacecolor', col{6}, 'markersize', 8)
xlabel('Orientation (deg)')
ylabel('Response (ips)')
axis([0 360 0 maxPlot])

subplot(4,5,7)
semilogx(stimCon, rateCo, '-', 'linewidth', 2, 'color', col{7})
hold on, box off, axis square
semilogx([.01 1], [S.sfm(1).exp.sponRateMean S.sfm(1).exp.sponRateMean], 'k--', 'linewidth', 2)
semilogx(stimCon, S.sfm(1).exp.conRateMean, 'ko-', 'markerfacecolor', col{7}, 'markersize', 8)
xlabel('Contrast (%)')
ylabel('Response (ips)')
axis([.01 1 0 maxPlot])

for iW = 1:5
    for iE = 1:2
        panelNum = ((iE+1)*5) + iW;
        
        descr_model = S.sfm.descr_model;
        
        subplot(4,5,1)
        loglog(S.sfm(1).exp.sfRateMean{iW}{iE}, S.sfm(1).exp.sfRateVar{iW}{iE}, 'ko', 'markerfacecolor', col{iW}, 'markersize', 6)
        hold on, box off, axis square
        axis([.01 1000 .01 1000])
        
        subplot(4,5,panelNum)
        semilogx([.1 10], [S.sfm(1).exp.sponRateMean S.sfm(1).exp.sponRateMean], 'k--', 'linewidth', 2)
        hold on, box off, axis square

    % newer form
        if plot_fullSim % then use the sf_c and responses from modelSimulation.m
            semilogx(S.sfm.mod.sim.sf_c, S.sfm.mod.sim.resp{iW}{iE}, 'ro', 'color', col{iW});
            semilogx(plotSf, flexible_Gauss(descr_model.func_fit{iW}{iE}, plotSf), '-', 'linewidth', 2, 'color', col{iW});
            semilogx(plotSf, flexible_Gauss(descr_model.func_fit{1}{1}, plotSf), '-', 'linewidth', 1, 'color', col{1});
        else
            semilogx(stimSf, rateSfm{iW}{iE}, '-', 'linewidth', 2, 'color', col{iW});
            semilogx(stimSf, rateSfm{1}{1}, '-', 'linewidth', 1, 'color', col{1});
        end
        
    % plot descr. fit to data
        semilogx(stimSf, S.sfm(1).exp.sfRateMean{iW}{iE}, 'ko', 'markerfacecolor', col{iW}, 'markersize', 8);
        semilogx(plotSf, flexible_Gauss(descr_model.data_fit{iW}{iE}, plotSf), '-', 'color', col{iW});
        %         semilogx(plotSf, flexible_Gauss(descr_model.data_fit{iW}{iE}, plotSf), '-', 'linewidth', 1, 'color', col{iW});
        
    % as it was...(pre-01/30)
%         if plot_fullSim % then use the sf_c and responses from modelSimulation.m
%             semilogx(S.sfm.mod.sim.sf_c, S.sfm.mod.sim.resp{iW}{iE}, '-', 'linewidth', 2, 'color', col{iW});
%             semilogx(S.sfm.mod.sim.sf_c, S.sfm.mod.sim.resp{1}{1}, '-', 'linewidth', 1, 'color', col{1});
%         else
%             semilogx(stimSf, rateSfm{iW}{iE}, '-', 'linewidth', 2, 'color', col{iW});
%             semilogx(stimSf, rateSfm{1}{1}, '-', 'linewidth', 1, 'color', col{1});
%         end
        
%         semilogx(stimSf, S.sfm(1).exp.sfRateMean{iW}{iE}, 'ko-', 'markerfacecolor', col{iW}, 'markersize', 8)
        axis([.1 10 0 maxPlot])
        
        if iW == 5 && iE == 1
            legend('spon. rate', 'model resp.', 'fit to model', 'fit to model (1, 1)', 'real resp.', 'fit to data', 'Location', 'BestOutside');
        end
        
        if iW == 1
            xlabel('Spatial frequency (c/deg)')
            ylabel('Response (ips)')
        end
        
        if iW == 5 && iE == 2
            subplot(4,5,1)
            loglog(mu, mu + params(12)*mu.^2, 'k-','linewidth', 2)
        end
    end
end

cd(currentPath)
'''

def organize_responses(responses, S):
    # given the trial-by-trial responses and the cell structure, give back 
    #   the orientation responses,
    #   the CRF
    #   the SFM responses
    
    # Analyze the stimulus-driven responses for orientation tuning
    oriBlockIDs = numpy.concatenate((numpy.arange(131, 155+1, 2), numpy.arange(132, 136+1, 2)))
    # hard-coded, from Robbe; +1 so that we include actual endpoint
    iC = 0;
    
    for iB in oriBlockIDs:
        indCond = S['sfm']['exp']['trial']['blockID'] == iB;
        if indCond: # i.e. isn't empty
            rateOr[iC] = mean(responses[indCond]);
            iC = iC+1;
    
    # Analyze the stimulus-driven responses for the contrast response function
    conBlockIDs = numpy.arange(138, 156+1, 2);
    iC = 0;
    
    for iB in conBlockIDs:
        indCond = S['sfm']['exp']['trial']['blockID'] == iB;
        if indCond: # i.e. isn't empty
            rateCo[iC] = mean(responses[indCond]);
            iC = iC+1;

    
'''

% Analyze the stimulus-driven responses for the contrast response function
conBlockIDs = [138:2:156];
iC = 0;

for iB = conBlockIDs
    indCond = find(S.sfm(1).exp.trial.blockID == iB);    
    if ~isempty(indCond)
        iC         = iC+1;
        rateCo(iC) = mean(respModel{1}(indCond));
    end
end


% Analyze the stimulus-driven responses for the spatial frequency mixtures
for iE = 1:2
    for iW = 1:5

        StimBlockIDs  = ((iW-1)*(13*2)+1)+(iE-1) : 2 : ((iW)*(13*2)-5)+(iE-1);
        nStimBlockIDs = length(StimBlockIDs);
        
        % Initialize Variables        
        rate{iW}{iE} = nan(1, nStimBlockIDs);
        iC = 0;

        for iB = StimBlockIDs
            indCond = find(S.sfm(1).exp.trial.blockID == iB);
            if ~isempty(indCond)
                iC                  = iC+1;
                rateSfm{iW}{iE}(iC) = mean(respModel{1}(indCond));
            end
        end
    end
end
'''