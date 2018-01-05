%% First, run sfMixAltLoad and load the corresponding output file
fprintf('\t   ****************************************************************\n');
fprintf('************First, run sfMixAltLoad (if needed) and load the corresponding output file************\n');
fprintf('\t   ****************************************************************\n');

fprintf('NOTE: Must have CNS shared expo resources to run (best if on CNS machine...)');

cellNames = {'m670l14', 'm670l16', 'm670l22', 'm670l38', 'm670l41', 'm670l46'};
cellId = 6;
% at CNS
% dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
% savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';
% personal mac
dataPath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
savePath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';

%% Run?
sfMixAltLoad(cellNames{cellId}, 'loadPath', dataPath, 'savePath', savePath);

%% load

load([savePath, cellNames{cellId}, '_sfm']);

myround = @(x, digits) round(x*10.^digits)./(10.^digits);
conDig = 3; % round contrast to the 3rd digit

%% determine the contrasts, sfs, and dispersions

data = S.sfm.exp.trial;

all_cons = unique(myround(data.total_con, conDig));
all_cons = all_cons(~isnan(all_cons));

all_sfs = unique(data.cent_sf);
all_sfs = all_sfs(~isnan(all_sfs));

all_disps = unique(data.num_comps);
all_disps = all_disps(all_disps>0); % ignore zero...

nCons = length(all_cons);
nSfs = length(all_sfs);
nDisps = length(all_disps);

con_diffs = diff(all_cons);
closest_cons = all_cons(con_diffs>0.01);

%% organize data
respMean = NaN(nDisps, nSfs, nCons);
respVar = NaN(nDisps, nSfs, nCons);

for d = 1 : nDisps
    val_con_by_disp{d} = [];
    
    for con = 1 : nCons
        for sf = 1 : nSfs
            
            valid_disp = data.num_comps == all_disps(d);
            valid_sf = data.cent_sf == all_sfs(sf);
            valid_con = myround(data.total_con, conDig) == all_cons(con);
            
            valid_tr = find(valid_disp & valid_sf & valid_con);
                       
            respMean(d, sf, con) = mean(data.spikeCount(valid_tr));
            respVar(d, sf, con) = std((data.spikeCount(valid_tr)));
            
        end
        
        if ~isnan(nanmean(respMean(d, :, con)))
            val_con_by_disp{d} = [val_con_by_disp{d} con];
        end 
    end    
end

%% Plots by dispersion

for d = 1 : nDisps
    figure();
    v_cons = val_con_by_disp{d};

    maxResp = max(max(respMean(d, :, :)));
    
    for c = length(v_cons) : -1 : 1
        subplot(length(v_cons), 1, length(v_cons) - (c-1));
        v_sfs = ~isnan(respMean(d, :, v_cons(c)));
        
        errorbar(all_sfs(v_sfs), respMean(d, v_sfs, v_cons(c)), respVar(d, v_sfs, v_cons(c)));
        xlim([min(all_sfs) max(all_sfs)]);
        ylim([0 1.2*maxResp]);
        
        set(gca, 'xscale', 'log');
%         set(gca, 'yscale', 'log');
        xlabel('sf (c/deg)'); ylabel('resp (sps)');
        title(sprintf('D%g: contrast: %.3f', d, all_cons(v_cons(c))));
        
    end
end

%% Plot just "sfMix" contrasts
% i.e. highest (up to) 4 contrasts for each dispersion
mixCons = 4;

maxResp = max(max(max(respMean(:, :, :))));

figure();
for d = 1 : nDisps
    v_cons = val_con_by_disp{d};
    n_v_cons = length(v_cons);
    v_cons = v_cons(max(1, n_v_cons -mixCons+1):n_v_cons); % max(1, .) for when there are fewer contrasts than 4

    for c = length(v_cons) : -1 : 1
        subplot(mixCons, nDisps, d + (length(v_cons)-c)*mixCons);
        v_sfs = ~isnan(respMean(d, :, v_cons(c)));
        
        errorbar(all_sfs(v_sfs), respMean(d, v_sfs, v_cons(c)), respVar(d, v_sfs, v_cons(c)));
        xlim([min(all_sfs) max(all_sfs)]);
        ylim([0 1.2*maxResp]);

        set(gca, 'xscale', 'log');
%         set(gca, 'yscale', 'log');
        xlabel('sf (c/deg)'); ylabel('resp (sps)');
        title(sprintf('D%g: contrast: %.3f', d, all_cons(v_cons(c))));
    end
end

%% CRF

for d = 1 : nDisps
    figure();

    for sf = 1 : nSfs
        subplot(1, nSfs, sf);
%         subplot(d, nSfs, (d-1)*nSfs + sf);
        v_cons = ~isnan(respMean(d, sf, :));
        n_cons = sum(v_cons);
        
        errorbar(all_cons(v_cons), reshape([respMean(d, sf, v_cons)], 1, n_cons), reshape([respVar(d, sf, v_cons)], 1, n_cons));
        set(gca, 'xscale', 'log');
        set(gca, 'yscale', 'log');
        xlabel('contrast'); ylabel('resp (sps)');
        title(sprintf('D%g: sf: %.3f', d, all_sfs(sf)));
        
    end
end


