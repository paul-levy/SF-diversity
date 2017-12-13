%% First, run sfMixAltLoad and load the corresponding output file
fprintf('\t   ****************************************************************\n');
fprintf('************First, run sfMixAltLoad (if needed) and load the corresponding output file************\n');
fprintf('\t   ****************************************************************\n');

cellNames = {'m000r0'};
dataPath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/recordings/';
savePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/altExp/analysis/';

sfMixAltLoad(cellName, 'loadPath', dataPath, 'savePath', savePath);
%% load

load([savePath, cellNames{1}, '_sfm']);

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

    for c = 1 : length(v_cons)
        subplot(1, length(v_cons), c);
        v_sfs = ~isnan(respMean(d, :, v_cons(c)));
        
        errorbar(all_sfs(v_sfs), respMean(d, v_sfs, v_cons(c)), respVar(d, v_sfs, v_cons(c)));
        set(gca, 'xscale', 'log');
        xlabel('sf (c/deg)'); ylabel('resp (sps)');
        title(sprintf('D%g: contrast: %.3f', d, all_cons(v_cons(c))));
        
    end
end

%% Plot just "sfMix" contrasts
% i.e. highest (up to) 4 contrasts for each dispersion
mixCons = 4;

figure();
for d = 1 : nDisps
    v_cons = val_con_by_disp{d};
    v_cons = v_cons(max(1, end-mixCons+1):end); % max(1, .) for when there are fewer contrasts than 4

    for c = length(v_cons) : -1 : 1
        subplot(nDisps, mixCons, (d-1)*mixCons + length(v_cons)-(c-1));
        v_sfs = ~isnan(respMean(d, :, v_cons(c)));
        
        errorbar(all_sfs(v_sfs), respMean(d, v_sfs, v_cons(c)), respVar(d, v_sfs, v_cons(c)));
        set(gca, 'xscale', 'log');
        xlabel('sf (c/deg)'); ylabel('resp (sps)');
        title(sprintf('D%g: contrast: %.3f', d, all_cons(v_cons(c))));
        
    end
end




