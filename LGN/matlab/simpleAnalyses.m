
%% sf tuning
conz = unique(S.sfm.exp.trial.con{1}(singles));

for i = 1 : length(conz)
    curr_tr = find(S.sfm.exp.trial.con{1} == conz(i));
    curr_sfs = S.sfm.exp.trial.sf{1}(curr_tr);
    unique_sfs = unique(curr_sfs);
    curr_frs = [];
    for j = 1 : length(unique_sfs)
        curr_trz = find(S.sfm.exp.trial.sf{1}(curr_tr) == unique_sfs(j));
        curr_frs(j) = mean(S.sfm.exp.trial.spikeCount(curr_tr(curr_trz)));
    end
    
    figure();
    plot(unique_sfs, curr_frs); hold on;
    title(sprintf('%.2f%% contrast', 100*conz(i)));
    xlabel('sf (cpd)');
    ylabel('firing rate (sps)');
end