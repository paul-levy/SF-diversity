function [ output_args ] = organize_responses( input_args )
%ORGANIZE_RESPONSES Summary of this function goes here
%   Detailed explanation goes here

%BOOTSTRAP_TRIAL Creates a bootstrapped recording for a given cell

data = cell.sfm.exp;
epsilon = 1e-4;

% only look at trials with correct orientation
ori_pref = mode(data.trial.ori{1}(:));
ori_check = abs(data.trial.ori{1}(:) - ori_pref) < epsilon;

% set spontaneous rate
boot_cell.sfm.exp.sponRateMean = cell.sfm.exp.sponRateMean;

%% the work
for family = 1 : 5
    for contrast_hi = 1 :2
        
        % set SF center values
        boot_cell.sfm.exp.sf{family}{contrast_hi} = cell.sfm.exp.sf{family}{contrast_hi};
        
        % only look at trials with correct center contrast for given family/contrast
        center_con = get_center_con(family, contrast_hi);
        con_check = abs(data.trial.con{1}(:) - center_con) < epsilon;
        
        sf_centers  = data.sf{1}{1};

        for sf_i = 1 : length(sf_centers)
            % what indices can I grab from?
            sf_check = (data.trial.sf{1}(:) == sf_centers(sf_i));
            indices = find(con_check & ori_check & sf_check);
            
            n_samples = length(indices);
%             fprintf('%d SAMPLES: Family %d, con %d, sf_center %d\n', n_samples, family, contrast_hi, sf_i);
            
            if family == 1 && contrast_hi == 1 && sf_i == 1
                curr_add = 1 : n_samples;
            else
                curr_add = curr_add(end) + 1 : curr_add(end) + n_samples;
            end
            
            % the indices of bootstrapped trials
            boot_indices = datasample(indices, n_samples, 'Replace', true);
            
            % set trial data
            boot_cell.sfm.exp.trial.sf{1}(curr_add) = cell.sfm.exp.trial.sf{1}(boot_indices);
            boot_cell.sfm.exp.trial.spikeCount(curr_add) = cell.sfm.exp.trial.spikeCount(boot_indices);
            boot_cell.sfm.exp.trial.duration(curr_add) = cell.sfm.exp.trial.duration(boot_indices);
            boot_cell.sfm.exp.trial.con{1}(curr_add) = cell.sfm.exp.trial.con{1}(boot_indices);
            boot_cell.sfm.exp.trial.ori{1}(curr_add) = cell.sfm.exp.trial.ori{1}(boot_indices);
            
            % set condition averages
            boot_cell.sfm.exp.sfRateMean{family}{contrast_hi}(sf_i) = mean(boot_cell.sfm.exp.trial.spikeCount(curr_add) ./ boot_cell.sfm.exp.trial.duration(curr_add));
            
        end
        
    end
end

end

