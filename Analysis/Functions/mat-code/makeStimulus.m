function [Or, Tf, Co, Ph, Sf, trial_used] = makeStimulus(stimFamily, conLevel, sf_c, ori, tf_c, template)

% 1/23/16 - This function is used to make arbitrary stimuli for use with
% the Robbe V1 model. Rather than fitting the model responses at the actual
% experimental stimuli, we instead can simulate from the model at any
% arbitrary spatial frequency in order to determine peak SF
% response/bandwidth/etc

% If argument 'template' is given, then orientation, phase, and tf_center will be
% taken from the template, which will be actual stimuli from that cell

%% Fixed parameters

num_families = 5;
num_gratings = 9;

spreadVec = logspace(log10(.125), log10(1.25), num_families);
octSeries  = linspace(1.5, -1.5, num_gratings);

%% set contrast and spatial frequency

if conLevel == 1
    total_contrast = 1;
elseif conLevel == 2
    total_contrast = 1/3;
else
    warning('Contrast should be given as 1 [full] or 2 [low/one-third]; setting contrast to 1 (full)');
    total_contrast = 1;
end
        
spread     = spreadVec(stimFamily);
profTemp   = normpdf(octSeries, 0, spread);
profile    = profTemp/sum(profTemp);

if stimFamily == 1 % do this for consistency with actual experiment - for stimFamily 1, only one grating is non-zero; round gives us this
    profile = round(profile);    
end

Sf = 2.^(octSeries(:) + log2(sf_c))'; % final spatial frequency
Co = profile .* total_contrast; % final contrast 

%% The others

if nargin() == 6 % then we've passed in template!
    
   
    % get orientation - IN RADIANS
    OriVal = mode(template.sfm.exp.trial.ori{1}) * pi/180; % pick arbitrary grating, mode for this is cell's pref Ori for experiment
    Or = repmat(OriVal, [1 num_gratings]);
    
    if isfield(template, 'trial_used') % use specified trial
        trial_to_copy = template.trial_used;
    else % get random trial for phase, TF
        % we'll draw from a random trial with the same stimulus family/contrast
        valid_blockIDs = ((stimFamily-1)*(13*2)+1)+(conLevel-1) : 2 : ((stimFamily)*(13*2)-5)+(conLevel-1); % from Robbe's plotSfMix
        num_blockIDs = numel(valid_blockIDs);
        % for phase and TF
        valid_trials = find(template.sfm.exp.trial.blockID == valid_blockIDs(randi(num_blockIDs))); % pick a random block ID
        trial_to_copy = valid_trials(randi(numel(valid_trials))); % pick a random trial from within this
    end

    trial_used = trial_to_copy;
    
    % grab Tf and Phase [IN RADIANS] from each grating for the given trial
    Tf = cellfun(@(x) x(trial_to_copy), template.sfm.exp.trial.tf);    
    Ph = cellfun(@(x) x(trial_to_copy) * pi/180, template.sfm.exp.trial.ph);    
    
%     fprintf('\t...using trial %d...\n', trial_to_copy);
    
else
    
    % orientation
    if numel(ori) == 1
        Or(1:num_gratings) = repmat(ori, [1 num_gratings]);
    elseif numel(ori) == num_gratings
        Or(1:num_gratings) = ori;
    else
        warning('Incorrect number of orientations, using only first value');
        Or(1:num_gratings) = repmat(ori(1), [1 num_gratings]);
    end
    
    Tf = random('norm', tf_c, tf_c/5, [1 num_gratings]); % sigma = 1/5fth of mean tf_mu/5 - used in the actual experiment
    Ph = rand(1, num_gratings) * 360; % phases are uniform, random distributed over [0 360]
    
 end



