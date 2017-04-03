function [magno, parvo] = make_LGN(eccentricity, freqz)
% parameters set by scanning "the literature"
% eccentricity factor will work for our recorded range - [3, 6] degrees

% Assumptions:
% 1) Contrast tuning does not change within 3 - 6 degrees eccentricity
%	- support: 	Spear et. al, 1994; figures 6, 7
% 2) Center and surround are in constant ratio within 3 - 6 degrees eccentricity
%	- support:
% 3) Center radius grows linearly within 3 - 6 degrees eccentricity
%   - support: 	Spear et. al, 1994; figure 8
%				
% 4) Surround mechanism gain is as stated
%	- support: Levitt et. al, 2001; table 3

%% CRF
% Alitto et. al 2011
c50_m = 0.15;
c50_p = 0.5;

n = 1; % we need to keep things linear!

magno.c50 = c50_m;
parvo.c50 = c50_p;

% magno.crf = make_CRF(c50_m, n);
% parvo.crf = make_CRF(c50_p, n);

%% Spatial frequency tuning

radius_ratio_m = 0.10/0.72; % from Croner, Kaplan, 1994
radius_ratio_p = 0.04/0.3;  % from Croner, Kaplan, 1994

gain_ratio = 0.55;           % from Sokol (thesis) & Levitt et. al, 2001

magno_min_rc = 0.06;        % visual degrees, at 3 degrees eccentricity
parvo_min_rc = 0.03;        % visual degrees, at 3 degrees eccentricity

eccentricity_scale_m = 1.5; % i.e. per degree eccentricity, r_c goes up by 1.5x
eccentricity_scale_p = eccentricity_scale_m; % i.e. per degree eccentricity
min_eccen = 3;

curr_rc_m = magno_min_rc * (1 + eccentricity_scale_m * (eccentricity - min_eccen));
magno.sf = best_DoG(1, 1 / (pi * curr_rc_m), gain_ratio, radius_ratio_m, freqz);

curr_rc_p = parvo_min_rc * (1 + eccentricity_scale_p * (eccentricity - min_eccen));
parvo.sf = best_DoG(1, 1 / (pi * curr_rc_p), gain_ratio, radius_ratio_p, freqz);

end

