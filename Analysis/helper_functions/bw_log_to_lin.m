function [lin_bw, sf_range] = bw_log_to_lin(log_bw, pref_sf)

% given the preferred SF and octave bandwidth, returns the corresponding
% (linear) bounds in cpd

less_half = 2.^(log2(pref_sf) - log_bw/2);
more_half = 2.^(log_bw/2 + log2(pref_sf));

sf_range = [less_half more_half];

lin_bw = more_half - less_half;

end

