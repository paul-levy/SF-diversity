function [log_bw] = bw_lin_to_log(lin_low, lin_high)

% Given the low/high sf in cpd, returns number of octaves separating the
% two values

log_bw = log(lin_high/lin_low);
% log_bw = log(10.^lin_high)/log(10.^lin_low) - 2;

end

