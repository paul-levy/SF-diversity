function [SF, bw_log] = compute_SF_BW(model, fit, height, sf_range, debug)

% 1/16/17 - This was corrected in the lead up to SfN (sometime 11/16). I had been computing
% octaves not in log2 but rather in log10 - it is field convention to use
% log2!

% Height is defined RELATIVE to baseline
% i.e. baseline = 10, peak = 50, then half height is NOT 25 but 30

% do I want to plot and print?
if nargin() < 5
    debug = 0;
end

bw_log = NaN;
SF = nan(2, 1);

% left-half
left_full_bw = 2 * (fit(4) * sqrt(2*log(1/height))); % + log(fit(3));
left_cpd = fit(3) * exp(-(fit(4) * sqrt(2*log(1/height))));

% right-half
right_full_bw = 2 * (fit(5) * sqrt(2*log(1/height))); % + log(fit(3));
right_cpd = fit(3) * exp((fit(5) * sqrt(2*log(1/height))));

if left_cpd > sf_range(1) && right_cpd < sf_range(end)
    SF = [left_cpd right_cpd];
    bw_log = log2(right_cpd / left_cpd);
end
% otherwise we don't have defined BW!

% keyboard;

if debug == 1
    value_to_find = fit(2) * height + fit(1);
    n_steps = 500;
    SF_plot = logspace(log10(0.1), log10(10), n_steps);
    figure();
    semilogx(SF_plot, eval(sprintf('%s(fit, SF_plot)', model)));
    hold on;
    semilogx([SF(1) SF(2)], [value_to_find value_to_find], 'r--');
    xlabel('SF (cpd)');
    ylabel('Response (ips)');
    title('Fit');
end

end

