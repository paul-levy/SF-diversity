%% testing make_LGN function

freqz = logspace(log10(0.1), log10(100), 100);

conz_lin = linspace(0, 1, 100);
conz_log = logspace(-3, 0, 100);

set(figure(), 'OuterPosition', [200 200 1200 1200]);

eccentricity = 3.5;
[m, p] = make_LGN(eccentricity, freqz);

subplot(4, 2, 1);
semilogx(conz_log, m.crf(conz_log), 'r', conz_log, p.crf(conz_log), 'k', conz_log, m.crf(conz_log) ./ p.crf(conz_log));
legend('Magno', 'Parvo', 'M/P', 'Location', 'Best');
title('CRF - log');

subplot(4, 2, 2);
plot(conz_lin, m.crf(conz_lin), 'r', conz_lin, p.crf(conz_lin), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
axis equal;
title('CRF - linear');

subplot(4, 2, 3);
loglog(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([1e-2 1.1e0]);
title(sprintf('SF tuning - loglog - ecc %.2f', eccentricity));

subplot(4, 2, 4);
semilogx(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([0 1.1e0]);
title(sprintf('SF tuning - semilogx - ecc %.2f', eccentricity));

eccentricity = 5;
[m, p] = make_LGN(eccentricity, freqz);

subplot(4, 2, 5);
loglog(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([1e-2 1.1e0]);
title(sprintf('SF tuning - loglog - ecc %.2f', eccentricity));

subplot(4, 2, 6);
semilogx(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([0 1.1e0]);
title(sprintf('SF tuning - semilogx - ecc %.2f', eccentricity));

eccentricity = 10;
[m, p] = make_LGN(eccentricity, freqz);

subplot(4, 2, 7);
loglog(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([1e-2 1.1e0]);
title(sprintf('SF tuning - loglog - ecc %.2f', eccentricity));

subplot(4, 2, 8);
semilogx(freqz, m.sf(freqz), 'r', freqz, p.sf(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
xlim([1e-1 1e2]); ylim([0 1.1e0]);
title(sprintf('SF tuning - semilogx - ecc %.2f', eccentricity));


%% scratch pad - Tony's form

% gain = 100;
% gain_sur = 0.6;
% f_c = 10.61;
% j_s = 0.3;
% j_s2 = 0.4;

% from Croner, Kaplan, 1994
rc_m = 0.10;
rs_m = 0.72;

rc_p = 0.04;
rs_p = 0.3;

freqz = logspace(log10(0.1), log10(30), 100);
magno = best_DoG(100, 1 / (pi*rc_m), 0.5, rc_m / rs_m, freqz);
parvo = best_DoG(100, 1 / (pi*rc_p), 0.5, rc_p / rs_p, freqz);

set(figure(), 'OuterPosition', [200 200 1200 1200]);

subplot(2, 2, 1);
loglog(freqz, magno(freqz), freqz, parvo(freqz));
ylim([1e-2 1.1e0]);
legend('Magno', 'Parvo', 'Location', 'Best');
title('SF tuning');

subplot(2, 2, 2);
semilogx(freqz, magno(freqz), freqz, parvo(freqz));
ylim([1e-2 1.1e0]);
legend('Magno', 'Parvo', 'Location', 'Best');
title('SF tuning');

%% scratch pad - difference with eccentricity
clear all;

kc_sach = 2e3;
rc_sach = 0.115;
ks_sach = 30;
rs_sach = 0.75;

sach = make_DoG(kc_sach, rc_sach, ks_sach, rs_sach);

% numbers from Croner, Kaplan, 1994 (table 1)

kc_m = 148;
rc_m = 0.10;
ks_m = 1.1;
rs_m = 0.72;

kc_p_close = 325.2;
rc_p_close = 0.03;
ks_p_close = 4.4;
rs_p_close = 0.18;

kc_p_far = 114.7;
rc_p_far = 0.05;
ks_p_far = 0.7;
rs_p_far = 0.43;

magno = make_DoG(kc_m, rc_m, ks_m, rs_m);
p_near = make_DoG(kc_p_close, rc_p_close, ks_p_close, rs_p_close);
p_far = make_DoG(kc_p_far, rc_p_far, ks_p_far, rs_p_far);

freqz = logspace(log10(0.1), log10(30), 100);

set(figure(), 'OuterPosition', [200 200 1200 1200]);

% subplot(2, 2, 1);
% loglog(freqz, m_c(freqz), 'b', freqz, m_s(freqz), 'c');
% ylim([1e0 1.5e2]);
% legend('Center', 'Surround', 'Location', 'Best');
% title('Magno SF tuning');
% 
% subplot(2, 2, 2);
% loglog(freqz, p_c(freqz), 'b', freqz, p_s(freqz), 'c');
% ylim([1e0 1.5e2]);
% legend('Center', 'Surround', 'Location', 'Best');
% title('Parvo SF tuning');

magno = @(f) magno(f) ./ max(magno(freqz));
p_near = @(f) p_near(f) ./ max(p_near(freqz));
p_far = @(f) p_far(f) ./ max(p_far(freqz));

subplot(2, 2, 3);
loglog(freqz, magno(freqz), freqz, p_near(freqz), 'r', freqz, p_far(freqz), 'k');
legend('Magno', 'Parvo near', 'Parvo far', 'Location', 'Best');
ylim([1e-2 1.5e0]);
title('SF tuning - loglog');

subplot(2, 2, 4);
semilogx(freqz, magno(freqz), freqz, p_near(freqz), 'r', freqz, p_far(freqz), 'k');
legend('Magno', 'Parvo near', 'Parvo far', 'Location', 'Best');
ylim([0 1.5e0]);
title('SF tuning - semilogx');


%% real thing
clear all;
% CRF

c50_m = 0.15;
c50_p = 0.5;

exp = 1; % we need to keep things linear!

CRF_m = make_CRF(c50_m, exp);
CRF_p = make_CRF(c50_p, exp);

conz_lin = linspace(0, 1, 100);
conz_log = logspace(-3, 0, 100);

% SF tuning
central_sf_one = 3.5;
central_sf_two = 2.5;
peak_separation = 1; % in octaves, between P and M cells
[~, prefs] = bw_log_to_lin(peak_separation, central_sf_one);

gc_m = 100;
rc_m = 1.2;
gs_m = 25;
rs_m = 1.34;
prefSF_m = prefs(1);    % because magno have lower SF pref

gc_p = 100;
rc_p = 1.8;
gs_p = 60;
rs_p = 2;
prefSF_p = prefs(2);    % becuase parvo have higher SF pref

freqz = logspace(log10(0.1), log10(30), 100);

[sf_m, m_c, m_s] = alt_DoG(gc_m, rc_m, gs_m, rs_m, prefSF_m, freqz);
[sf_p, p_c, p_s] = alt_DoG(gc_p, rc_p, gs_p, rs_p, prefSF_p, freqz);

[~, prefs] = bw_log_to_lin(peak_separation, central_sf_two);
prefSF_m = prefs(1);    % because magno have lower SF pref
prefSF_p = prefs(2);    % because magno have lower SF pref

[sf_m_diff, m_c, m_s] = alt_DoG(gc_m, rc_m, gs_m, rs_m, prefSF_m, freqz);
[sf_p_diff, p_c, p_s] = alt_DoG(gc_p, rc_p, gs_p, rs_p, prefSF_p, freqz);

set(figure(), 'OuterPosition', [200 200 1200 1200]);

subplot(3, 2, 1);
semilogx(conz_log, CRF_m(conz_log), 'r', conz_log, CRF_p(conz_log), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
title('CRF - log');

subplot(3, 2, 2);
plot(conz_lin, CRF_m(conz_lin), 'r', conz_lin, CRF_p(conz_lin), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
title('CRF - linear');

subplot(3, 2, 3);
loglog(freqz, sf_m(freqz), 'r', freqz, sf_m_diff(freqz), 'k');
legend('Magno', 'Magno2', 'Location', 'Best');
ylim([1e-2 1.1]);
title('SF tuning - loglog');

subplot(3, 2, 4);
semilogx(freqz, sf_m(freqz), 'r', freqz, sf_m_diff(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
ylim([0 1.1]);
title('SF tuning - semilogx');

subplot(3, 2, 5);
loglog(freqz, sf_m(freqz), 'r', freqz, sf_p_diff(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
ylim([1e-2 1.1]);
title('SF tuning - loglog');

subplot(3, 2, 6);
semilogx(freqz, sf_m(freqz), 'r', freqz, sf_p_diff(freqz), 'k');
legend('Magno', 'Parvo', 'Location', 'Best');
ylim([0 1.1]);
title('SF tuning - semilogx');


