function resp_rate = flexible_Gauss(params, stim_sf)

% The descriptive model used to fit cell tuning curves - in this way, we
% can read off preferred SF, octave bandwidth, and response amplitude

respFloor       = params(1);
respRelFloor    = params(2);
sfPref          = params(3);
sigmaLow        = params(4);
sigmaHigh       = params(5);

% Tuning function
sf0   = stim_sf/sfPref;

sigma = sigmaLow .* ones(size(sf0));

sigma(sf0 > 1) = sigmaHigh;

shape = exp((-log(sf0).^2)./(2*sigma.^2));

resp_rate  = max(0.1, respFloor + respRelFloor*shape);

% keyboard;