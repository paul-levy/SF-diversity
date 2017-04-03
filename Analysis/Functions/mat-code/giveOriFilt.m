
function [filt] = giveOriFilt(imSizeDeg, pixSizeDeg, prefSf, prefOri, dOrder, aRatio)

% giveOriFilt returns a spatial filter which is a dth-order derivative of
% a 2-D Gaussian. 
%
% imSizeDeg  = image size in degrees
% pixSizeDeg = pixel size in degrees
% prefSf     = peak spatial frequency in cycles/deg
% prefOri    = peak orientation in radians
% dOrder     = derivative order (integer)
% aRatio     = aspect ratio of the Gaussian 
%
% oriTuning = (cos(oris-prefOri).^2 .* exp((aRatio^2-1)*cos(oris-prefOri).^2)).^(dOrder/2);
% sfTuning  = sfs.^dOrder .* exp(- sfs.^2 ./ (2*sx^2));


pixPerDeg = 1/pixSizeDeg;
npts2     = round(0.5*imSizeDeg*pixPerDeg);
psfPixels = 2*npts2*prefSf/pixPerDeg;                                      % convert peak sf from cycles/degree to pixels
sx        = psfPixels/max(sqrt(dOrder), 0.01);                             % MAGIC
sy        = sx/aRatio;
[X,Y]     = meshgrid(-npts2:npts2, -npts2:npts2);
rX        = cos(prefOri)*X+sin(prefOri)*Y;  
rY        = -sin(prefOri)*X+cos(prefOri)*Y;
ffilt     = exp(-(rX.^2/(2*sx^2) + rY.^2/(2*sy^2))) .* (-sqrt(-1)*rX).^dOrder;
filt      = real(fftshift(ifft2(ifftshift(ffilt))));
