import math, cmath, numpy, os
from helper_fcns import makeStimulus, random_in_range, getNormParams, genNormWeights, setSigmaFilter, evalSigmaFilter
from scipy.stats import norm, mode, lognorm, nbinom
from numpy.matlib import repmat
import time

import pdb

fft = numpy.fft

# orientation filter used in plotting (only, I think?)
def oriFilt(imSizeDeg, pixSizeDeg, prefSf, prefOri, dOrder, aRatio):
    
    # returns a spatial filter which is dth-order derivate of 2-D Gaussian

    # imSizeDeg  = image size in degrees
    # pixSizeDeg = pixel size in degrees
    # prefSf     = peak spatial frequency in cycles/deg
    # prefOri    = peak orientation in radians
    # dOrder     = derivative order (integer)
    # aRatio     = aspect ratio of the Gaussian 
    #
    # oriTuning = (cos(oris-prefOri).^2 .* exp((aRatio^2-1)*cos(oris-prefOri).^2)).^(dOrder/2);
    # sfTuning  = sfs.^dOrder .* exp(- sfs.^2 ./ (2*sx^2));


    pixPerDeg = 1/pixSizeDeg;
    npts2     = round(0.5*imSizeDeg*pixPerDeg);
    psfPixels = 2*npts2*prefSf/pixPerDeg;                                      # convert peak sf from cycles/degree to pixels
    sx        = psfPixels/max(math.sqrt(dOrder), 0.01);                             # MAGIC
    sy        = sx/aRatio;
    
    [X, Y] = numpy.mgrid[-npts2:npts2, -npts2:npts2];
    rX    = numpy.cos(prefOri) * X + numpy.sin(prefOri) * Y;
    rY    = -numpy.sin(prefOri) * X + numpy.cos(prefOri) * Y;

    ffilt = numpy.exp(-(pow(rX, 2) / (2 * pow(sx, 2)) + pow(rY, 2) / (2 * pow(sy, 2)))) * pow(-1j*rX, dOrder);
    
    filt = fft.fftshift(fft.ifft2(fft.ifftshift(ffilt)));
    return filt.real;

# SFMSimpleResp - Used in Robbe V1 model - excitatory, linear filter response
def SFMSimpleResp(S, channel, stimParams = []):
    '''
    # S is the cellStructure
    # channel is a dictionary with the parameters specifying the filter/model
    # returns object with simpleResp and other things

    # SFMSimpleResp       Computes response of simple cell for sfmix experiment

    # SFMSimpleResp(varargin) returns a simple cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # 1/23/17 - Edits: Added stimParamsm, make_own_stim so that I can set what
    # stimuli I want when simulating from model
    '''
    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;
        if not 'template' in stimParams:
            stimParams['template'] = S;
        if not 'repeats' in stimParams:
            stimParams['repeats'] = 10; # why 10? To match experimental #repetitions

    # Load the data structure
    T = S.get('sfm');

    # Get preferred stimulus values
    prefSf = channel.get('pref').get('sf');                              # in cycles per degree
    # CHECK LINE BELOW
    prefTf = round(numpy.nanmean(T.get('exp').get('trial').get('tf')[0]));     # in cycles per second

    # Get directional selectivity - removed 7/18/17

    # Get derivative order in space and time
    dOrdSp = channel.get('dord').get('sp');
    dOrdTi = channel.get('dord').get('ti');

    # Get aspect ratio in space - removed 7/18/17

    # Get spatial coordinates
    xCo = 0;                                                              # in visual degrees, centered on stimulus center
    yCo = 0;                                                              # in visual degrees, centered on stimulus center

    # Store some results in M
    M = dict();
    pref = dict();
    dord = dict();
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    dord.setdefault('sp', dOrdSp);
    dord.setdefault('ti', dOrdTi);
    
    M.setdefault('pref', pref);
    M.setdefault('dord', dord);
    
    # Pre-allocate memory
    z             = T.get('exp').get('trial');
    nSf           = 1;
    nGratings     = 5;
    nFrames       = 120;
    if make_own_stim == 1:
        nTrials = stimParams.get('repeats'); # to keep consistent with number of repetitions used for each stim. condition
    else: # CHECK THIS GUY BELOW
        nTrials = len(z['num']);
    
    # set it zero
    M['simpleResp'] = numpy.zeros((nFrames, nTrials));

    # Compute simple cell response for all trials
    for p in range(nTrials): 
    
        # Set stim parameters
        if make_own_stim == 1:

            all_stim = makeStimulus(stimParams.get('stimFamily'), stimParams.get('conLevel'), \
                                                                    stimParams.get('sf_c'), stimParams.get('template'));

            stimOr = all_stim.get('Ori');
            stimTf = all_stim.get('Tf');
            stimCo = all_stim.get('Con');
            stimPh = all_stim.get('Ph');
            stimSf = all_stim.get('Sf');
        else:
            stimOr = numpy.empty((nGratings,));
            stimTf = numpy.empty((nGratings,));
            stimCo = numpy.empty((nGratings,));
            stimPh = numpy.empty((nGratings,));
            stimSf = numpy.empty((nGratings,));
            
            for iC in range(nGratings):
                stimOr[iC] = z.get('ori')[iC][p] * math.pi/180; # in radians
                stimTf[iC] = z.get('tf')[iC][p];          # in cycles per second
                stimCo[iC] = z.get('con')[iC][p];         # in Michelson contrast
                stimPh[iC] = z.get('ph')[iC][p] * math.pi/180;  # in radians
                stimSf[iC] = z.get('sf')[iC][p];          # in cycles per degree
                
        if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
            continue;
                
        # I. Orientation, spatial frequency and temporal frequency
        # Compute orientation tuning - removed 7/18/17

        # Compute spatial frequency tuning
        sfRel = stimSf / prefSf;
        s     = pow(stimSf, dOrdSp) * numpy.exp(-dOrdSp/2 * pow(sfRel, 2));
        sMax  = pow(prefSf, dOrdSp) * numpy.exp(-dOrdSp/2);
        sNl   = s/sMax;
        selSf = sNl;

        # Compute temporal frequency tuning
        tfRel = stimTf / prefTf;
        t     = pow(stimTf, dOrdTi) * numpy.exp(-dOrdTi/2 * pow(tfRel, 2));
        tMax  = pow(prefTf, dOrdTi) * numpy.exp(-dOrdTi/2);
        tNl   = t/tMax;
        selTf = tNl;

        # II. Phase, space and time
        omegaX = stimSf * numpy.cos(stimOr); # the stimulus in frequency space
        omegaY = stimSf * numpy.sin(stimOr);
        omegaT = stimTf;

        P = numpy.empty((nFrames, 3)); # nFrames for number of frames, two for x and y coordinate, one for time
        P[:,0] = 2*math.pi*xCo*numpy.ones(nFrames,);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P[:,1] = 2*math.pi*yCo*numpy.ones(nFrames,); # P(:,0) and p(:,1) describe location of the filters in space

        # Pre-allocate some variables
        if nSf == 1:
            respSimple = numpy.zeros(nFrames,);
        else:
            respSimple = numpy.zeros(nFrames, nSf);

        for iF in range(nSf):
            if isinstance(xCo, int):
                factor = 1;
            else:
                factor = len(xCo);

            linR1 = numpy.zeros((nFrames*factor, nGratings)); # pre-allocation
            linR2 = numpy.zeros((nFrames*factor, nGratings));
            linR3 = numpy.zeros((nFrames*factor, nGratings));
            linR4 = numpy.zeros((nFrames*factor, nGratings));
            
            computeSum = 0; # important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response

            for c in range(nGratings): # there are up to nine stimulus components
                selSi = selSf[c]*selTf[c]; # filter sensitivity for the sinusoid in the frequency domain

                if selSi != 0 and stimCo[c] != 0:
                    computeSum = 1;
                                   
                    # Use the effective number of frames displayed/stimulus duration
                    stimPos = numpy.asarray(range(nFrames))/float(nFrames) + \
                                            stimPh[c] / (2*math.pi*stimTf[c]); # 120 frames + the appropriate phase-offset
                    P3Temp  = numpy.full_like(P[:, 1], stimPos);
                    #P3Temp  = repmat(stimPos, 1, len(xCo));
                    P[:,2]  = 2*math.pi*P3Temp; # P(:,2) describes relative location of the filters in time.

                    omegas = numpy.vstack((omegaX[c], omegaY[c], omegaT[c])); # make this a 3 x len(omegaX) array
                    rComplex = selSi*stimCo[c]*numpy.exp(1j*numpy.dot(P, omegas));

                    linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
                    linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
                    linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
                    linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);

                if computeSum == 1:
                    respSimple1 = numpy.maximum(0, linR1.sum(1)); # superposition and half-wave rectification,...
                    respSimple2 = numpy.maximum(0, linR2.sum(1));
                    respSimple3 = numpy.maximum(0, linR3.sum(1));
                    respSimple4 = numpy.maximum(0, linR4.sum(1));

                    # if channel is tuned, it is phase selective...
                    if nSf == 1:
                        if channel.get('dord').get('sp') != 0:
                            respSimple = respSimple1;
                        elif channel.get('dord').get('sp') == 0:
                            respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
                                + pow(respSimple3, 2) + pow(respSimple4, 2); 
                            respSimple = numpy.sqrt(respComplex);                         
                    else:        
                        if channel.get('dord').get('sp') != 0:
                            respSimple[iF, :] = respSimple1;
                        elif channel.get('dord').get('sp') == 0:
                            respComplex = pow(respSimple1, 2) + pow(respSimple2, 2) \
                                + pow(respSimple3, 2) + pow(respSimple4, 2); 
                            respSimple[iF, :] = numpy.sqrt(respComplex);
                        
        #pdb.set_trace();
            
        # Store response in desired format
        M['simpleResp'][:,p] = respSimple;
        
    return M;
        
def SFMNormResp(unitName, loadPath, normPool, stimParams = []):

# returns M which contains normPool response, trial_used, filter preferences

# SFNNormResp       Computes normalization response for sfMix experiment

# SFMNormResp(unitName, varargin) returns a simulated V1-normalization
# response to the mixture stimuli used in sfMix. The normalization pool
# consists of spatially distributed filters, tuned for orientation, spatial 
# frequency, and temporal frequency. The tuning functions decribe responses
# of spatial filters obtained by taking a d-th order derivative of a 
# 2-D Gaussian. The pool includes both filters that are broadly and 
# narrowly tuned for spatial frequency. The filters are chosen such that 
# their summed responses approximately tile the spatial frequency domain
# between 0.3 c/deg and 10 c/deg.

# 1/23/17 - Edited, like SFMSimpleResp, to allow for making own stimuli

# 1/25/17 - Allowed 'S' to be passed in by checking if unitName is a string
#       or not (if ischar...)
#   Discovery that rComplex = line is QUITE SLOW. Need to speed up...
#  So, decompose that operation into [static] + [dynamic]
#   where static is the part of computation that doesn't change with frame
#   and dynamic does, meaning only the latter needs to computed repeatedly!

# 1/30/17 - Allow for passing in of specific trials from which to draw phase/tf

    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;

    # If unitName is char/string, load the data structure
    # Otherwise, we assume that we already are passing loaded data structure
    if isinstance(unitName, str):
        template = numpy.load(loadPath + unitName + '_sfm.npy')
        S = template.item();
    else:
        S = unitName;
        
    if make_own_stim:
        if not 'template' in stimParams:
            stimParams['template'] = S;
        if not 'repeats' in stimParams:
            stimParams['repeats'] = 10; # why 10? To match experimental #repetitions
    
    T = S['sfm']; # we assume first sfm if there exists more than one
        
    # Get filter properties in spatial frequency domain
    gain = numpy.empty((len(normPool.get('n'))));
    for iB in range(len(normPool.get('n'))):
        prefSf_new = numpy.logspace(numpy.log10(.1), numpy.log10(30), normPool.get('nUnits')[iB]);
        if iB == 0:
            prefSf = prefSf_new;
        else:
            prefSf = [prefSf, prefSf_new];
        gain[iB]   = normPool.get('gain')[iB];
       
    # Get filter properties in direction of motion and temporal frequency domain
    # for prefOr: whatever mode is for stimulus orientation of any component, that's prefOr
    # for prefTf: whatever mean is for stim. TF of any component, that's prefTf
    prefOr = (math.pi/180)*mode(T.get('exp').get('trial').get('ori')[0]).mode;   # in radians
    prefTf = round(numpy.nanmean(T.get('exp').get('trial').get('tf')[0]));       # in cycles per second
    
    # Compute spatial coordinates filter centers (all in phase, 4 filters per period)
    
    stimSi = T.get('exp').get('size'); # in visual degrees
    stimSc = 1.75;                     # in cycles per degree, this is approximately center frequency of stimulus distribution
    nCycle = stimSi*stimSc;
    radius = math.sqrt(pow(math.ceil(4*nCycle), 2)/math.pi);
    vec    = numpy.arange(-math.ceil(radius), math.ceil(radius)+1, 1);
    # hideous python code...fix this when you are learned
    yTemp  = .25/stimSc*repmat(vec, len(vec), 1).transpose();
    xTemp  = .25/stimSc*repmat(vec, len(vec), 1);
    ind    = numpy.sign(stimSi/2 - numpy.sqrt(pow(xTemp,2) + pow(yTemp,2)));
    mask  = numpy.ma.masked_greater(ind, 0).mask; # find the ind > 0; returns object, .mask to get bool array
    xCo = xTemp[mask]; # in visual degrees, centered on stimulus center
    yCo = yTemp[mask]; # in visual degrees, centered on stimulus center
    
    # Store some results in M
    M          = dict();
    pref       = dict();
    pref.setdefault('or', prefOr);
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    M.setdefault('pref', pref);
                               
    # Pre-allocate memory
    z          = T['exp']['trial'];
    nSf        = 0;
    nStimComp  = 5;
    nFrames    = 120;
    
    if not isinstance(prefSf, int):
        for iS in range(len(prefSf)):
            nSf = nSf + len(prefSf[iS]);
    else:
        nSf = 1;
    
    if make_own_stim == 1:
        nTrials = stimParams['repeats']; # keep consistent with 10 repeats per stim. condition
    else:
        nTrials  = len(z.get('num'));

    trial_used = numpy.zeros(nTrials);
        
    M['normResp'] = numpy.zeros((nTrials, nSf, nFrames));
    
    # Compute normalization response for all trials
    if not make_own_stim:
        print('Computing normalization response for ' + unitName + ' ...');

    for p in range(nTrials):
        
        #pdb.set_trace();
        
        #if round(p/156) == p/156: # 156 is from Robbe...Comes from 10 repeats --> 1560(ish) total trials
            #print('\n normalization response computed for {0} of {1} repeats...'.format(1+p/156, round(nTrials/156)));
        
        # Set stim parameters
        if make_own_stim == 1:
            # So... If we want to pass in specific trials, check that those
            # trials exist
            # if there are enough for the trial 'p' we are at now, then
            # grab that one; otherwise get the first
            if 'trial_used' in stimParams:
                if stimParams.get('trial_used') >= p:
                    stimParams['template']['trial_used'] = stimParams.get('trial_used')[p];
                else:
                    stimParams['template']['trial_used'] = stimParams.get('trial_used')[0];
            
            all_stim = makeStimulus(stimParams.get('stimFamily'), stimParams.get('conLevel'), \
                                    stimParams.get('sf_c'), stimParams.get('template'));
            
            stimOr = all_stim.get('Ori');
            stimTf = all_stim.get('Tf');
            stimCo = all_stim.get('Con');
            stimPh = all_stim.get('Ph');
            stimSf = all_stim.get('Sf');
            trial_used[p] = all_stim.get('trial_used');
            
        else:
            stimOr = numpy.empty((nStimComp,));
            stimTf = numpy.empty((nStimComp,));
            stimCo = numpy.empty((nStimComp,));
            stimPh = numpy.empty((nStimComp,));
            stimSf = numpy.empty((nStimComp,));
            for iC in range(nStimComp):
                stimOr[iC] = z.get('ori')[iC][p] * math.pi/180; # in radians
                stimTf[iC] = z.get('tf')[iC][p];          # in cycles per second
                stimCo[iC] = z.get('con')[iC][p];         # in Michelson contrast
                stimPh[iC] = z.get('ph')[iC][p] * math.pi/180;  # in radians
                stimSf[iC] = z.get('sf')[iC][p];          # in cycles per degree

        if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
            continue;
                
        # I. Orientation, spatial frequency and temporal frequency
        # matrix size: 5 x nFilt (i.e., number of stimulus components by number of orientation filters)
          
        # Compute SF tuning
        for iB in range(len(normPool.get('n'))):
            sfRel = repmat(stimSf, len(prefSf[iB]), 1).transpose() / repmat(prefSf[iB], nStimComp, 1);
            s     = pow(repmat(stimSf, len(prefSf[iB]), 1).transpose(), normPool.get('n')[iB]) \
                        * numpy.exp(-normPool.get('n')[iB]/2 * pow(sfRel, 2));
            sMax  = pow(repmat(prefSf[iB], nStimComp, 1), normPool.get('n')[iB]) * numpy.exp(-normPool.get('n')[iB]/2);
            if iB == 0:
                selSf = gain[iB] * s / sMax;
            else:
                selSf = [selSf, gain[iB] * s/sMax];
                
        # Orientation
        selOr = numpy.ones(nStimComp); 
        # all stimulus components of the spatial frequency mixtures were shown at the cell's preferred direction of motion
        
        # Compute temporal frequency tuning
        dOrdTi = 0.25; # derivative order in the temporal domain, d = 0.25 ensures broad tuning for temporal frequency
        tfRel = stimTf / prefTf;
        t     = pow(stimTf, dOrdTi) * numpy.exp(-dOrdTi/2 * pow(tfRel, 2));
        tMax  = pow(prefTf, dOrdTi) * numpy.exp(-dOrdTi/2);
        tNl   = t/tMax;
        selTf = tNl;
    
        # II. Phase, space and time
        omegaX = stimSf * numpy.cos(stimOr); # the stimulus in frequency space
        omegaY = stimSf * numpy.sin(stimOr);
        omegaT = stimTf;

        P = numpy.empty((nFrames*len(xCo), 3)); # nFrames for number of frames, two for x and y coordinate, one for time
        P[:,0] = 2*math.pi*repmat(xCo, 1, nFrames); # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P[:,1] = 2*math.pi*repmat(yCo, 1, nFrames); # P(:,1) and p(:,2) describe location of the filters in space
               
        # Pre-allocate some variables
        respComplex = numpy.zeros((nSf, len(xCo), 120));
        
        selSfVec = numpy.zeros((nStimComp, nSf));
        where = 0;
        for iB in range(len(selSf)):
            selSfVec[:, where:where+normPool.get('nUnits')[iB]] = selSf[iB];
            where = where + normPool.get('nUnits')[iB];
        
        # Modularize computation - Compute the things that are same for all filters (iF)
        for c in range(nStimComp):  # there are up to nine stimulus components

            if stimCo[c] != 0: #if (selSi ~= 0 && stimCo(c) ~= 0)

                # Use the effective number of frames displayed/stimulus duration
                stimPos = numpy.asarray(range(nFrames))/float(nFrames) + \
                                        stimPh[c] / (2*math.pi*stimTf[c]); # 120 frames + the appropriate phase-offset
                P3Temp  = repmat(stimPos, 1, len(xCo));
                P[:,2]  = 2*math.pi*P3Temp; # P(:,2) describes relative location of the filters in time.
            
                omegas = numpy.vstack((omegaX[c], omegaY[c], omegaT[c])); # make this a 3 x len(omegaX) array
                
                rComplex_curr = stimCo[c]*numpy.exp(1j*numpy.dot(P, omegas));
                if c == 0:
                    rComplex_static = rComplex_curr;
                else:
                    rComplex_static = numpy.append(rComplex_static, rComplex_curr, 1);
        
        for iF in range(nSf):
            linR1 = numpy.zeros((nFrames*len(xCo), nStimComp)); # pre-allocation
            linR2 = numpy.zeros((nFrames*len(xCo), nStimComp));
            linR3 = numpy.zeros((nFrames*len(xCo), nStimComp));
            linR4 = numpy.zeros((nFrames*len(xCo), nStimComp));
            computeSum = 0;  # important: if stim contrast or filter sensitivity = zero, no point in computing  response
                    
            # Modularize - Now do the things that are filter-dependent
            for c in range(nStimComp): # there are up to nine stimulus components
                selSi = selOr[c]*selSfVec[c,iF]*selTf[c];    # filter sensitivity for the sinusoid in the frequency domain

                if selSi > 1 or stimCo[c] > 1:
                    pdb.set_trace();

                if selSi != 0 and stimCo[c] != 0:
                    #print('HERE!');
                    #print('SelSi: {0} | stimCo[c]: {1}'.format(selSi, stimCo[c]));
                    computeSum = 1;
                    # now were mostly repeating a simple multiply rather
                    # than exp...
                    rComplex = selSi * rComplex_static[:, c];
        
                    linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
                    linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
                    linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
                    linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);
            
            if computeSum == 1:
                respSimple1 = pow(numpy.maximum(0, linR1.sum(1)), 2); # superposition and half-wave rectification,...
                respSimple2 = pow(numpy.maximum(0, linR2.sum(1)), 2);
                respSimple3 = pow(numpy.maximum(0, linR3.sum(1)), 2);
                respSimple4 = pow(numpy.maximum(0, linR4.sum(1)), 2);
                
                respComplex[iF,:,:] = numpy.reshape(respSimple1 + respSimple2 + respSimple3 + respSimple4, [len(xCo), 120]);
        
                if numpy.count_nonzero(numpy.isnan(respComplex[iF,:,:])) > 0:
                    pdb.set_trace();
                
        # integration over space (compute average response across space, normalize by number of spatial frequency channels)

        respInt = respComplex.mean(1) / len(normPool.get('n'));

        # square root to bring everything in linear contrast scale again
        M['normResp'][p,:,:] = respInt;   

    M.setdefault('trial_used', trial_used);
 
    # if you make/use your own stimuli, just return the output, M;
    # otherwise, save the responses
   
    # THIS NOT GUARANTEED :)
    if not make_own_stim: 
        print('Saving, it seems. In ' + str(loadPath));
        # Save the simulated normalization response in the units structure
        if 'mod' not in S['sfm']:
          S['sfm']['mod'] = dict();
        S['sfm']['mod']['normalization'] = M;
        numpy.save(loadPath + unitName + '_sfm.npy', S)
        
    return M;

def GetNormResp(iU, loadPath, stimParams = []):
   
    
    # GETNORMRESP    Runs the code that computes the response of the
    # normalization pool for the recordings in the SfDiv project.
    # Returns 'M', result from SFMNormResp
    

    # Robbe Goris, 10-30-2015

    M = dict();
    
    # Set characteristics normalization pool
    # The pool includes broad and narrow filters

    # The exponents of the filters used to approximately tile the spatial frequency domain
    n = numpy.array([.75, 1.5]);
    # The number of cells in the broad/narrow pool
    nUnits = numpy.array([12, 15]);
    # The gain of the linear filters in the broad/narrow pool
    gain = numpy.array([.57, .614]);

    normPool = {'n': n, 'nUnits': nUnits, 'gain': gain};
    
    if isinstance(iU, int):
        dataList = numpy.load(loadPath + 'dataList.npy');
        dataList = dataList.item();
        unitName = str(dataList['unitName'][iU]);
        M = SFMNormResp(unitName, loadPath, normPool);
    else:
        unitName = iU;
        M = SFMNormResp(unitName, [], normPool, stimParams);

    return M;

def SFMGiveBof(params, structureSFM, normType):
    # Computes the negative log likelihood for the LN-LN model
    # Returns NLL ###, respModel, E

    # 00 = preferred spatial frequency   (cycles per degree)
    # 01 = derivative order in space
    # 02 = normalization constant        (log10 basis)
    # 03 = response exponent
    # 04 = response scalar
    # 05 = early additive noise
    # 06 = late additive noise
    # 07 = variance of response gain    

    # if fitType == 1
    # currently, no 08; alternatively, 08 = asymmetry (typically bounded [-0.35, 0.35])
    # if fitType == 2
    # 08 = mean of normalization weights gaussian
    # 09 = std of ...
    # if fitType == 3
    # 08 = offset of c50 tuning filter (filter bounded between [sigOffset, 1]
    # 09/10 = standard deviations to the left and right of the peak of the c50 filter
    # 11 = peak (in sf cpd) of c50 filter

    print('ha!');
    
    T = structureSFM['sfm'];

    # Get parameter values
    # Excitatory channel
    pref = {'sf': params[0]};
    dord = {'sp': params[1], 'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'dord': dord};

    # Inhibitory channel
    # nothing in this current iteration - 7/7/17

     # Other (nonlinear) model components
    sigma    = pow(10, params[2]); # normalization constant
    # respExp  = 2; # response exponent
    respExp  = params[3]; # response exponent
    scale    = params[4]; # response scalar

    # Noise parameters
    noiseEarly = params[5];   # early additive noise
    noiseLate  = params[6];  # late additive noise
    varGain    = params[7];  # multiplicative noise

    ### Normalization parameters
    normParams = getNormParams(params, normType);
    if normType == 1: # flat
      inhAsym = normParams;
    elif normType == 2: # gaussian weighting
      gs_mean = normParams[0];
      gs_std  = normParams[1];
    elif normType == 3: # two-halved gaussian for c50
      # sigma calculation
      offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
      stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
      stdRight     = normParams[2]; # '' to the right '' 
      sfPeak       = normParams[3]; # where is the gaussian peak?
    else:
      inhAsym = normParams;

    # Evaluate prior on response exponent -- corresponds loosely to the measurements in Priebe et al. (2004)
    priorExp = lognorm.pdf(respExp, 0.3, 0, numpy.exp(1.15)); # matlab: lognpdf(respExp, 1.15, 0.3);
    NLLExp   = 0; #-numpy.log(priorExp);

    # Compute weights for suppressive signals
    nInhChan = T['mod']['normalization']['pref']['sf'];
    nTrials = len(T['exp']['trial']['num']);
    inhWeight = [];
    nFrames = 120; # always

    if normType == 3: 
      filter = setSigmaFilter(sfPeak, stdLeft, stdRight);
      scale_sigma = -(1-offset_sigma);
      evalSfs = structureSFM['sfm']['exp']['trial']['sf'][0]; # the center SF of all stimuli
      sigmaFilt = evalSigmaFilter(filter, scale_sigma, offset_sigma, evalSfs);
    else:
      sigmaFilt = numpy.square(sigma); # i.e. square the normalization constant

    if normType == 2:
      inhWeightMat = genNormWeights(structureSFM, nInhChan, gs_mean, gs_std, nTrials);
    else: # normType == 1 or anything else,
      for iP in range(len(nInhChan)):
          inhWeight = numpy.append(inhWeight, 1 + inhAsym*(numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                              - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP]))));
      # assumption (made by Robbe) - only two normalization pools
      inhWeightT1 = numpy.reshape(inhWeight, (1, len(inhWeight)));
      inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
      inhWeightT3 = numpy.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
      inhWeightMat  = numpy.tile(inhWeightT3, (1,1,nFrames));

    # Evaluate sfmix experiment
    for iR in range(1): #range(len(structureSFM['sfm'])): # why 1 for now? We don't have S.sfm as array (just one)
        T = structureSFM['sfm']; # [iR]

        # Get simple cell response for excitatory channel
        E = SFMSimpleResp(structureSFM, excChannel);  

        # Extract simple cell response (half-rectified linear filtering)
        Lexc = E['simpleResp'];

        # Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to         bring everything in linear contrast scale again)
        Linh = numpy.sqrt((inhWeightMat*T['mod']['normalization']['normResp']).sum(1)).transpose();

        # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
        numerator     = noiseEarly + Lexc;
        denominator   = pow(sigmaFilt + pow(Linh, 2), 0.5); # square Linh added 7/24 - was mistakenly not fixed earlier
        # denominator   = pow(pow(sigma, 2) + pow(Linh, 2), 0.5); # square Linh added 7/24 - was mistakenly not fixed earlier
        ratio         = pow(numpy.maximum(0, numerator/denominator), respExp);
        meanRate      = ratio.mean(0);
        respModel     = noiseLate + scale*meanRate; # respModel[iR]

        # Get predicted spike count distributions
        mu  = numpy.maximum(.01, T['exp']['trial']['duration']*respModel); # The predicted mean spike count; respModel[iR]
        var = mu + (varGain*pow(mu,2));                        # The corresponding variance of the spike count
        r   = pow(mu,2)/(var - mu);                           # The parameters r and p of the negative binomial distribution
        p   = r/(r + mu);

        # Evaluate the model
        lsq = numpy.square(numpy.sqrt(respModel) - numpy.sqrt(T['exp']['trial']['spikeCount']));
        NLL = numpy.mean(lsq); # was 1*lsq
        #llh = nbinom.pmf(T['exp']['trial']['spikeCount'], r, p); # Likelihood for each pass under doubly stochastic model
        #NLLtempSFM = numpy.mean(-numpy.log(llh)); # The negative log-likelihood of the whole data-set; [iR]

    # Combine data and prior
    #NLL = NLLtempSFM + NLLExp; # sum over NLLtempSFM if you allow it to be d>1

    return NLL, respModel;
    #return {'NLL': NLL, 'respModel': respModel, 'Exc': E};

def SFMsimulate(params, structureSFM, stimFamily, con, sf_c, unweighted = 0, normType=1):
    # Currently, will get slightly different stimuli for excitatory and inhibitory/normalization pools
    # But differences are just in phase/TF, but for TF, drawn from same distribution, anyway...
    # 4/27/18: if unweighted = 1, then do the calculation/return normResp with weights applied; otherwise, just return the unweighted filter responses

    # 00 = preferred spatial frequency   (cycles per degree)
    # 01 = derivative order in space
    # 02 = normalization constant        (log10 basis)
    # 03 = response exponent
    # 04 = response scalar
    # 05 = early additive noise
    # 06 = late additive noise
    # 07 = variance of response gain    
    # 08 = inhibitory asymmetry (i.e. tilt of gain over SF for weighting normalization pool responses)
    # OR
    # 08/09 = mean/std of gaussian used for weighting normalization filters
    # OR
    # 08 = offset in c50 filter (bounded b/t [offset, 1])
    # 09/10 = std to the left/right of the peak of the c50 filter

    #print('simulate!');

    T = structureSFM['sfm'];

    # Get parameter values
    # Excitatory channel
    pref = {'sf': params[0]};
    dord = {'sp': params[1], 'ti': 0.25}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'dord': dord};

    # Inhibitory channel

    # Other (nonlinear)  components
    sigma    = pow(10, params[2]); # normalization constant
    # respExp  = 2; # response exponent
    respExp  = params[3]; # response exponent
    scale    = params[4]; # response scalar

    # Noise parameters
    noiseEarly = params[5];   # early additive noise
    noiseLate  = params[6];  # late additive noise
    varGain    = params[7];  # multiplicative noise

    ### Normalization parameters
    normParams = getNormParams(params, normType);
    if normType == 1:
      inhAsym = normParams[0];
    elif normType == 2:
      gs_mean = normParams[0];
      gs_std  = normParams[1];
    elif normType == 3:
      # sigma calculation
      offset_sigma = normParams[0];  # c50 filter will range between [v_sigOffset, 1]
      stdLeft      = normParams[1];  # std of the gaussian to the left of the peak
      stdRight     = normParams[2]; # '' to the right '' 
      sfPeak       = normParams[3]; # where is the gaussian peak?
    else:
      inhAsym = normParams[0];
    
    # Get stimulus structure ready...
    stimParams = dict();
    stimParams['stimFamily'] = stimFamily;
    stimParams['conLevel'] = con;
    stimParams['sf_c'] = sf_c;
    stimParams['repeats'] = 1; # defaults to 10 anyway, in makeStimulus.py
    
    # Compute weights for suppressive signals
    nInhChan = T['mod']['normalization']['pref']['sf'];
    nTrials = stimParams['repeats'];
    inhWeight = [];
    nFrames = 120; # always

    if normType == 3:
      filter = setSigmaFilter(filterPeak, stdLeft, stdRight);
      scale_sigma = -(1-offset_sigma);
      evalSfs = structureSFM['sfm']['exp']['trial']['sf'][0]; # the center SF of all stimuli
      sigmaFilt = evalSigmaFilter(filter, scale_sigma, offset_sigma, evalSfs);
    else:
      sigmaFilt = numpy.square(sigma); # i.e. normalization constant squared

    if normType == 2:
      inhWeightMat = genNormWeights(structureSFM, nInhChan, gs_mean, gs_std, nTrials);
    else: # normType == 1 or anything else, we just go with 
      for iP in range(len(nInhChan)):
          inhWeight = numpy.append(inhWeight, 1 + inhAsym*(numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                              - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP]))));
      # assumption (made by Robbe) - only two normalization pools
      inhWeightT1 = numpy.reshape(inhWeight, (1, len(inhWeight)));
      inhWeightT2 = repmat(inhWeightT1, nTrials, 1);
      inhWeightT3 = numpy.reshape(inhWeightT2, (nTrials, len(inhWeight), 1));
      inhWeightMat  = numpy.tile(inhWeightT3, (1,1,nFrames));

    # Evaluate sfmix experiment
    T = structureSFM['sfm']; # [iR]
    
    #pdb.set_trace();
    # Get simple cell response for excitatory channel
    E = SFMSimpleResp(structureSFM, excChannel, stimParams);  

    # Extract simple cell response (half-rectified linear filtering)
    Lexc = E['simpleResp'];

    # Get inhibitory response (pooled responses of complex cells tuned to wide range of spatial frequencies, square root to bring everything in linear contrast scale again)
    normResp = GetNormResp(structureSFM, [], stimParams);
    if unweighted == 1:
      return [], [], Lexc, normResp['normResp'], [];
    Linh = numpy.sqrt((inhWeightMat*normResp['normResp']).sum(1)).transpose();

    # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
    numerator     = noiseEarly + Lexc;
    # taking square root of denominator (after summing squares...) to bring in line with computation in Carandini, Heeger, Movshon, '97
    denominator   = pow(sigmaFilt + pow(Linh, 2), 0.5); # squaring Linh - edit 7/17
    ratio         = pow(numpy.maximum(0, numerator/denominator), respExp);
    meanRate      = ratio.mean(0);
    respModel     = noiseLate + scale*meanRate; # respModel[iR]

    return respModel, Linh, Lexc, normResp['normResp'], denominator;
