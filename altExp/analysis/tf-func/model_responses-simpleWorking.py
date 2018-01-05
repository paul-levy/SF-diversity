import math, cmath, numpy, os
from makeStimulus import makeStimulus 
from scipy.stats import norm, mode, lognorm, nbinom
from numpy.matlib import repmat
import time

import tensorflow as tf

import pdb

fft = numpy.fft
tf_pi = tf.constant(math.pi);

def ph_test(S, p):
    
    pdb.set_trace();
    
    z = S['sfm']['exp']['trial'];
    
    stimOr = numpy.empty((nGratings,));
    stimTf = numpy.empty((nGratings,));
    stimCo = numpy.empty((nGratings,));
    stimPh = numpy.empty((nGratings,));
    stimSf = numpy.empty((nGratings,));
               
    for iC in range(9):
        stimOr[iC] = z.get('ori')[iC][p] * math.pi/180; # in radians
        stimTf[iC] = z.get('tf')[iC][p];          # in cycles per second
        stimCo[iC] = z.get('con')[iC][p];         # in Michelson contrast
        stimPh[iC] = z.get('ph')[iC][p] * math.pi/180;  # in radians
        stimSf[iC] = z.get('sf')[iC][p];          # in cycles per degree
                
    return StimOr, stimTf, stimCo, stimPh, stimSf;

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
def SFMSimpleResp(ph_stimOr, ph_stimTf, ph_stimCo, ph_stimSf, ph_stimPh, nTrials, channel, stimParams = []):
    # returns object (class?) with simpleResp and other things

    # SFMSimpleResp       Computes response of simple cell for sfmix experiment

    # SFMSimpleResp(varargin) returns a simple cell response for the
    # mixture stimuli used in sfMix. The cell's receptive field is the n-th
    # derivative of a 2-D Gaussian that need not be circularly symmetric.

    # 1/23/17 - Edits: Added stimParamsm, make_own_stim so that I can set what
    # stimuli I want when simulating from model

    make_own_stim = 0;
    if stimParams: # i.e. if we actually have non-empty stimParams
        make_own_stim = 1;
        if not stimParams.haskey('template'):
            stimParams.setdefault('template', S);
        if not stimParams.haskey('repeats'):
            stimParams.setdefault('repeats', 10); # why 10? To match experimental

    # Get preferred stimulus values
    prefOr = tf_pi/180 * channel.get('pref').get('or');                # in radians
    prefSf = channel.get('pref').get('sf');                              # in cycles per degree
    # CHECK LINE BELOW  
    prefTf = tf.round(tf.reduce_mean(ph_stimTf[0]));     # in cycles per second
    #prefTf = tf.constant(prefTf);

    # Get directional selectivity
    ds = channel.get('ds');

    # Get derivative order in space and time
    dOrdSp = channel.get('dord').get('sp');
    dOrdTi = channel.get('dord').get('ti');

    # Get aspect ratio in space
    aRatSp = channel.get('arat').get('sp');

    # Get spatial coordinates
    xCo = 0;                                                              # in visual degrees, centered on stimulus center
    yCo = 0;                                                              # in visual degrees, centered on stimulus center

    # Store some results in M
    M = dict();
    pref = dict();
    arat = dict();
    dord = dict();
    pref.setdefault('sf', prefSf);
    pref.setdefault('tf', prefTf);
    pref.setdefault('xCo', xCo);
    pref.setdefault('yCo', yCo);
    arat.setdefault('sp', aRatSp);
    dord.setdefault('sp', dOrdSp);
    dord.setdefault('ti', dOrdTi);
    
    M.setdefault('pref', pref);
    M.setdefault('arat', arat);
    M.setdefault('dord', dord);
    M.setdefault('ds', ds);
    
    # Pre-allocate memory
    nTrials       = nTrials;
    nSf           = 1;
    nGratings     = 9;
    nFrames       = 120;
    if make_own_stim == 1:
        nTrials = stimParams.get('repeats'); # to keep consistent with number of repetitions used for each stim. condition
    else: # CHECK THIS GUY BELOW
        nTrials = nTrials;
    
    # set it zero
    M['simpleResp'] = tf.zeros((nFrames, nTrials));
    
    # Set stim parameters
    if make_own_stim == 1:
        to_rad = 1;
    else:
        to_rad = tf_pi/180;
            
    stimOr = tf.zeros((nGratings,));
    stimTf = tf.zeros((nGratings,));
    stimCo = tf.zeros((nGratings,));
    stimPh = tf.zeros((nGratings,));
    stimSf = tf.zeros((nGratings,));
            
    #pdb.set_trace();
            
    stimOr = tf.multiply(ph_stimOr, to_rad);  # in radians
    stimTf = ph_stimTf;                       # in cycles per second
    stimCo = ph_stimCo;                       # in Michelson contrast
    stimPh = tf.multiply(ph_stimPh, to_rad);  # in radians
    stimSf = ph_stimSf;                       # in cycles per degree
        
    # Compute simple cell response for all trials
    for p in range(nTrials): 
        
        print('Trial number ' + str(p));
    
        '''
        # Set stim parameters
        if make_own_stim == 1:
            to_rad = 1;
        else:
            to_rad = tf_pi/180;
            
        stimOr = tf.zeros((nGratings,));
        stimTf = tf.zeros((nGratings,));
        stimCo = tf.zeros((nGratings,));
        stimPh = tf.zeros((nGratings,));
        stimSf = tf.zeros((nGratings,));
            
        pdb.set_trace();
            
        for iC in range(9):
            stimOr[iC] = tf.multiply(ph_stimOr[iC][p], to_rad);  # in radians
            stimTf[iC] = ph_stimTf[iC][p];                       # in cycles per second
            stimCo[iC] = ph_stimCo[iC][p];                       # in Michelson contrast
            stimPh[iC] = tf.multiply(ph_stimPh[iC][p], to_rad);  # in radians
            stimSf[iC] = ph_stimSf[iC][p];                       # in cycles per degree
        '''

        if tf.is_nan(stimOr) is None:
            continue;
        #if tf.count_nonzero(tf.is_nan(stimOr)): # then this is a blank stimulus, no computation to be done
        #    continue;
                
        #pdb.set_trace();
                
        # I. Orientation, spatial frequency and temporal frequency
        diffOr = tf.subtract(prefOr, tf.to_float(stimOr)); 
        # matrix size: 9 x nFilt (i.e., number of stimulus components by number of orientation filters)
        o      = tf.pow(tf.square(tf.cos(diffOr)) * \
                     tf.exp(tf.square(aRatSp)-1 * tf.square(tf.cos(diffOr))), dOrdSp/2);
        oMax   = tf.pow(tf.exp(tf.square(aRatSp) -1), dOrdSp/2);
        oNl    = tf.divide(o, oMax);
        e      = 1 + (ds*.5*(-1+(tf.sign(diffOr + tf_pi/2))));
        selOr  = tf.multiply(oNl, e);

        if channel.get('dord').get('sp') == 0:
            selOr[:] = 1;

        # Compute spatial frequency tuning
        sfRel = tf.divide(stimSf, prefSf);
        s     = tf.multiply(tf.pow(stimSf, dOrdSp), tf.exp(-dOrdSp/2 * tf.square(sfRel)));
        sMax  = tf.multiply(tf.pow(prefSf, dOrdSp), tf.exp(-dOrdSp/2));
        sNl   = tf.divide(s, sMax);
        selSf = sNl;

        # Compute temporal frequency tuning
        tfRel = tf.divide(stimTf, prefTf);
        t     = tf.multiply(tf.pow(stimTf, dOrdTi), tf.exp(-dOrdTi/2 * tf.square(tfRel)));
        tMax  = tf.multiply(tf.pow(prefTf, dOrdTi), tf.exp(-dOrdTi/2));
        tNl   = tf.divide(t, tMax);
        selTf = tNl;

        # II. Phase, space and time
        omegaX = tf.multiply(stimSf, tf.cos(stimOr)); # the stimulus in frequency space
        omegaY = tf.multiply(stimSf, tf.sin(stimOr));
        omegaT = stimTf;

        P_x = 2*tf_pi*xCo*tf.ones(nFrames,);  # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P_y = 2*tf_pi*yCo*tf.ones(nFrames,); # P(:,0) and p(:,1) describe location of the filters in space

        # Pre-allocate some variables
        if nSf == 1:
            respSimple = tf.zeros(nFrames,);
        else:
            respSimple = tf.zeros(nFrames, nSf);

        for iF in range(nSf):
            if isinstance(xCo, int):
                factor = 1;
            else:
                factor = len(xCo);

            linR1 = tf.constant(-1);
            linR2 = tf.constant(-1);
            linR3 = tf.constant(-1);
            linR4 = tf.constant(-1);
            #linR1 = tf.zeros((nFrames*factor, nGratings)); # pre-allocation
            #linR2 = tf.zeros((nFrames*factor, nGratings));
            #linR3 = tf.zeros((nFrames*factor, nGratings));
            #linR4 = tf.zeros((nFrames*factor, nGratings));
            
            computeSum = 0; # important constant: if stimulus contrast or filter sensitivity equals zero there is no point in computing the response

            #pdb.set_trace();
            
            for c in range(nGratings): # there are up to nine stimulus components
                
                #print('\tGrating number ' + str(c));
                
                selSi = selOr[c]*selSf[c]*selTf[c]; # filter sensitivity for the sinusoid in the frequency domain

                if selSi != 0 and stimCo[c] != 0:
                    computeSum = 1;
                        
                    # Use the effective number of frames displayed/stimulus duration
                    stimPos = tf.range(nFrames, dtype=tf.float32)/nFrames + stimPh[c] / (2*tf_pi*stimTf[c]); 
                    # ABOVE: 120 frames + appropriate phase-offset
                    PtTemp  = tf.multiply(tf.ones(nFrames), stimPos);
                    #P3Temp  = repmat(stimPos, 1, len(xCo));
                    P_t  = 2*tf_pi*PtTemp; # P_t describes relative location of the filters in time.
                    
                    omegas = tf.stack([omegaX[c], omegaY[c], omegaT[c]]); # make this a 3 x len(omegaX) array
                    Ps = tf.transpose(tf.stack([P_x, P_y, P_t]));
                    intermediate = tf.tensordot(Ps, omegas, 1); # why save intermediate? must convert in next step - cleaner
                    # ABOVE: Is tensordot OK??? Check dis
                    rComplex_part = tf.exp(tf.multiply(1j, tf.cast(intermediate, tf.complex128)));
                    rComplex = tf.cast(selSi*stimCo[c], tf.complex128) * rComplex_part;

                    #pdb.set_trace();
                    
                    # FIX THIS - figure out how to concatenate for each component, extract proper real/imag with sign
                    linR1 = tf.cond(linR1<0, lambda: tf.real(rComplex), \
                                    lambda: tf.stack([tf.cast(linR1, dtype=tf.float64), tf.real(rComplex)], axis = 0));
                    
                    linR2 = tf.cond(linR2<0, lambda: -1*tf.real(rComplex), \
                                    lambda: tf.stack([tf.cast(linR2, dtype=tf.float64), -1*tf.real(rComplex)], axis = 0));

                    linR3 = tf.cond(linR3<0, lambda: tf.imag(rComplex), \
                                    lambda: tf.stack([tf.cast(linR3, dtype=tf.float64), tf.imag(rComplex)], axis = 0));
                    
                    linR4 = tf.cond(linR4<0, lambda: -1*tf.imag(rComplex), \
                                    lambda: tf.stack([tf.cast(linR4, dtype=tf.float64), -1*tf.imag(rComplex)], axis = 0));
                    
                    '''
                    if linR1 is None:
                        linR1 = tf.real(rComplex);
                    else:
                        linR1 = tf.concat([linR1, tf.real(rComplex)], axis = 0);
                        
                    if linR2 is None:
                        linR2 = -1*tf.real(rComplex);
                    else:
                        linR2 = tf.concat([linR2, -1*tf.real(rComplex)], axis = 0);

                    if linR3 is None:
                        linR3 = tf.imag(rComplex);
                    else:
                        linR3 = tf.concat([linR3, tf.imag(rComplex)], axis = 0);
                        
                    if linR4 is None:
                        linR4 = -1*tf.imag(rComplex);
                    else:
                        linR4 = tf.concat([linR4, -1*tf.imag(rComplex)], axis = 0);
                    '''
                    '''
                    linR1[:,c] = rComplex.real.reshape(linR1[:,c].shape);  # four filters placed in quadrature
                    linR2[:,c] = -1*rComplex.real.reshape(linR2[:,c].shape);
                    linR3[:,c] = rComplex.imag.reshape(linR3[:,c].shape);
                    linR4[:,c] = -1*rComplex.imag.reshape(linR4[:,c].shape);
                    '''
                if computeSum == 1:
                    compZero = tf.constant(0, dtype=linR1.dtype);
                    # superposition and half-wave rectification,...
                    respSimple1 = tf.maximum(compZero, tf.reduce_sum(linR1, 1)); 
                    respSimple2 = tf.maximum(compZero, tf.reduce_sum(linR2, 1));
                    respSimple3 = tf.maximum(compZero, tf.reduce_sum(linR3, 1));
                    respSimple4 = tf.maximum(compZero, tf.reduce_sum(linR4, 1));

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
                        
        # Store response in desired format
        #M['simpleResp'][:,p] = respSimple;
        
    return respSimple;
        
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
        if not stimParams.haskey('template'): 
            stimParams.setdefault('template', S);
        if not stimParams.haskey('repeats'):
            stimParams.setdefault('repeats', 10); # why 10? To match experimental
    
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
    prefOr = (tf_pi/180)*mode(T.get('exp').get('trial').get('ori')[0]).mode;   # in radians
    prefTf = round(numpy.nanmean(T.get('exp').get('trial').get('tf')[0]));       # in cycles per second
    
    # Compute spatial coordinates filter centers (all in phase, 4 filters per period)
    
    stimSi = T.get('exp').get('size'); # in visual degrees
    stimSc = 1.75;                     # in cycles per degree, this is approximately center frequency of stimulus distribution
    nCycle = stimSi*stimSc;
    radius = math.sqrt(pow(math.ceil(4*nCycle), 2)/tf_pi);
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
    nStimComp  = 9;
    nFrames    = 120;
    
    if not isinstance(prefSf, int):
        for iS in range(len(prefSf)):
            nSf = nSf + len(prefSf[iS]);
    else:
        nSf = 1;
    
    if make_own_stim == 1:
        nTrials = stimParams.repeats; # keep consistent with 10 repeats per stim. condition
    else:
        nTrials  = len(z.get('num'));

    trial_used = numpy.zeros(nTrials);
        
    M['normResp'] = numpy.zeros((nTrials, nSf, nFrames));
    
    # Compute normalization response for all trials
    if not make_own_stim:
        print('Computing normalization response for ' + unitName + ' ...');
        
    # Get all stimulus stuff right
    stimOr = numpy.empty((nStimComp,nTrials,));
    stimTf = numpy.empty((nStimComp,nTrials,));
    stimCo = numpy.empty((nStimComp,nTrials,));
    stimPh = numpy.empty((nStimComp,));
    stimSf = numpy.empty((nStimComp,));
    
    stimOr[iC] = z.get('ori') * tf_pi/180; # in radians
    stimTf[iC] = z.get('tf');          # in cycles per second
    stimCo[iC] = z.get('con');         # in Michelson contrast
    stimPh[iC] = z.get('ph') * tf_pi/180;  # in radians
    stimSf[iC] = z.get('sf');          # in cycles per degree
        
    for p in range(nTrials):
        
        #pdb.set_trace();
        
        if round(p/156) == p/156: # 156 is from Robbe...Comes from 10 repeats --> 1560(ish) total trials
            print('\n normalization response computed for {0} of {1} repeats...'.format(1+p/156, round(nTrials/156)));
        
        # Set stim parameters
        if make_own_stim == 1:
            # So... If we want to pass in specific trials, check that those
            # trials exist
            # if there are enough for the trial 'p' we are at now, then
            # grab that one; otherwise get the first
            if stimParams.haskey('trial_used'):
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
            print('Hello!');
            '''
            stimOr = numpy.empty((nStimComp,));
            stimTf = numpy.empty((nStimComp,));
            stimCo = numpy.empty((nStimComp,));
            stimPh = numpy.empty((nStimComp,));
            stimSf = numpy.empty((nStimComp,));
            for iC in range(nStimComp):
                stimOr[iC] = z.get('ori')[iC][p] * tf_pi/180; # in radians
                stimTf[iC] = z.get('tf')[iC][p];          # in cycles per second
                stimCo[iC] = z.get('con')[iC][p];         # in Michelson contrast
                stimPh[iC] = z.get('ph')[iC][p] * tf_pi/180;  # in radians
                stimSf[iC] = z.get('sf')[iC][p];          # in cycles per degree
            '''

        if numpy.count_nonzero(numpy.isnan(stimOr)): # then this is a blank stimulus, no computation to be done
            continue;
                
        # I. Orientation, spatial frequency and temporal frequency
        # matrix size: 9 x nFilt (i.e., number of stimulus components by number of orientation filters)
          
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
        P[:,0] = 2*tf_pi*repmat(xCo, 1, nFrames); # P is the matrix that contains the relative location of each filter in space-time (expressed in radians)
        P[:,1] = 2*tf_pi*repmat(yCo, 1, nFrames); # P(:,1) and p(:,2) describe location of the filters in space
               
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
                stimPos = numpy.asarray(range(nFrames))/nFrames + \
                                        stimPh[c] / (2*tf_pi*stimTf[c]); # 120 frames + the appropriate phase-offset
                P3Temp  = repmat(stimPos, 1, len(xCo));
                P[:,2]  = 2*tf_pi*P3Temp; # P(:,2) describes relative location of the filters in time.
            
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
        
        #pdb.set_trace();
        
        respInt = respComplex.mean(1) / len(normPool.get('n'));

        # square root to bring everything in linear contrast scale again
        M['normResp'][p,:,:] = respInt;   

    M.setdefault('trial_used', trial_used);
    
    # if you make/use your own stimuli, just return the output, M;
    # otherwise, save the responses
   
    # THIS NOT GUARANTEED :)
    if not make_own_stim:
        # Save the simulated normalization response in the units structure
        S['sfm']['mod']['normalization'] = M;
        numpy.save(loadPath + unitName + '_sfm.npy', S)
        
    return M;

def GetNormResp(iU, stimParams = []):
   
    
    # GETNORMRESP    Runs the code that computes the response of the
    # normalization pool for the recordings in the SfDiv project.
    # Returns 'M', result from SFMNormResp
    

    # Robbe Goris, 10-30-2015

    # Edit - Paul Levy, 1/23/17 to give option 2nd parameter for passing in own
    # stimuli (see SFMNormResp for more details)
    # 1/25/17 - Allowed 'S' to be passed in by checking if unitName is numeric
    # or not (if isnumeric...)
    
    M = dict();
    
    # Set paths
    base = '/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/'; # CNS
    currentPath  = base + 'Analysis/Scripts/';
    loadPath     = base + 'Analysis/Structures/';
    functionPath = base + 'Analysis/Functions/';
    
    # Set characteristics normalization pool
    # The pool includes broad and narrow filters

    # The exponents of the filters used to approximately tile the spatial frequency domain
    n = numpy.array([.75, 1.5]);
    # The number of cells in the broad/narrow pool
    nUnits = numpy.array([12, 15]);
    # The gain of the linear filters in the broad/narrow pool
    gain = numpy.array([.57, .614]);

    normPool = {'n': n, 'nUnits': nUnits, 'gain': gain};
    
    dataList = numpy.load(loadPath + 'dataList.npy');
    dataList = dataList.item();
    
    if isinstance(iU, int):
        unitName = dataList['unitName'][iU];
    else:
        unitName = iU;
    
    M = SFMNormResp(unitName, loadPath, normPool, stimParams);

    return M;

def SFMGiveBof(params, structureSFM):
    # Computes the negative log likelihood for the LN-LN model
    # Returns NLL, respModel, E

    # 01 = preferred direction of motion (degrees)
    # 02 = preferred spatial frequency   (cycles per degree)
    # 03 = aspect ratio 2-D Gaussian
    # 04 = derivative order in space
    # 05 = directional selectivity
    # 06 = gain inhibitory channel
    # 07 = normalization constant        (log10 basis)
    # 08 = response exponent
    # 09 = response scalar
    # 10 = early additive noise
    # 11 = late additive noise
    # 12 = variance of response gain    
    # 13 = asymmetry suppressive signal 

    T = structureSFM['sfm'];

    # Get parameter values
    # Excitatory channel
    pref = {'or': params[0], 'sf': params[1]};
    arat = {'sp': params[2]};
    dord = {'sp': params[3], 'ti': tf.constant(0.25)}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'arat': arat, 'dord': dord, 'ds': params[4]};

    # Inhibitory channel
    inhChannel = {'gain': params[5], 'asym': params[12]};

     # Other (nonlinear) model components
    sigma    = tf.pow(10, params[6]); # normalization constant
    # respExp  = 2; # response exponent
    respExp  = params[7]; # response exponent
    scale    = params[8]; # response scalar

    # Noise parameters
    noiseEarly = params[9];   # early additive noise
    noiseLate  = params[10];  # late additive noise
    varGain    = params[11];  # multiplicative noise

    # Evaluate prior on response exponent -- corresponds loosely to the measurements in Priebe et al. (2004)
    priorExp = lognorm.pdf(respExp, 0.3, 0, numpy.exp(1.15)); # matlab: lognpdf(respExp, 1.15, 0.3);
    NLLExp   = -numpy.log(priorExp) / len(T['exp']['trial']['spikeCount']);
    # why divide by number of trials? because we take mean of NLLs, so this way it's fair to add them

    # Compute weights for suppressive signals
    nInhChan = T['mod']['normalization']['pref']['sf'];
    nTrials = len(T['exp']['trial']['num']);
    inhWeight = [];
    nFrames = 120; # always
    for iP in range(len(nInhChan)):
        inhWeight = numpy.append(inhWeight, 1 + \
                                 tf.multiply(inhChannel['asym'], (numpy.log(T['mod']['normalization']['pref']['sf'][iP]) \
                                            - numpy.mean(numpy.log(T['mod']['normalization']['pref']['sf'][iP])))));

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
        temp = tf.multiply(inhWeightMat, tf.reduce_sum(T['mod']['normalization']['normResp'], 1));
        Linh = tf.transpose(tf.sqrt(temp));

        # Compute full model response (the normalization signal is the same as the subtractive suppressive signal)
        numerator     = tf.add(tf.add(noiseEarly, Lexc), tf.multiply(inhChannel['gain'], Linh));
        denominator   = tf.add(tf.square(sigma), Linh);
        ratio         = tf.pow(tf.maximum(0, tf.divide(numerator,denominator)), respExp);
        meanRate      = tf.reduce_mean(ratio);
        respModel     = tf.add(noiseLate, tf.multiply(scale, meanRate)); # respModel[iR]

        # Get predicted spike count distributions
        mu  = tf.maximum(.01, tf.multiply(T['exp']['trial']['duration'], respModel)); 
                         # The predicted mean spike count; respModel[iR]
        var = tf.add(mu, tf.multiply(varGain, tf.square(mu))); # The corresponding variance of the spike count
        r   = tf.divide(tf.square(mu), tf.subtract(var, mu)); # The parameters r and p of the negative binomial distribution
        p   = tf.divide(r, tf.add(r, mu));

        # Evaluate the model
        llh = nbinom.pmf(T['exp']['trial']['spikeCount'], r, p); # Likelihood for each pass under doubly stochastic model
        NLLtempSFM = numpy.sum(-numpy.log(llh)); # The negative log-likelihood of the whole data-set; [iR]      
        # take sum of above...
        
        #r_teef = tf.placeholder(tf.float32);
        #p_teef = tf.placeholder(tf.float32);
        data_ph = tf.placeholder(tf.float32);
        log_lh = negBinom(data_ph, r, p);
        #log_lh = negBinom(T['exp']['trial']['spikeCount'], r_teef, p_teef);
        
        sess = tf.Session();
        llh_tea = sess.run(log_lh, {data_ph: T['exp']['trial']['spikeCount']}); #r, p_teef: p});
        

    # Combine data and prior
    NLL_numpy = NLLtempSFM + NLLExp; # sum over NLLtempSFM if you allow it to be d>1
    NLL = tf.add(tf.reduce_mean(-llh_tea), NLLExp);
    
    return {'NLL': NLL, 'NLL_numpy': NLL_numpy, 'respModel': respModel, 'Exc': E};

def negBinom(x, r, p):
    # We assume that r & p are tf placeholders/variables; x is a constant
    # Negative binomial is:
        # gamma(x+r) * (1-p)^x * p^r / (gamma(x+1) * gamma(r))
    
    # Here we return the log negBinomial:

    pdb.set_trace();
    
    x = tf.to_float(x); # convert to float32just like r, p
    naGam = tf.multiply(x, tf.log(1-p)) + tf.multiply(r, tf.log(p));
    haanGam = tf.lgamma(tf.add(x, r)) - tf.lgamma(tf.add(x, 1)) - tf.lgamma(r);
    
    return tf.add(naGam, haanGam);
    
    
def runModel(modParams, ph_stimOr, ph_stimTf, ph_stimCo, ph_stimSf, ph_stimPh, ph_spikeCount):
    # Does the work after setting up the model - this will call SFMGiveBof...
    
    
    res = SFMSimpleResp(ph_stimOr, ph_stimTf, ph_stimCo, ph_stimSf, ph_stimPh, channel);
    
def setModel(cellNum):
    # Given just a cell number, will fit the Robbe V1 model to the data
    
    ########
    # Set up model parameters - i.e. trainable variables!
    ########
    
    # 00 = preferred direction of motion (degrees)
    # 01 = preferred spatial frequency   (cycles per degree)
    # 02 = aspect ratio 2-D Gaussian
    # 03 = derivative order in space
    # 04 = directional selectivity
    # 05 = gain inhibitory channel
    # 06 = normalization constant        (log10 basis)
    # 07 = response exponent
    # 08 = response scalar
    # 09 = early additive noise
    # 10 = late additive noise
    # 11 = variance of response gain    
    # 12 = asymmetry suppressive signal
    n_params = 13;
    mod_params  = tf.Variable(tf.zeros(n_params));
    '''
    mod_ori     = tf.Variable(tf.zeros(1)); # default is trainable (i.e. make model differentiable w.r.t.)
    mod_psf     = tf.Variable(tf.zeros(1));
    mod_ar      = tf.Variable(tf.zeros(1));
    mod_do      = tf.Variable(tf.zeros(1));
    mod_ds      = tf.Variable(tf.zeros(1));
    mod_inhGain = tf.Variable(tf.zeros(1));
    mod_normSig = tf.Variable(tf.zeros(1));
    mod_respExp = tf.Variable(tf.zeros(1));
    mod_respScl = tf.Variable(tf.zeros(1));
    mod_noiseEr = tf.Variable(tf.zeros(1));
    mod_noiseLt = tf.Variable(tf.zeros(1));
    mod_respVar = tf.Variable(tf.zeros(1));
    mod_inhAsym = tf.Variable(tf.zeros(1));
    '''
    
    #########
    # Organize paramters for passing into functions
    #########
    # Excitatory channel
    pref = {'or': mod_params[0], 'sf': mod_params[1]};
    arat = {'sp': mod_params[2]};
    dord = {'sp': mod_params[3], 'ti': tf.constant(0.25)}; # deriv order in temporal domain = 0.25 ensures broad tuning for temporal frequency
    excChannel = {'pref': pref, 'arat': arat, 'dord': dord, 'ds': mod_params[4]};
    
    #########
    # Now get all the data we need for tf_placeholders 
    #########
    loc_data = '/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/';
    #loc_data = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/Analysis/Structures/';
    
    dataList = numpy.load(loc_data + 'dataList.npy').item();
    dataNames = dataList['unitName'];
    S = numpy.load(loc_data + dataNames[cellNum] + '_sfm.npy').item();
    
    # stimulus information
    trial_inf = S['sfm']['exp']['trial'];
    
    stimOr = trial_inf['ori'];
    stimTf = trial_inf['tf'];
    stimCo = trial_inf['con'];
    stimSf = trial_inf['sf'];
    stimPh = trial_inf['ph'];
    #nTrials = len(trial_inf['num']);
    
    # cell responses
    spikes = trial_inf['spikeCount'];
    stim_dur = trial_inf['duration'];
    
    #########
    # Set up the call
    #########
    ph_stimOr = tf.placeholder(tf.float32);
    ph_stimTf = tf.placeholder(tf.float32, stimTf.shape);
    ph_stimCo = tf.placeholder(tf.float32);
    ph_stimSf = tf.placeholder(tf.float32);
    ph_stimPh = tf.placeholder(tf.float32);
    #ph_nTrials = tf.placeholder(tf.float32);
    
    #pdb.set_trace();
    
    okok = SFMSimpleResp(ph_stimOr, ph_stimTf, ph_stimCo, ph_stimSf, ph_stimPh, excChannel);
    
    m = tf.Session();
    #tf.get_collection();
    m.run(tf.global_variables_initializer());
    foh_real = m.run(hurr, feed_dict={ph_stimOr: stimOr, ph_stimTf: stimTf, ph_stimCo: stimCo, \
                            ph_stimSf: stimSf, ph_stimPh: stimPh});
    
    