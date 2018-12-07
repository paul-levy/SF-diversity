import numpy, scipy, math
from scipy.stats import mode
from scipy.signal import sawtooth
from numpy.matlib import repmat
from numpy.random import random
from os import chdir as cd
from model_responses import SFMGiveBof

from scipy.optimize import minimize

import pdb

def fitRealData(iU, linear = 0):

# fitRealData    Fits an LN-LN model to cell responses elicited by gratings
# and mixture stimuli. 
# The fitting algorithm is !!!. 
# The model is fit with a multistart procedure and semi-randomized starting 
# values. All texpat and spatial frequency data are fitted

    # Set paths - pick one
    base = '/e/3.2/p1/plevy/SF_diversity/sfDiv-OriModel/'; # CNS
    # base = '/home/pl1465/modelCluster/'; # cluster
    # base = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/'; # local
   
    currentPath  = base + 'sfDiv-python/Analysis/Functions';
    loadPath     = base + 'sfDiv-python/Analysis/Structures';
    functionPath = base + 'sfDiv-python/Analysis/Functions';

    cd(currentPath);
    
    # Set constants
    nMultiStarts = 10;
    bestNLL = [];

    # Load data files
    cd(loadPath)
    dataList = numpy.load('dataList.npy').item();    

    # Loads S
    loadNameSfm = dataList['unitName'][iU] + '_sfm.npy';
    S = numpy.load(loadNameSfm).item();
    cd(currentPath)

    ## Constraints are set in the form of lower and upper parameter bounds
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

    lowerBound = numpy.zeros(13);   upperBound     = numpy.zeros(13);    
    lowerBound[0]  = -180;          upperBound[0]  = 440;                  
    lowerBound[1]  = .05;           upperBound[1]  = 15;                  
    lowerBound[2]  = .1;            upperBound[2]  = 4;                    
    lowerBound[3]  = .1;            upperBound[3]  = 6;                    
    lowerBound[4]  = 0;             upperBound[4]  = 1;                    
    lowerBound[5]  = -1;            upperBound[5]  = 0;                   
    lowerBound[6]  = -3;            upperBound[6]  = 1;                
    #    lowerBound[6]  = 2;             upperBound[7]  = 2;                 
    lowerBound[7]  = 1;             upperBound[7]  = 10;                 
    lowerBound[8]  = 1e-3;          upperBound[8]  = 1e9;                  
    lowerBound[9] = 0;              upperBound[9] = 1;                
    lowerBound[10] = 0;             upperBound[10] = 100;                
    lowerBound[11] = 10e-3;         upperBound[11] = 10e1;                
    lowerBound[12] = -.35;          upperBound[12] = .35;                

    # Some useful values
    oriPref = mode(S['sfm']['exp']['trial']['ori'][0]).mode * numpy.pi/180;
    prefSf_trials = S['sfm']['exp']['trial']['con'][0] == 0.01;
    sfPref  = numpy.unique(S['sfm']['exp']['trial']['sf'][0][prefSf_trials]);
    rMax    = max(S['sfm']['exp']['oriRateMean']);
    
    # Now fit LN-LN model
    for iR in range(nMultiStarts):

        # Clear memory
        if bestNLL:
            del bestNLL;

        # Set fit options
#        if (iR/2 == round(iR/2)):
#            options = optimset('Display', 'iter', 'Maxiter', 20, 'MaxFuneval', 1000, 'Algorithm', 'sqp');
#        else:
#            options = optimset('Display', 'iter', 'Maxiter', 20, 'MaxFuneval', 1000, 'Algorithm', 'interior-point');
#        end

        # Define the objective function, set the startvalues, perform the fit
        print('\n \n')
        print("Fitting model for " + dataList['unitName'][iU]);
        cd(functionPath)

    #    if linear == 1
    #        obFun = SFMGiveBof_linear(params, S);
    #    else
    #        obFun = @(params) SFMGiveBof(params, S);
    #    end

        try:
            if linear == 1:
                bestNLL    = S['sfm']['mod']['fit_lin']['NLL']; # Check for previously saved fit outcome
                bestParams = numpy.reshape(S['sfm']['mod']['fit_lin']['params'], [1, len(lowerBound)]);
            else:
                bestNLL    = S['sfm']['mod']['fit']['NLL']; # Check for previously saved fit outcome
                bestParams = numpy.reshape(S['sfm']['mod']['fit']['params'], [1, len(lowerBound)]);
            print('Current best fit is ' + str(bestNLL));

            if iR == 0:
                if linear == 1:
                    startvalues = numpy.reshape(S['sfm']['mod']['fit_lin']['params'], [1, len(lowerBound)]);
                else:
                    startvalues = numpy.reshape(S['sfm']['mod']['fit']['params'], [1, len(lowerBound)]);
            else:
                startvalues = [bestParams[0:1], bestParams[2:-1]*(.5+random(len(bestParams[2:-1])))];
        except:
            print('Previous fit statistics not found.');
            startvalues = [oriPref, sfPref, 2, 2, 0.5, -0.05, 0, 3.0, rMax*10, 0.1, 0.1, 0.1, 0];

        try:
            opteemize = minimize(SFMGiveBof, startvalues, method='Nelder-Mead', \
                                 args=(S,), options={'xtol': 1e-4, 'disp': True, 'maxiter': 20});
            
            #pdb.set_trace();
            
            NLL = opteemize['fun'];
            modelParams = opteemize['x'];
            print('Result (NLL current | NLL best)? ' + str(NLL) + ' || ' + str(bestNLL));
            modelParams[0] = 180*(1+scipy.signal.sawtooth(math.pi/180 * modelParams[0])); 
            # Ensure that preferred orientation is expressed between 0 and 360

            # Store outcome if better than previous best outcome
            if bestNLL and NLL: # i.e. if they both exist
                if NLL < bestNLL:
                    print('Better fit found!');
                    if linear == 1:
                              S['sfm']['mod']['fit_lin']['NLL']    = NLL;
                              S['sfm']['mod']['fit_lin']['params'] = modelParams;
                    else:
                              S['sfm']['mod']['fit']['NLL']    = NLL;
                              S['sfm']['mod']['fit']['params'] = modelParams;            

                    cd(loadPath)
                    numpy.save(loadNameSfm, S);
                    print('Saving fit model for ' + N['unitName'][iU]);
                else:
                    print('No better fit found...');

            else:
                if linear == 1:
                    S['sfm']['mod']['fit_lin']['NLL']   = NLL;
                    S['sfm']['mod']['fit_lin']['params'] = modelParams;
                else:
                    S['sfm']['mod']['fit']['NLL'] = NLL;
                    S['sfm']['mod']['fit']['params'] = modelParams;          

                cd(loadPath)
                numpy.save(loadNameSfm, S);
                print(['Saving fit model for ' + N['unitName'][iU]])

        except:
            print('error, fit failed')


    cd(currentPath)

