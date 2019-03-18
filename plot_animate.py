import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # to avoid GUI/cluster issues...
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import matplotlib.animation as anim
import matplotlib.cm as cm
import seaborn as sns
sns.set(style='ticks');
import itertools
import helper_fcns as hf
# also, fix warnings so they don't repeat
import warnings
warnings.filterwarnings('once')
# import "partial"
from functools import partial # https://alexgude.com/blog/matplotlib-blitting-supernova/

import pdb

# Before any plotting, fix plotting paramaters
plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');
from matplotlib import rcParams
rcParams['font.size'] = 20;
rcParams['pdf.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['ps.fonttype'] = 42 # should be 42, but there are kerning issues
rcParams['lines.linewidth'] = 2.5;
rcParams['axes.linewidth'] = 1.5;
rcParams['lines.markersize'] = 5;
rcParams['font.style'] = 'oblique';
# NOTE: Why do you need to specify plt.rc instead of just rc..? Dunno, since all the above work without specifying plt.rc
# But, I spent several hours attempting to fix this; I realized the only meaningful difference between the original version
# (coded in a .ipynb, working) and the "production" version was this
plt.rcParams['animation.ffmpeg_path'] = '/users/plevy/miniconda3/bin/ffmpeg' # path to ffmpeg binary

## fixed constants
conDig = 3; # round contrast to the 3rd digit
# at CNS
basePath = '/arc/2.2/p1/plevy/SF_diversity/sfDiv-OriModel/sfDiv-python/';
# personal mac
# basePath = '/Users/paulgerald/work/sfDiversity/sfDiv-OriModel/sfDiv-python/';

## get input
which_cell  = int(sys.argv[1]);
lossType = int(sys.argv[2]);
expDir   = sys.argv[3]; 
fitType  = int(sys.argv[4]);
desired_dur = int(sys.argv[5]); # in seconds, for the whole animation
stepsize = int(sys.argv[6]); # in frames, i.e. how many optimization steps between adjacent frames

dataPath = basePath + expDir + 'structures/'
save_loc = basePath + expDir + 'figures/'

dataList = np.load(dataPath + 'dataList.npy', encoding='latin1').item();
expInd, expType = hf.get_exp_ind(dataPath, dataList['unitName'][which_cell-1])

fitBase = 'fitList_190226c';
# first the fit type
if fitType == 1:
  fitSuf = '_flat';
elif fitType == 2:
  fitSuf = '_wght';
# then the loss type
if lossType == 1:
  lossSuf = '_sqrt_details.npy';
elif lossType == 2:
  lossSuf = '_poiss_details.npy';
elif lossType == 3:
  lossSuf = '_modPoiss_details.npy';
elif lossType == 4:
  lossSuf = '_chiSq_details.npy';
# load fit details
fitName = str(fitBase + fitSuf + lossSuf);
fitDetails = hf.np_smart_load(dataPath + fitName)
fitDet = fitDetails[which_cell-1];
# load rvc/ph fits, if applicable
rvcFits = hf.get_rvc_fits(dataPath, expInd, which_cell);
phFits = hf.get_rvc_fits(dataPath, expInd, which_cell, rvcName='phaseAdvanceFits');
# load cell structure
cellStruct = hf.np_smart_load(dataPath + dataList['unitName'][which_cell-1] + '_sfm.npy');
data = cellStruct['sfm']['exp']['trial'];

# also, get save directory
compDir  = str(fitBase + '_anim' + lossSuf.replace('details', ''));
subDir   = compDir.replace('fitList', 'fits').replace('.npy', '');
save_loc = str(save_loc + subDir + '/');
if not os.path.exists(save_loc):
  os.makedirs(save_loc);

### set up 
nstepsTot = len(fitDet['loss'])
nFrames = np.int(np.ceil(nstepsTot/stepsize))

## indices for accessing parameters
[prefSf, dOrder, normConst, respExp, normMu, normStd] = [0, 1, 2, 3, 8, 9];
finalParams = fitDet['params'][-1]; # i.e. parameters at end of optimization

## reshape the model/exp responses by condition, group by dispersion
# we reshape the responses to combine within dispersion, i.e. (nDisp, nSf*nCon)
shapeByDisp = lambda resps: resps.reshape((resps.shape[0], resps.shape[1]*resps.shape[2]));
measured_resps = hf.organize_resp(data['spikeCount'], cellStruct, expInd)[2] # 3rd output is organized sfMix resp.
measured_byDisp = shapeByDisp(measured_resps)
nDisps = len(measured_byDisp);

## get the final filter tunings
omega = np.logspace(-2, 2, 1000); # where are we evaluating?
# first, normalization
inhSfTuning = hf.getSuppressiveSFtuning(sfs=omega);
nInhChan = cellStruct['sfm']['mod']['normalization']['pref']['sf'];
nTrials =  inhSfTuning.shape[0];
if fitType == 2:
    gs_mean, gs_std = [finalParams[normMu], finalParams[normStd]]
    inhWeight = hf.genNormWeights(cellStruct, nInhChan, gs_mean, gs_std, nTrials, expInd);
    inhWeight = inhWeight[:, :, 0]; # genNormWeights gives us weights as nTr x nFilters x nFrames - we have only one "frame" here, and all are the same                                                                                                                           
    # first, tuned norm:
    sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
    sfNorm = sfNorm/np.amax(np.abs(sfNorm));
    # update function to be used below
    updateInhWeight = lambda mn, std: hf.genNormWeights(cellStruct, nInhChan, mn, std, nTrials, expInd)[:,:,0];
else:
    # then, untuned norm:
    inhAsym = 0;
    inhWeight = [];
    for iP in range(len(nInhChan)):
        inhWeight = np.append(inhWeight, 1 + inhAsym * (np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]) - np.mean(np.log(cellStruct['sfm']['mod']['normalization']['pref']['sf'][iP]))));
    sfNorm = np.sum(-.5*(inhWeight*np.square(inhSfTuning)), 1);
    sfNorm = sfNorm/np.amax(np.abs(sfNorm));
    # update function to be used below
    updateInhWeight = lambda mn, std: inhWeight; # it's a dummy f'n to keep consistency across conditions
# the update function
updatesfNorm    = lambda mn, std: np.sum(-.5*(updateInhWeight(mn, std)*np.square(inhSfTuning)), 1)
updatesfNormFinal = lambda asArr: updatesfNorm(asArr[0], asArr[1])/np.amax(np.abs(updatesfNorm(asArr[0], asArr[1])))
# then excitatory
sfExc = [];
pSF, dOrd = [finalParams[prefSf], finalParams[dOrder]];
sfRel = omega/pSF;
s     = np.power(omega, dOrd) * np.exp(-dOrd/2 * np.square(sfRel));
sMax  = np.power(pSF, dOrd) * np.exp(-dOrd/2);
sfExc = s/sMax;
# write the update functions
updateS = lambda pSF, dOrd, xval: np.power(xval, dOrd) * np.exp(-dOrd/2 * np.square(xval/pSF));
updatesfExc = lambda pSF, dOrd, xval: updateS(pSF, dOrd, xval) / (np.power(pSF, dOrd) * np.exp(-dOrd/2))
updatesfExcFinal = lambda asArr: updatesfExc(asArr[0], asArr[1], omega); # just dummy to plug in omega as xval

sfTuning = [sfExc, sfNorm];
sfTuningUpdates = [updatesfExcFinal, updatesfNormFinal]
nsfTuning = len(sfTuning);

## compute the response non-linearity curve
# last but not least...and not last... response nonlinearity
modExp = finalParams[respExp];
nlinIn = np.linspace(-1,1,100);
nlinPlot = lambda xval, respExp: np.power(np.maximum(0, xval), respExp);
nlinFinal = nlinPlot(nlinIn, modExp);
### set up (end of)

### the real plotting
step = 0;

disp_colors = cm.rainbow(np.linspace(0, 1, len(measured_byDisp)))
disp_labels = ['disp: %d' % (i+1) for i in range(nDisps)]
# indices for accessing the proper plots
loss_start = 0;
disp_start = 1;
filt_start = disp_start + nDisps;
nlin_start = filt_start + 1; # nonlinear properties
norm_start = nlin_start + 1;
tune_start = norm_start + 1;
nlinc_start = tune_start + len(sfTuning);

## figure initialization
nrows = 4;
ncols = 2;
f, ax = plt.subplots(nrows, ncols, figsize=(12*nrows, 16*ncols)) # moved to top

## create/initialize artists
# loss
loss, = ax[0,0].semilogy([], [], 'ro', markersize=20, animated=True)
# responses
measured_byDisp = shapeByDisp(measured_resps)
resps = [ax[0,1].plot(x, x, 'o', color=c, label=l, animated=True) for x,c,l in zip(measured_byDisp,disp_colors,disp_labels)]
# filter properties
filt, = ax[1,0].semilogx([], [], 'ro', markersize=20, animated=True)
# non-linear properties
nlin, = ax[1,1].plot([], [], 'ro', markersize=20, animated=True)
# normalization
norm, = ax[2,0].semilogx([], [], 'ro', markersize=20, animated=True)
# important parameters in text - not doing for now...
# filter tunings
tune_labels = ['exc', 'norm']; tune_colors = ['k', 'r']
tune = [ax[3,0].semilogx(omega, curve, label=l, color=c, animated=True) for curve,l,c in zip(sfTuning, tune_labels, tune_colors)]
# response nonlinearity (curve)
nlin_curve, = ax[3,1].plot(nlinIn,nlinFinal, 'r-', animated=True)
## combine the lists of plot/line elements
to_update = [loss] + hf.flatten_list(resps) + [filt] + [nlin] + [norm] + hf.flatten_list(tune) + [nlin_curve]; 

## init function
def init_fig(fig, ax, artists): # plot static things on fig/ax; artists are just passed through

    # plot the loss
    ax[0, 0].semilogy(range(nstepsTot), fitDet['loss'], 'k-', animated=False)
    ax[0, 0].set_xlabel('optimization step');
    ax[0, 0].set_ylabel('loss value')
    ax[0, 0].set_title('Loss')
    # plot the response in each trial
    ax[0, 1].plot([0, np.nanmax(measured_resps)], [0, np.nanmax(measured_resps)], 'k--', animated=False)
    ax[0, 1].set_xlabel('measured response');
    ax[0, 1].set_ylabel('model response')
    ax[0, 1].axis('equal')
    ax[0, 1].legend();
    ax[0, 1].set_title('Responses')
    # plot filter properties
    ax[1, 0].semilogx([x[prefSf] for x in fitDet['params']], [x[dOrder] for x in fitDet['params']], 'k-', animated=False)
    ax[1, 0].set_xlabel('preferred spatial frequency (cpd)');
    ax[1, 0].set_ylabel('derivate order')
    ax[1, 0].set_xlim([1e-1, 1e1])
    ax[1, 0].set_title('Filter properties')
    # plot non-linear properties
    ax[1, 1].plot([np.power(10, x[normConst]) for x in fitDet['params']], [x[respExp] for x in fitDet['params']], 'k-', animated=False)
    ax[1, 1].set_xlabel('normalization constant');
    ax[1, 1].set_ylabel('response exponent')
    ax[1, 1].set_title('Non-linear properties')
    # plot normalization properties
    if len(fitDet['params'][0])>9: # doesn't matter which "step", all have same # params
        ax[2, 0].semilogx([np.exp(x[normMu]) for x in fitDet['params']], [x[normStd] for x in fitDet['params']], 'k-', animated=False)
        ax[2, 0].set_xlabel('pool mean (cpd)');
        ax[2, 0].set_ylabel('pool std')
        ax[2, 0].set_xlim([1e-1, 1e1])
        l_bound = hf.getConstraints(fitType=2)[normStd][0]; # lower bound on norm std when using weighted fit
        ax[2, 0].semilogx([1e-1, 1e1], [l_bound, l_bound], 'k--')
        ax[2, 0].set_title('Normalization properties')
    # writing out parameters in text - not now
    ax[2,1].axis('off')
    # filter tunings
    ax[3,0].set_xlim([omega[0], omega[-1]]); # limits
    ax[3,0].set_ylim([-1.1, 1.1]);
    ax[3,0].semilogx([omega[0], omega[-1]], [0, 0], 'k--') # reference lines
    ax[3,0].semilogx([.01, .01], [-1.5, 1], 'k--')
    ax[3,0].semilogx([.1, .1], [-1.5, 1], 'k--')
    ax[3,0].semilogx([1, 1], [-1.5, 1], 'k--')
    ax[3,0].semilogx([10, 10], [-1.5, 1], 'k--')
    ax[3,0].semilogx([100, 100], [-1.5, 1], 'k--')
    ax[3,0].set_xlabel('spatial frequency (c/deg)');
    ax[3,0].set_ylabel('Normalized response (a.u.)');
    ax[3,0].set_title('Filter tuning')
    ax[3,0].legend();
    # response nonlinearity (curve)
    ax[3,1].plot([-1, 1], [0, 0], 'k--')
    ax[3,1].plot([0, 0], [-.1, 1], 'k--')
    ax[3,1].plot(nlinIn, nlinPlot(nlinIn, 1), 'k--', linewidth=1) # i.e. response exponent of 1
    ax[3,1].set_xlim([-1, 1]);
    ax[3,1].set_ylim([-.1, 1]);
    ax[3,1].axis('equal');
    ax[3,1].set_title('Response non-linearity')
    # adjust subplots, make overall title
    fig.subplots_adjust(wspace=0.2, hspace=0.3);
    try:
        cellStr = dataList['unitType'][which_cell-1]
    except:
        cellStr = dataList['unitArea'][which_cell-1]
    fig.suptitle('%s #%d (in %s)' % (cellStr, which_cell, expDir));
    
    # despine...
    for r, c in itertools.product(range(nrows), range(ncols)):
        ax[r,c].tick_params(width=2, length=16, direction='out'); # major
        ax[r,c].tick_params(width=2, length=8, direction='out', which='minor'); # minor
        sns.despine(ax=ax[r,c], offset=10, trim=False);
        
    return artists;

## animation
def updatefig(frames, artists): # ignore frames; artists will be to_update (i.e. the animated bits)
    global step
    if (step<nstepsTot):
        step += stepsize
    if step>=nstepsTot:
        step=nstepsTot-1

    # update loss
    artists[loss_start].set_data(step, fitDet['loss'][step])
    # update model responses
    resps_org = hf.organize_resp(fitDet['resp'][step], cellStruct, expInd)[2]
    [artists[disp_start + i].set_data(data,mod) for i,data,mod in zip(range(nDisps), measured_byDisp, shapeByDisp(resps_org)) ]
    # get current params
    params_curr = fitDet['params'][step]
    # update filter properties
    artists[filt_start].set_data(params_curr[prefSf], params_curr[dOrder])
    # update non_linear properties
    artists[nlin_start].set_data(np.power(10, params_curr[normConst]), params_curr[respExp])
    # update normalization properties
    if len(params_curr)>9:
        artists[norm_start].set_data(np.exp(params_curr[normMu]), params_curr[normStd])
    # update text parameters
    # update tuning curves
    if fitType==2:
        normPrm = [params_curr[normMu], params_curr[normStd]]
    else:
        normPrm = [0, 0]; # dummy vars
    excPrm = [params_curr[prefSf], params_curr[dOrder]]
    updatePrm = [excPrm, normPrm]
    [artists[tune_start + i].set_data(omega, lam(prm)) for i,lam, prm in zip(range(nsfTuning), sfTuningUpdates, updatePrm)]
    # update resp non-linearity curve
    artists[nlinc_start].set_data(nlinIn, nlinPlot(nlinIn, params_curr[respExp]))   
    return artists;

## Set up animation
init_f = partial(init_fig, fig=f, ax=ax, artists=to_update)
update_f = partial(updatefig, artists=to_update)
# interval is time between frames, in milliseconds
ani = anim.FuncAnimation(fig=f, func=update_f, init_func=init_f, frames=nFrames, blit=True, interval=20, repeat_delay=1500)

Writer = anim.writers['ffmpeg']
writer = Writer(fps=np.int(np.floor(nFrames/desired_dur)), bitrate=-1)
save_name = save_loc + 'cell_%02d%s.mp4' % (which_cell, fitSuf);
ani.save(save_name, writer=writer)
