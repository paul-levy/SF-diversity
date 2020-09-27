import numpy as np
import helper_fcns

### Similar to helper_fcns, but meant specifically for the sfBB_* series of experiments

def get_maskOnly_resp(expInfo):
  ''' return the DC, F1 matrices [mean, s.e.m.] for responses to the mask only in the sfBB_* series 
  '''

  byTrial = expInfo['trial'];  

  # make a 3d matrix of mask responses - SF x CON x [mean, SEM]
  maskCon, maskSf = expInfo['maskCon'], expInfo['maskSF'];
  respMatrixDC = np.nan * np.zeros((len(maskCon), len(maskSf), 2));
  respMatrixF1 = np.nan * np.zeros((len(maskCon), len(maskSf), 2));

  # first, the logical which gives mask-only trials
  maskOnly = np.logical_and(byTrial['maskOn'], ~byTrial['baseOn']);

  for mcI, mC in enumerate(maskCon):
      conOk = (byTrial['con'][0,:] == mC)
      for msI, mS in enumerate(maskSf):
          sfOk = (byTrial['sf'][0,:] == mS)
          trialsOk = np.logical_and(maskOnly, np.logical_and(conOk, sfOk));

          currDC, currF1 = expInfo['spikeCounts'][trialsOk], expInfo['f1_mask'][trialsOk];
          dcMean, f1Mean = np.mean(currDC), np.mean(currF1)
          respMatrixDC[mcI, msI, :] = [dcMean, np.std(currDC)/len(currDC)]
          respMatrixF1[mcI, msI, :] = [f1Mean, np.std(currF1)/len(currF1)]

  return respMatrixDC, respMatrixF1;


