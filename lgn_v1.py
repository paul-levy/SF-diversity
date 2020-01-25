import numpy as np
import helper_fcns as hf
from scipy.ndimage import gaussian_filter as gauss_filt
import itertools
import os

import warnings
warnings.filterwarnings('once')

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pltSave
import matplotlib.cm as cm
import seaborn as sns

plt.style.use('https://raw.githubusercontent.com/paul-levy/SF_diversity/master/paul_plt_style.mplstyle');

#########
# Explanation
#########
# In this .py file, we'll put the code for building, analyzing, and plotting LGN-V1 models. This is a companion
# to the lgn-v1-construction.ipynb file.

# Functions:
########
### I. Ringach, 2004 model
########

def loc_x(sc_factor=1/2.0, lm=1, sig_pos=0.155, is_vector=1):
  ''' lm (lambda) is 1; sig_pos is 0.155; sc_factor=1 in Ringach, 2004
      -- but sc_factor=1/2 gives better approx. of results in that paper
      default is to vectorize the function (for multiple i,j), but can be returned for individual i,j values
   '''
  if is_vector == 1:
    return lambda i, j: lm*sc_factor*(np.add(i,j)) + np.random.normal(loc=0, scale=lm*sig_pos, size=len(i));
  else:
    return lambda i, j: lm*sc_factor*(np.add(i,j)) + np.random.normal(loc=0, scale=lm*sig_pos, size=1);

def loc_y(sc_factor=1/2.0, lm=1, sig_pos=0.155, is_vector=1):
  ''' lm (lambda) is 1; sig_pos is 0.155; sc_factor=1 in Ringach, 2004
      -- but sc_factor=1/2 gives better approx. of results in that paper
      default is to vectorize the function (for multiple i,j), but can be returned for individual i,j values
   '''
  if is_vector == 1:
    return lambda i, j: lm*sc_factor*np.sqrt(3)*(np.subtract(i,j)) + np.random.normal(loc=0, scale=lm*sig_pos, size=len(i));
  else:
    return lambda i, j: lm*sc_factor*np.sqrt(3)*(np.subtract(i,j)) + np.random.normal(loc=0, scale=lm*sig_pos, size=1);

def rgc_lattice(steps, sc_factor=1/2.0, lm=1, sig_pos=0.155, is_vector=1):
  ''' steps: +/- # integer steps for "i" and "j"
      -- returns the lists for on, off coordinate pairs
  '''
  # how will we compute the x and y coordinates?
  lx = loc_x(sc_factor, lm, sig_pos, is_vector);
  ly = loc_y(sc_factor, lm, sig_pos, is_vector);

  # create the sampling grid
  xs = np.arange(-steps, steps+1);
  xcoor, ycoor = np.meshgrid(xs, xs);
  all_coords = np.transpose(np.vstack((xcoor.ravel(), ycoor.ravel())));
  xc, yc = all_coords[:, 0], all_coords[:, 1];

  # sample the ON cells and package into coordinates
  xcs, ycs = lx(xc, yc), ly(xc, yc);
  on_locs = np.hstack((np.expand_dims(xcs, axis=-1), np.expand_dims(ycs, axis=-1)));

  # sample the OFF cells and package into coordinates 
  xcs, ycs = lx(xc, yc), ly(xc, yc);
  off_locs = np.hstack((np.expand_dims(xcs, axis=-1), np.expand_dims(ycs, axis=-1)));

  return on_locs, off_locs;

def lgn_lattice(on_locs, off_locs, resample_factor=0):
  ''' given RGC on/off locations, resample the locations N*"resample_factor" times and add to the existing RGC locs to make the LGN layer
      -- in ringach, 2004, resample_factor is 1.5 (s.t. LGN has 2.5x RGC neurons)
      -- but that is for cat, and in primate, we want 0x!
      returns: list of on,off coordinate pairs for LGN layer
  '''

  all_locs = np.vstack((on_locs, off_locs));
  all_ids = np.hstack((np.ones((len(on_locs), )), np.zeros((len(off_locs), ))));

  n_RGC = len(all_locs);
  n_to_draw = np.int(resample_factor*n_RGC) 

  inds = np.random.randint(low=0, high=n_RGC, size=n_to_draw)
  new_locs = all_locs[inds];
  new_ids = all_ids[inds];

  LGN_locs = np.vstack((all_locs, new_locs))
  LGN_ids = np.hstack((all_ids, new_ids))

  on_inds = np.where(LGN_ids == 1)[0];
  off_inds = np.where(LGN_ids == 0)[0];

  return LGN_locs[on_inds], LGN_locs[off_inds];
