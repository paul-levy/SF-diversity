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
### I. Ringach, 2004 model - in the below section, all default values are from Ringach (2004) unless otherwise noted
####### loc_x: give X coordinates of RGC location
####### loc_y: give Y ...
####### rgc_lattice: create the full RGC lattice 
####### lgn_lattice: given the RGC locations, resample (as specified) to create LGN lattice
####### v1_cell: creates an example V1 neuron; will create RGC/LGN lattice if not passed in; otherwise, just samples existing lattice
########

def loc_x(sc_factor=1/2.0, lmda=1, sig_pos=0.155, is_vector=1):
  ''' lmda (lambda) is 1; sig_pos is 0.155; sc_factor=1 in Ringach, 2004
      -- but sc_factor=1/2 gives better approx. of results in that paper
      default is to vectorize the function (for multiple i,j), but can be returned for individual i,j values
   '''
  if is_vector == 1:
    return lambda i, j: lmda*sc_factor*(np.add(i,j)) + np.random.normal(loc=0, scale=lmda*sig_pos, size=len(i));
  else:
    return lambda i, j: lmda*sc_factor*(np.add(i,j)) + np.random.normal(loc=0, scale=lmda*sig_pos, size=1);

def loc_y(sc_factor=1/2.0, lmda=1, sig_pos=0.155, is_vector=1):
  ''' lmda (lambda) is 1; sig_pos is 0.155; sc_factor=1 in Ringach, 2004
      -- but sc_factor=1/2 gives better approx. of results in that paper
      default is to vectorize the function (for multiple i,j), but can be returned for individual i,j values
   '''
  if is_vector == 1:
    return lambda i, j: lmda*sc_factor*np.sqrt(3)*(np.subtract(i,j)) + np.random.normal(loc=0, scale=lmda*sig_pos, size=len(i));
  else:
    return lambda i, j: lmda*sc_factor*np.sqrt(3)*(np.subtract(i,j)) + np.random.normal(loc=0, scale=lmda*sig_pos, size=1);

def rgc_lattice(steps, sc_factor=1/2.0, lmda=1, sig_pos=0.155, is_vector=1):
  ''' steps: +/- # integer steps for "i" and "j"
      -- returns the lists for on, off coordinate pairs
  '''
  # how will we compute the x and y coordinates?
  lx = loc_x(sc_factor, lmda, sig_pos, is_vector);
  ly = loc_y(sc_factor, lmda, sig_pos, is_vector);

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
      returns: list of (x,y) coordinate pairs for LGN layer, and on/off indices
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

  return LGN_locs, LGN_ids, on_inds, off_inds;

def v1_cell(v1_loc = None, LGN_locs=None, LGN_ids=None, nsteps=12, lmda=1, ctrMult=0.7, pmax=0.85, connMult=0.97, smax=1, synMult=1.1, relToLmda=1, near_thresh=3):
  ''' The default values here (as above) are from Ringach (2004)
      - pick a V1 location - then, compute all of the LGN neurons with a RF whose distance is less than "near_thresh" from 
          that V1 location
      - from there, we compute if the cell is connected (based on "p" eq. above) and assign a strength (from "s" above)
      --- note: if LGN_locs/LGN_ids are None, we'll compute a default-parameter LGN lattice
      --- note: sig_ctr (size of LGN fields), sig_conn (std specyfing fall-off of conxn prob.), sig_syn (as sig_conn, but for syn. strength) 
                 are given relative to lmda as default (but can be specified otherwise, in which case we pass in the values directly)
      --- note: near_thresh (within what distance do we consider connections), bndsMult (sample the resultant lattice out to near_thresh*bndsMult units),
                 and n_samps have been determined by trial-and-error to work well in most use cases
  '''
  
  ### overhead - set up the parameters which determine the 
  if relToLmda == 1: # default
    sig_ctr = ctrMult*lmda;
    sig_conn = connMult*sig_ctr;
    sig_syn = synMult*sig_ctr;
  else: # we just pass in the values from there
    sig_ctr = ctrMult;
    sig_conn = connMult;
    sig_syn = synMult;

  # sum(dist, 1) means we can pass in list of X and Y coordinates
  gauss = lambda x, y, scale, sig: scale*np.exp(-(np.sum(np.square(x-y), 1))/(2*np.square(sig)))

  p = lambda x, y: gauss(x, y, pmax, sig_conn) # probability of connection
  s = lambda x, y: gauss(x, y, smax, sig_syn) # strength of connection
  
  ### create LGN lattice if needed
  if LGN_locs is None or LGN_ids is None: # then create a default-setting lattice
    onloc, offloc = rgc_lattice(steps=nsteps, lmda=lmda);
    LGN_locs, LGN_ids, _, _ = lgn_lattice(onloc, offloc);

  ### set V1 location, if not already specified
  if v1_loc == None:
    # -- first, random location
    rand_dist = 0.4*nsteps*np.random.random(); # 0.4 is just a useful distance (stay away from edges)
    rand_phi = 2*np.pi*np.random.random()-np.pi; # random angle!
    # now, put into cartesian coordinates
    xcr, ycr = rand_dist*np.cos(rand_phi), rand_dist*np.sin(rand_phi)
    v1_loc = [xcr, ycr];

  ### now, create the neuron!
  # -- compute the distance and find which LGN neurons are near
  curr_dist = np.sqrt(np.sum(np.square(v1_loc - LGN_locs), 1));
  LGN_near = np.where(curr_dist < near_thresh)[0]

  # -- compute the connection probability, see which are connected, and compute sign/weight of those connected ones
  conxn_prob = p(v1_loc, LGN_locs[LGN_near]);

  which_connected = np.where(np.random.binomial(1, conxn_prob))[0];
  not_connected = np.setdiff1d(np.arange(len(LGN_near)), which_connected);
  which_on  = np.where(LGN_ids[LGN_near[which_connected]]==1);
  which_off = np.where(LGN_ids[LGN_near[which_connected]]==0);

  on_locs = LGN_locs[LGN_near[which_connected[which_on]]];
  off_locs = LGN_locs[LGN_near[which_connected[which_off]]];

  on_str = s(v1_loc, on_locs);
  off_str = s(v1_loc, off_locs);

  locations = (on_locs, off_locs);
  syn_strs = (on_str, off_str);

  return locations, syn_strs;

def v1_sample(locs, syn_strs, bndsMult=70, n_samps=1001):
  ''' with a V1 neuron created, now sample it to determine inputs/tuning
  '''
