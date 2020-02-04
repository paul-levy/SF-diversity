import numpy as np
import helper_fcns as hf
from scipy.ndimage import gaussian_filter as gauss_filt
import itertools
import os

import pdb

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
####### v1_sample: sample the v1_cell to get the evaluated field and inputs
####### analyze_filt: given a V1 cell, make the fourier transform of the spatial RF; find the peak response; measure ori/sf tuning
#######

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
      - RETURN: the locations and strengths of the connected neurons (evaluating is a separate function, "v1_sample") AND prms
      -           prms.sig_ctr/prms.sig_conn/prms.sig_syn/prms.lmda/prms.near_thresh
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

  prms = dict();
  prms['lmda'] = lmda;
  prms['sig_ctr'] = sig_ctr;
  prms['sig_conn'] = sig_conn;
  prms['sig_syn'] = sig_syn;
  prms['near_thresh'] = near_thresh;
  prms['v1_loc'] = v1_loc;
  prms['gauss'] = gauss;

  return locations, syn_strs, prms;

def v1_sample(locs, syn_strs, prms, bndsMult=70, n_samps=1001):
  ''' with a V1 neuron created, now sample it to determine inputs/tuning
  '''
  # unpack the needed parameters from prms
  sig_ctr = prms['sig_ctr'];
  near_thresh = prms['near_thresh'];
  v1_loc = prms['v1_loc'];
  gauss = prms['gauss'];

  # set up the coordinates
  bnds = bndsMult*near_thresh;
  xs = np.linspace(-bnds, bnds, n_samps) + v1_loc[0];
  ys = np.linspace(-bnds, bnds, n_samps) + v1_loc[1];
  xcoor, ycoor = np.meshgrid(xs, ys);
  all_coords = np.transpose(np.vstack((xcoor.ravel(), ycoor.ravel())));

  lgn_field = np.zeros(len(all_coords), )
  on_sum = np.zeros_like(lgn_field);
  off_sum = np.zeros_like(lgn_field)
  on_fields = [];
  off_fields = [];
  all_fields = (on_fields, off_fields);
  ids = (1, 0); # on--off

  for all_field, lcs, syn_str, id_curr in zip(all_fields, locs, syn_strs, ids):
    for i in range(len(lcs)):
      curr_loc = lcs[i];
      curr_field = gauss(curr_loc, all_coords, syn_str[i], sig_ctr)

      all_field.append(curr_field);

      if id_curr == 1:
        on_sum = on_sum + curr_field;
      elif id_curr == 0:
        off_sum = off_sum + curr_field;

  sums = (on_sum, off_sum);
  tot_field = sums[0] - sums[1]; # i.e. subtract the "off" from the "on"

  return tot_field, sums, all_fields;

################

def analyze_filt(v1_field, lgn_field, n_samps, bounds):
  ''' given the (1D) V1 field (and n_samps/bounds), compute the 2D Fourier transform
      - from this, we can compute:
      -- pref. ori, ori. tuning
      -- pref. sf, sf. tuning
      -- LGN tuning evaluated at the same SF/ORI values
      -- reconstruct the (spatial) V1 field as a check
  '''

  v1_plot = np.reshape(v1_field, (n_samps, n_samps)); # rearrange the field in a 2D plot
  ft = np.fft.fft2(v1_plot)
  # - and what frequencies?? well, [-n_samps/2, n_samps/2]*freq_scale
  samp_step = np.round(2*bounds/n_samps, 3);
  freq_scale = samp_step;
  freqs = freq_scale*np.linspace(-(n_samps/2), n_samps/2, n_samps)
  xfreq, yfreq = np.meshgrid(freqs, freqs);
  v1_ft = np.abs(np.fft.fftshift(ft));
  filt_peak = np.argmax(v1_ft);
  xmax, ymax = np.unravel_index(filt_peak, v1_ft.shape, order='F')
  xfr_peak, yfr_peak = freqs[xmax], freqs[ymax];
  peak_sf = np.sqrt(np.square(xfr_peak) + np.square(yfr_peak));
  peak_ori = np.mod(np.round(hf.angle_xy([xfr_peak], [yfr_peak])[0], 3), 180);

  #####
  # What is the tuning? i.e. SF and ORI?
  #####
  all_freqs = np.transpose(np.vstack((xfreq.ravel(), yfreq.ravel())));
  # -- first, get the valid ori
  oris = np.mod(hf.angle_xy(all_freqs[:, 0], all_freqs[:,1]), 180); # with no D.S., mod by 180
  val_oris = np.where(np.mod(np.abs(oris-peak_ori), 180)<1)[0]; # np.deg2rad(5e-2) if radians
  # -- then, the valid sfs
  sfs = np.sqrt(np.square(all_freqs[:, 0])+np.square(all_freqs[:, 1]))
  val_sfs = np.where(np.abs(sfs-peak_sf)<0.5*samp_step)[0];

  #####
  ## NOW -- plot sf tuning measurements (fixed ori)
  #####
  sf_bound = freq_scale*np.maximum(15*peak_sf, .55*freqs[-1]); # heuristic...should be updated
  plt_oris = val_oris[np.where(sfs[val_oris]<1.25*sf_bound)]
  sf_vals = sfs[plt_oris]
  sf_curve = v1_ft[[x for x in np.unravel_index(plt_oris, v1_ft.shape, order='C')]]
  sf_order = np.argsort(sf_vals);
  sf_resp_norm = np.divide(sf_curve[sf_order], np.max(sf_curve));
  ## -- and get the LGN filter tuning, too
  indiv_shp = np.reshape(lgn_field, v1_plot.shape) # lgn_field is just any of the LGN fields - they're identical!
  indiv_ft = np.fft.fft2(indiv_shp)
  indiv_plot = np.abs(np.fft.fftshift(indiv_ft));
  # ---- and evaluate at the same locations
  lgn_curve = indiv_plot[[x for x in np.unravel_index(plt_oris, v1_ft.shape, order='F')]]
  lgn_resp_norm = np.divide(lgn_curve[sf_order], np.max(lgn_curve));
  #####
  ## THEN -- plot ori tuning measurements (fixed SF)
  #####
  core_oris = np.where(np.array([np.round(freqs[y], 3) for x,y in [np.unravel_index(d, v1_ft.shape, order='F') for d in val_sfs]])>=0);
  plt_sfs = val_sfs[core_oris]
  ori_vals = oris[plt_sfs]
  ori_curve = v1_ft[[x for x in np.unravel_index(plt_sfs, v1_ft.shape, order='C')]]
  ori_order = np.argsort(ori_vals);
  ori_resp_norm = np.divide(ori_curve[ori_order], np.max(ori_curve));

  #####
  # organize the tuning
  #####
  sf_tune = dict();
  sf_tune['peak'] = peak_sf;
  sf_tune['curve'] = sf_resp_norm;
  sf_tune['sf_vals'] = sf_vals[sf_order]
  sf_tune['ori_vals'] = plt_oris; # why? well, the SF tuning is evaluated at a particular orientation - which values?
  ori_tune = dict();
  ori_tune['peak'] = peak_ori;
  ori_tune['curve'] = ori_resp_norm;
  ori_tune['ori_vals'] = ori_vals[ori_order]
  ori_tune['sf_vals'] = plt_sfs; # why? well, the ORI tuning is evaluated at a particular SF - which values?
  # -- and, don't forget, the LGN tuning!
  lgn_tune = dict();
  lgn_tune['sf_vals'] = sf_vals[sf_order];
  lgn_tune['curve'] = lgn_resp_norm;

  # --- show that we can reconstruct it back into the spatial domain
  ift = (np.fft.ifft2(ft))
  v1_reconstruct = np.real(np.sign(ift))*np.abs(ift);

  return v1_ft, sf_tune, ori_tune, lgn_tune, v1_reconstruct;
