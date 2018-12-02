from helper_fcns import naka_rushton, fit_CRF, random_in_range, blankResp, np_smart_load
import os.path
import sys
import math, numpy
from scipy.stats import norm, mode, poisson
from scipy.stats.mstats import gmean as geomean
import scipy.optimize as opt
import pdb

def fit_all_CRF(cell_num, data_loc, each_c50, loss_type, n_iter = 1, each_expn = 0, each_base = 0, each_gain = 1):
    ''' Given cell#, data loc, load the data. Other inputs:
          each_c50/expn/base/gain : separate c50/expn/base/gain for each condition?
          n_iter                  : how many iterations to fit?
          
    '''
    print(str(n_iter) + ' fit attempts');
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    n_params = 5; # 4 for NR, 1 for varGain

    if each_c50 == 1:
      fit_key = 'fits_each_rpt';
    else:
      fit_key = 'fits_rpt';

    if loss_type == 1:
      loss_str = '-lsq';
    if loss_type == 2:
      loss_str = '-sqrt';
    if loss_type == 3:
      loss_str = '-poiss';
    if loss_type == 4:
      loss_str = '-poissMod';
    fits_name = 'crfFitsCom' + loss_str + '.npy';
    dataList = np_smart_load(str(data_loc + 'dataList.npy'));
    if os.path.isfile(data_loc + fits_name):
        crfFits = np_smart_load(str(data_loc + fits_name));
    else:
        crfFits = dict();

    # load cell information
    cellStruct = np_smart_load(str(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy'));
    data = cellStruct['sfm']['exp']['trial'];

    all_cons = np.unique(np.round(data['total_con'], conDig));
    all_cons = all_cons[~np.isnan(all_cons)];

    all_sfs = np.unique(data['cent_sf']);
    all_sfs = all_sfs[~np.isnan(all_sfs)];

    all_disps = np.unique(data['num_comps']);
    all_disps = all_disps[all_disps>0]; # ignore zero...

    nCons = len(all_cons);
    nSfs = len(all_sfs);
    nDisps = len(all_disps);

    nk_ru = dict();
    all_data = dict();

    # for use in fitting SF functions...
    _, _, blankResps = blankResp(cellStruct);    
    blankCons = np.zeros_like(blankResps);

    for d in range(nDisps):
        valid_disp = data['num_comps'] == all_disps[d];
	cons = [];
	resps = [];

	nk_ru[d] = dict();
	v_sfs = []; # keep track of valid sfs
	all_data[d] = dict();
        
	for sf in range(nSfs):

           valid_sf = data['cent_sf'] == all_sfs[sf];

           valid_tr = valid_disp & valid_sf;
           if np.all(np.unique(valid_tr) == False): # did we not find any trials?
	        continue;
	
	   v_sfs.append(sf);
	   nk_ru[d][sf] = dict(); # create dictionary here; thus, only valid sfs have valid keys
           # for unpacking loss/parameters later...
           nk_ru[d][sf]['params'] = np.nan * np.zeros((n_params, 1));
           nk_ru[d][sf]['loss'] = np.nan;

	   resps.append(np.hstack((blankResps, data['spikeCount'][valid_tr])));
           cons.append(np.hstack((blankCons, data['total_con'][valid_tr])));

        # save data for later use
        all_data[d]['resps'] = resps;
        all_data[d]['cons'] = cons;
        all_data[d]['valid_sfs'] = v_sfs;
    
        maxResp = np.max(np.max(resps));
	n_v_sfs = len(v_sfs);

        each_list = (each_base, each_gain, each_expn, each_c50);

        n_per_param = [1 if i == 0 else n_v_sfs for i in each_list];
        '''
    	if each_c50 == 1:
    	  n_c50s = n_v_sfs; # separate for each SF...
    	else:
	  n_c50s = 1;	
        '''
        init_base = 0.1;
        #bounds_base = (0, 0);
        bounds_base = (0.1, maxResp);
        init_gain = np.max(resps) - np.min(resps);
        bounds_gain = (0, 10*maxResp);
        init_expn = 2;
        bounds_expn = (0.5, 10);
        init_c50 = 0.1; #geomean(all_cons);
        bounds_c50 = (0.01, 10*max(all_cons)); # contrast values are b/t [0, 1]
        init_varGain = 1;
        bounds_varGain = (0.01, None);

 	base_inits = np.repeat(init_base, n_per_param[0]); # default is only one baseline per SF
	base_constr = [tuple(x) for x in np.broadcast_to(bounds_base, (n_per_param[0], 2))]
 	
	gain_inits = np.repeat(init_gain, n_per_param[1]); # gain is always separate for each SF
	gain_constr = [tuple(x) for x in np.broadcast_to(bounds_gain, (n_per_param[1], 2))]

	expn_inits = np.repeat(init_expn, n_per_param[2]); # exponent can be either, like baseline
	expn_constr = [tuple(x) for x in np.broadcast_to(bounds_expn, (n_per_param[2], 2))]
 		
	c50_inits = np.repeat(init_c50, n_per_param[3]); # repeat n_v_sfs times if c50 separate for each SF; otherwise, 1
	c50_constr = [tuple(x) for x in np.broadcast_to(bounds_c50, (n_per_param[3], 2))]

        init_params = np.hstack((c50_inits, expn_inits, gain_inits, base_inits, init_varGain));
    	boundsAll = np.vstack((c50_constr, expn_constr, gain_constr, base_constr, bounds_varGain));
    	boundsAll = [tuple(x) for x in boundsAll]; # turn the (inner) arrays into tuples...

        c50_ind = 0;
	expn_ind = n_per_param[3]; # the number of c50s...
	gain_ind = expn_ind+n_per_param[2]; # the number of exponents
	base_ind = gain_ind+n_per_param[1]; # always n_v_sfs gain parameters
        varGain_ind = base_ind+n_per_param[0];

	obj = lambda params: fit_CRF(cons, resps, params[c50_ind:c50_ind+n_per_param[3]], params[expn_ind:expn_ind+n_per_param[2]], params[gain_ind:gain_ind+n_per_param[1]], \
                                     params[base_ind:base_ind+n_per_param[0]], params[varGain_ind], loss_type);
	opts = opt.minimize(obj, init_params, bounds=boundsAll);

	curr_params = opts['x'];
	curr_loss = opts['fun'];

	for iter in range(n_iter-1): # now, extra iterations if chosen...
	  init_params = np.hstack((random_in_range(bounds_c50, n_c50s), random_in_range(bounds_expn), random_in_range(bounds_gain, n_v_sfs), random_in_range(bounds_base), random_in_range((bounds_varGain[0], 1))));

          # choose optimization method
          if np.mod(iter, 2) == 0:
             methodStr = 'L-BFGS-B';
          else:
             methodStr = 'TNC';

	  opt_iter = opt.minimize(obj, init_params, bounds=boundsAll, method=methodStr);

          if opt_iter['fun'] < curr_loss:
	    print('improve.');
	    curr_loss = opt_iter['fun'];
	    curr_params = opt_iter['x'];

        # now unpack...
        for sf_in in range(n_v_sfs):
          param_ind = [0 if i == 1 else sf_in for i in n_per_param];

          nk_ru[d][v_sfs[sf_in]]['params'][0] = curr_params[base_ind + param_ind[0]];
          nk_ru[d][v_sfs[sf_in]]['params'][1] = curr_params[gain_ind + param_ind[1]];
          nk_ru[d][v_sfs[sf_in]]['params'][2] = curr_params[expn_ind + param_ind[2]];
          nk_ru[d][v_sfs[sf_in]]['params'][3] = curr_params[c50_ind + param_ind[3]];
          # params (to match naka_rushton) are: baseline, gain, expon, c50
          nk_ru[d][v_sfs[sf_in]]['params'][4] = curr_params[varGain_ind];
          nk_ru[d][v_sfs[sf_in]]['loss'] = curr_loss

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + fits_name):
      print('reloading CRF Fits...');
      crfFits = np_smart_load(str(data_loc + fits_name));
    if cell_num-1 not in crfFits:
      crfFits[cell_num-1] = dict();
    crfFits[cell_num-1][fit_key] = nk_ru;
    crfFits[cell_num-1]['data'] = all_data;
    crfFits[cell_num-1]['blankResps'] = blankResps;

    np.save(data_loc + fits_name, crfFits);
    print('saving for cell ' + str(cell_num));

    return nk_ru;

def fit_all_CRF_boot(cell_num, data_loc, each_c50, loss_type, n_boot_iter = 1000):
    np = numpy;
    conDig = 3; # round contrast to the thousandth
    n_params = 5; # 4 for NR, 1 for varGain

    if each_c50 == 1:
	fit_key = 'fits_each';
    else:
	fit_key = 'fits';

    if loss_type == 1:
      loss_str = '-lsq';
    if loss_type == 2:
      loss_str = '-sqrt';
    if loss_type == 1:
      loss_str = '-poiss';
    if loss_type == 1:
      loss_str = '-poissMod';
    fits_name = 'crfFits' + loss_str + '.npy';

    dataList = np_smart_load(str(data_loc + 'dataList.npy'));
    if os.path.isfile(data_loc + fits_name):
        crfFits = np_smart_load(str(data_loc + fits_name));
    else:
        crfFits = dict();
    
    cellStruct = np_smart_load(str(data_loc + dataList['unitName'][cell_num-1] + '_sfm.npy'));
    data = cellStruct['sfm']['exp']['trial'];

    all_cons = np.unique(np.round(data['total_con'], conDig));
    all_cons = all_cons[~np.isnan(all_cons)];

    all_sfs = np.unique(data['cent_sf']);
    all_sfs = all_sfs[~np.isnan(all_sfs)];

    all_disps = np.unique(data['num_comps']);
    all_disps = all_disps[all_disps>0]; # ignore zero...

    nCons = len(all_cons);
    nSfs = len(all_sfs);
    nDisps = len(all_disps);

    nk_ru = dict();
    nk_ru_boot = dict();
    all_data = dict();

    for d in range(nDisps):
        valid_disp = data['num_comps'] == all_disps[d];
	cons = [];
	resps = [];

	nk_ru[d] = dict();
	nk_ru_boot[d] = dict();
	all_data[d] = dict();
	v_sfs = []; # keep track of valid sfs

        for sf in range(nSfs):

           valid_sf = data['cent_sf'] == all_sfs[sf];

           valid_tr = valid_disp & valid_sf;
           if np.all(np.unique(valid_tr) == False): # did we not find any trials?
	        continue;
	
	   v_sfs.append(sf);

	   nk_ru[d][sf] = dict(); # create dictionary here; thus, only valid sfs have valid keys
	   nk_ru_boot[d][sf] = dict(); # create dictionary here; thus, only valid sfs have valid keys
		# for unpacking loss/parameters later...
	   nk_ru[d][sf]['params'] = np.nan * np.zeros((n_params, 1));
	   nk_ru[d][sf]['loss'] = np.nan;
	   nk_ru_boot[d][sf]['params'] = np.nan * np.zeros((n_boot_iter, n_params));
	   nk_ru_boot[d][sf]['loss'] = np.nan * np.zeros((n_boot_iter, 1));

           resps.append(data['spikeCount'][valid_tr]);
           cons.append(data['total_con'][valid_tr]);

	# save data for later use
	all_data[d]['resps'] = resps;    
	all_data[d]['cons'] = cons;
	all_data[d]['valid_sfs'] = v_sfs;

        maxResp = np.max(np.max(resps));
	n_v_sfs = len(v_sfs);

    	if each_c50 == 1:
    	  n_c50s = n_v_sfs; # separate for each SF...
    	else:
	  n_c50s = 1;	

        init_base = 0.1;
        bounds_base = (0.01, maxResp);
        init_gain = np.max(resps) - np.min(resps);
        bounds_gain = (0, 10*maxResp);
        init_expn = 2;
        bounds_expn = (1, 10);
        init_c50 = 0.1; #geomean(all_cons);
        bounds_c50 = (0.01, 10*max(all_cons)); # contrast values are b/t [0, 1]
	init_varGain = 1;
        bounds_varGain = (0.01, None);

 	base_inits = np.repeat(init_base, 1); # only one baseline per SF
	base_constr = [tuple(x) for x in np.broadcast_to(bounds_base, (1, 2))]
 	
	gain_inits = np.repeat(init_gain, n_v_sfs); # ...and gain
	gain_constr = [tuple(x) for x in np.broadcast_to(bounds_gain, (n_v_sfs, 2))]
 		
	c50_inits = np.repeat(init_c50, n_c50s); # repeat n_v_sfs times if c50 separate for each SF; otherwise, 1
	c50_constr = [tuple(x) for x in np.broadcast_to(bounds_c50, (n_c50s, 2))]

        init_params = np.hstack((c50_inits, init_expn, gain_inits, base_inits, init_varGain));
    	boundsAll = np.vstack((c50_constr, bounds_expn, gain_constr, base_constr, bounds_varGain));
    	boundsAll = [tuple(x) for x in boundsAll]; # turn the (inner) arrays into tuples...

	expn_ind = n_c50s;
	gain_ind = n_c50s+1;
	base_ind = gain_ind+n_v_sfs; # only one baseline per dispersion...
	varGain_ind = base_ind+1;

        # first, fit original dataset
        obj = lambda params: fit_CRF(cons, resps, params[0:n_c50s], params[expn_ind], params[gain_ind:gain_ind+n_v_sfs], params[base_ind], params[varGain_ind], loss_type);
        opts_full = opt.minimize(obj, init_params, bounds=boundsAll);

        # now unpack...
        params = opts_full['x'];
        for sf_in in range(n_v_sfs):
          if n_c50s == 1:
            c50_ind = 0;
          else:
            c50_ind = sf_in;

          nk_ru[d][v_sfs[sf_in]]['params'][0] = params[base_ind];
          nk_ru[d][v_sfs[sf_in]]['params'][1] = params[gain_ind+sf_in];
          nk_ru[d][v_sfs[sf_in]]['params'][2] = params[expn_ind];
          nk_ru[d][v_sfs[sf_in]]['params'][3] = params[c50_ind];
          # params (to match naka_rushton) are: baseline, gain, expon, c50
          nk_ru[d][v_sfs[sf_in]]['params'][4] = params[varGain_ind];
          nk_ru[d][v_sfs[sf_in]]['loss'] = opts_full['fun'];

	# then, bootstrap resample
	for boot_i in range(n_boot_iter):
	  resamp_resps = [];
	  resamp_cons = [];

	  # resample the data
	  for sf_i in range(len(resps)):
	    resamp_inds = numpy.random.randint(0, len(resps[sf_i]), len(resps[sf_i]));
	    resamp_resps.append(resps[sf_i][resamp_inds]);
	    resamp_cons.append(cons[sf_i][resamp_inds]);

	  obj = lambda params: fit_CRF(resamp_cons, resamp_resps, params[0:n_c50s], params[expn_ind], params[gain_ind:gain_ind+n_v_sfs], params[base_ind], params[varGain_ind], loss_type);
	  opts = opt.minimize(obj, init_params, bounds=boundsAll);

          # now unpack...
	  params = opts['x'];
	  for sf_in in range(n_v_sfs):
            if n_c50s == 1:
              c50_ind = 0;
            else:
              c50_ind = sf_in;
            nk_ru_boot[d][v_sfs[sf_in]]['params'][boot_i, 0] = params[base_ind];
            nk_ru_boot[d][v_sfs[sf_in]]['params'][boot_i, 1] = params[gain_ind+sf_in];
            nk_ru_boot[d][v_sfs[sf_in]]['params'][boot_i, 2] = params[expn_ind];
            nk_ru_boot[d][v_sfs[sf_in]]['params'][boot_i, 3] = params[c50_ind];
	      # params (to match naka_rushton) are: baseline, gain, expon, c50
            nk_ru_boot[d][v_sfs[sf_in]]['params'][boot_i, 4] = params[varGain_ind];
	    nk_ru_boot[d][v_sfs[sf_in]]['loss'][boot_i] = opts['fun'];

    # update stuff - load again in case some other run has saved/made changes
    if os.path.isfile(data_loc + fits_name):
      print('reloading CRF Fits...');
      crfFits = np_smart_load(str(data_loc + fits_name));
    if cell_num-1 not in crfFits:
      crfFits[cell_num-1] = dict();
    crfFits[cell_num-1][fit_key] = nk_ru;
    crfFits[cell_num-1]['data'] = all_data;
    crfFits[cell_num-1][str(fit_key + '_boot')] = nk_ru_boot;

    np.save(data_loc + fits_name, crfFits);
    print('saving for cell ' + str(cell_num));

    return nk_ru;

if __name__ == '__main__':

    if len(sys.argv) < 4:
      print('uhoh...you need at least three arguments here');
      print('First is cell number, second is if c50 is fixed [0] or free [1] for each SF, third is loss_type (i.e. loss function), fourth [optional] is number of bootstrap iterations [default is 1000]');
      exit();

    print('Running cell ' + sys.argv[1] + '...');

    dataset = int(sys.argv[2]);

    if dataset == 1:
      data_loc = '/home/pl1465/SF_diversity/altExp/analysis/structures/';
    elif dataset == 2:
      data_loc = '/home/pl1465/SF_diversity/LGN/analysis/structures/';

    if len(sys.argv) > 5: # specify number of fit iterations
      print(' with ' + sys.argv[4] + ' iterations');
      fit_all_CRF(int(sys.argv[1]), data_loc, int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]));
    else: # all trials in each iteration
      fit_all_CRF(int(sys.argv[1]), data_loc, int(sys.argv[3]), int(sys.argv[4]));

