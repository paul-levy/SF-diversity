import subprocess
import numpy as np
import glob
import pdb
import os

def prog_name(filename):
  # using expo naming convention of mNpX#Y[progName].xml, return progName
  st, en = filename.find('['), filename.find(']');
  return filename[st+1:en];

def prog_root_str():
  return ['sf1', 'rvc1', 'tf1', 'rfsize', 'ori1'];

def find_basics(loc, root_name):
  # what are the root programs that we will look for?
  prog_root = prog_root_str();
  prog_files = [];

  rel_files = glob.glob('%s%s#*xml' % (loc, root_name));
  
  for ind, basic in enumerate(prog_root):
    not_found = True;
    for f in rel_files:
      if basic in f and not_found:
        prog_files.append(os.path.abspath(f));
        not_found = False;
    if not_found:
      prog_files.append('');

  return prog_files;

def simplify_name(name, reduceNums=True):
  # split the string into alpha and numeric characters
  # - if reduceNums is True, simplify things back to original naming scheme (i.e. remove padding)
  # - ^ this is used on the original machines (e.g. /v/acute/) because I pad numbers on work machines

  name = name.split('_')[0]; # some v1 cells have _glx### or _c### -- ignore that and only keep what comes before
  
  numer = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in name);
  orig_nums = [int(i) for i in numer.split()]

  alpha = ''.join((ch if ch not in '0123456789.-e' else ' ') for ch in name);
  alphas = ['%s' %s for s in alpha.split()]
  if reduceNums:
    # now, we take advantage of the fact that it's always m#r# or m#l#p# (i.e. alphaNUMalphaNUM...)
    base = ['%s%d' % (i,j) for i,j in zip(alphas, orig_nums)]
    full = ''.join(base);
  else:
    full = name;
  
  # then, if there is a '#' in this, remove it!
  if '#' in full:
    ind = full.find('#');
    full = full[0:ind]; # meaning, get rid of the # in this!
    
  return full, orig_nums;

def build_basic_lists(datalist_names, expExt, loc='./', subfolder='recordings/', folderByExpt=True, reduceNums=True):
  # returns a list of lists (sublist i has the basic charac. programs for cell i in datalist), and flattened list for copying via scp

  expInd = -1; # used for counting which subfolder to access, if needed
  curr_exp_num = -1;
  
  basics = [];
  for dn in datalist_names:
    # a quick processing, since on local machine, we have changed mN[r/l]%02d, but it's just *%d on this machine
    dn_trim, nums = simplify_name(dn, reduceNums); # nums[0] will be the m# to get
    if nums[0] != curr_exp_num:

      curr_exp_num = nums[0];
      expInd = expInd + 1;
      
    if type(subfolder) == list:
      subfold_curr = subfolder[expInd];
    else:
      subfold_curr = subfolder;
    
    if folderByExpt: # i.e. on /v/acute/ data server (groma)
      basics.append(find_basics(loc + 'm%d%s/%s' % (nums[0], expExt, subfold_curr), dn_trim));
    else: # i.e. on personal account (e.g. on sesto)
      basics.append(find_basics(loc + '%s' % (subfold_curr), dn_trim));

  basics_flat = [item for sublist in basics for item in sublist]
  # now, let's "trim" basics_flat to remove empty strings (i.e. non-found programs)
  blanks = np.where(~np.in1d(basics_flat, ''));
  basics_flat_trim = np.array(basics_flat)[blanks];
  
  prog_root_order = prog_root_str();

  return basics, basics_flat, basics_flat_trim, prog_root_order;

def batch_scp(to_copy, dest):
  # use this to scp from zemina to the destination (key exists for plevy@sesto.cns.nyu.edu)
  for tc in to_copy:
    subprocess.call(["scp", tc, dest]);
