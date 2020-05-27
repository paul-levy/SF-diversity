import helper_fcns as hf
import re
import glob
import os

import pdb

dirs = ['V1/', 'V1_orig/', 'altExp/']; # which experiments are we grabbing from?
to_check = ['structures/', 'recordings/']; # which subdirs are we looking in for files to copy?
file_ending = ['*.npy', '*sfMix*.xml']; # in structures/, we copy .npy; in recordings/, we copy .xml
splits = ['_sfm', '_|#']; # for structures/, split just based on _sfm; for recordings, first "_" or "#"

output_name = 'scp_list_200507.txt'

with open(output_name, 'w') as f: # overwite any existing file...
  for dr in dirs:
    pth = '%sstructures/' % dr;
    dl = hf.np_smart_load(pth + hf.get_datalist(dr));

    names = dl['unitName'];
    for nm in names:
      for subdir, ending, to_split in zip(to_check, file_ending, splits):
        splt = re.split('%s' % to_split, nm)[0]; # split the string on _ OR # to get the root of the file (i.e. animal#, penetration#, cell#)
        to_add = glob.glob('%s%s%s%s' % (dr, subdir, splt, ending));
        [f.write(x + '\n') for x in to_add]
