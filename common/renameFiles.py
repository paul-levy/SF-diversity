import sys
import os
import re
import warnings
import pdb

def rename_files(folder):
    # currently: pads to 2 digits!
    all_files = os.listdir(os.path.abspath(folder));
    for f in all_files:
      check = re.search(r'\d+', f);
      if check is None:
        continue;
      poss_ints = check.group();
      new_name = f.replace(poss_ints, '%02d' % int(poss_ints));
      os.rename(folder + f, folder + new_name);

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Need one argument! Folder to look into');
    
    folder = sys.argv[1];
 
    rename_files(folder);
