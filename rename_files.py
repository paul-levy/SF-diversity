import pdb
import os, sys

def rename_files(loc_data, files, fileExt='.xml', mBase='m6', trialRun=1, isDataList=False):
  # used for renaming files within a directory to pad numbers up to 2 digits
  # - i.e. m675p4r9 to m675p04r09
  # can also be used to rename files in a dataList if fileExt='' and files is the list of names in the dataList
  # trialRun: if True, just print what would be the renaming, but don't do it
  #           if False, then actually rename

  for ind, i in enumerate(files):
#     if i.find('#') >= 0:
#         os.rename(loc_data + i, loc_data + i.replace('#', ''))
#         print('IGNORE: renaming %s to %s' % (loc_data + i, loc_data + i.replace('#', '')))

    if i.find(mBase) >=0 and i.find(fileExt) >= 0: # .mat; or change to .xml/.exxd if changing names in /recordings

        ########
        # first, figure out where the meaningful part of the program name ends
        ########
        if isDataList: # why treat separately? well, some dataList entries have m###pN_blahblah, others are m#pN#Z
          endInd = i.find('_');
          if endInd == -1: # i.e.
            endInd = i.find('#') # if there isn't '_', then find '#'
            if endInd == -1:
              endInd = len(i); # then, just get the end
        else:
          if 'structures' in loc_data:
            endInd = i.find('_') # if changing in /structures/
          elif 'recordings' in loc_data:
            endInd = i.find('#') # if changing in /recordings/;
          else:
            print('uhoh...are you sure you are renaming the right directory?');
            endInd = 0;


        ## updating unit number ([r/l]##)
        try:
          r_ind = i.find('r'); # if updating r (unit number)
          if r_ind < 0 or r_ind > endInd: # i.e. then there is an R in the program name...
            r_ind = i.find('l')
          substr_to_replace = i[r_ind+1:endInd]
          #print('substr: %s' % substr_to_replace)
          new_str = i[0:r_ind+1] + '%02d' % int(substr_to_replace) + i[endInd:]
        except:
          new_str = i;

        ## updating penetration number (p##)
        try:
          p_ind = new_str.find('p'); # if updating p (penetration number)
          pEnd_ind = new_str.find('r') # if changing in /recordings/; will be "r" or "l"
          if pEnd_ind > endInd or pEnd_ind < 0: # i.e. the "r" is in the program name! or not here at all
            pEnd_ind = new_str.find('l');
          substr_to_replace = new_str[p_ind+1:pEnd_ind]
          #print('substr: %s' % substr_to_replace)
          new_str2 = new_str[0:p_ind+1] + '%02d' % int(substr_to_replace) + new_str[pEnd_ind:]
        except:
          new_str2 = new_str;

        ## finally, the part that applies to both!
        if new_str2 == i:
            continue;
        
        if trialRun == 1:
          print('renaming %s to %s' % (i, new_str2))
        else:
          os.rename(loc_data + i, loc_data + new_str2)

        files[ind] = new_str2;

  return files;

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Need one argument, at least! Folder to look into');
    
    loc_data = sys.argv[1];
    try:
      fileExt = sys.argv[2];
    except:
      fileExt = '.xml';
    try:
      mBase = sys.argv[3];
    except:
      mBase = 'm6';
    try:
      trRun = int(sys.argv[4]);
    except:
      trRun = 1;

    files = os.listdir(loc_data);

    rename_files(loc_data, files, fileExt, mBase, trRun);
