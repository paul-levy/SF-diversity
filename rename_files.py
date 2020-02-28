import os, sys

def rename_files(loc_data, fileExt='.xml', mBase='m6'):
  # used for renaming files within a directory to pad numbers up to 2 digits
  # - i.e. m675p4r9 to m675p04r09

  files = os.listdir(loc_data);

  for i in files:
#     if i.find('#') >= 0:
#         os.rename(loc_data + i, loc_data + i.replace('#', ''))
#         print('IGNORE: renaming %s to %s' % (loc_data + i, loc_data + i.replace('#', '')))

    if i.find(mBase) >=0 and i.find(fileExt) >= 0: # .mat; or change to .xml/.exxd if changing names in /recordings

        ## updating unit number ([r/l]##)
        try:
          r_ind = i.find('r'); # if updating r (unit number)
          if r_ind < 0:
            r_ind = i.find('l')
          pEnd_ind = i.find('#') # if changing in /recordings/;
  #        rEnd_ind = i.find('_') # if changing in /structures/
          substr_to_replace = i[r_ind+1:pEnd_ind]
          #print('substr: %s' % substr_to_replace)
          new_str = i[0:r_ind+1] + '%02d' % int(substr_to_replace) + i[pEnd_ind:]
        except:
          new_str = i;

        ## updating penetration number (p##)
        try:
          p_ind = new_str.find('p'); # if updating p (penetration number)
          try:
            pEnd_ind = new_str.find('r') # if changing in /recordings/; will be "r" or "l"
          except:
            pEnd_ind = new_str.find('l') # if changing in /recordings/; will be "r" or "l"
          substr_to_replace = new_str[p_ind+1:pEnd_ind]
          #print('substr: %s' % substr_to_replace)
          new_str2 = new_str[0:p_ind+1] + '%02d' % int(substr_to_replace) + new_str[pEnd_ind:]
        except:
          new_str2 = new_str;

        ## finally, the part that applies to both!
        if new_str2 == i:
            continue;

        #os.rename(loc_data + i, loc_data + new_str)
        print('renaming %s to %s' % (i, new_str2))

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Need one argument, at least! Folder to look into');
    
    folder = sys.argv[1];
    try:
      fileExt = sys.argv[2];
    except:
      fileExt = '.xml';
    try:
      mBase = sys.argv[3];
    except:
      mBase = 'm6';
      
 
    rename_files(folder, fileExt, mBase);
