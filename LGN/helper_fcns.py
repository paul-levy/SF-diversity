import numpy as np

# blankResp - return mean/std of blank responses (i.e. baseline firing rate) for Sach's experiment
# tabulateResponses - Organizes measured and model responses for Sach's experiment

def blankResp(data):
  blanks = np.where(data['cont'] == 0);

  mu = np.mean(data['f0'][blanks]);
  std = np.std(data['f0'][blanks]);

  return mu, std;

def tabulateResponses(data):
  ''' Given the dictionary containing all of the data, organize the data into the proper responses
  Specifically, we know that Sach's experiments varied contrast and spatial frequency
  Thus, we will organize responses along these dimensions
  '''
  all_cons = np.unique(data['cont']);
  all_cons = all_cons[all_cons>0];
  all_sfs = np.unique(data['sf']);

  f0 = dict();
  f0mean= np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f0sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1 = dict();
  f1mean = np.nan * np.zeros((len(all_cons), len(all_sfs))); 
  f1sem = np.nan * np.zeros((len(all_cons), len(all_sfs))); 

  
  for con in range(len(all_cons)):
    val_con = np.where(data['cont'] == all_cons[con]);
    for sf in range(len(all_sfs)):
      val_sf = np.where(data['sf'][val_con] == all_sfs[sf]);

      f0mean[con, sf] = data['f0'][val_con][val_sf];
      f0sem[con, sf] = data['f0sem'][val_con][val_sf];
      f1mean[con, sf] = data['f1'][val_con][val_sf];
      f1sem[con, sf] = data['f1sem'][val_con][val_sf];

  f0['mean'] = f0mean;
  f0['sem'] = f0sem;
  f1['mean'] = f1mean;
  f1['sem'] = f1sem;

  return [f0, f1], [all_cons, all_sfs];
